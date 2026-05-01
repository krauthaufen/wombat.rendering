// Reactive vertex attributes: when the outer
// `aval<HashMap<string, aval<BufferView>>>` flips between maps with
// different inner BufferView identities, the prepared-RO observes the
// new buffers without rebuilding the pipeline. When the new map drops
// a shader-required name, record() warn-once+skips.

import { describe, expect, it, vi } from "vitest";
import {
  AList, AdaptiveToken, AVal, HashMap, cval, transact, type aval,
} from "@aardworx/wombat.adaptive";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import {
  IBuffer,
  RenderTree,
  type BufferView,
  type Command,
  type DrawCall,
  type RenderObject,
  PipelineState,
} from "@aardworx/wombat.rendering/core";
import { allocateFramebuffer, createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { Runtime } from "@aardworx/wombat.rendering/runtime";
import { makeEffect } from "./_makeEffect.js";
import { MockGPU } from "./_mockGpu.js";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

function singleAttribEffect() {
  return makeEffect(
    `
      function vsMain(input: { position: V3f }): { gl_Position: V4f } {
        return { gl_Position: new V4f(input.position.x, input.position.y, input.position.z, 1.0) };
      }
      function fsMain(_input: {}): { outColor: V4f } {
        return { outColor: new V4f(1.0, 0.0, 0.0, 1.0) };
      }
    `,
    [
      { name: "vsMain", stage: "vertex",
        inputs: [{ name: "position", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] }],
      },
      { name: "fsMain", stage: "fragment",
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ],
  );
}

function bv(label?: string): aval<BufferView> {
  return cval<BufferView>({
    buffer: { ...IBuffer.fromHost(new ArrayBuffer(36)), label } as IBuffer,
    offset: 0, count: 3, stride: 12, format: "float32x3",
  });
}

describe("RenderObject vertexAttributes — reactive map", () => {
  it("flipping the outer map to a new buffer set does not rebuild the pipeline", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device });
    const eff = singleAttribEffect();

    const mapA = HashMap.empty<string, aval<BufferView>>().add("position", cval<BufferView>({
      buffer: IBuffer.fromHost(new ArrayBuffer(36)),
      offset: 0, count: 3, stride: 12, format: "float32x3",
    }));
    // mapB has a LARGER buffer so the AdaptiveBuffer must reallocate
    // (different GPU handle observable in setVertexBufferCalls).
    const mapB = HashMap.empty<string, aval<BufferView>>().add("position", cval<BufferView>({
      buffer: IBuffer.fromHost(new ArrayBuffer(360)),
      offset: 0, count: 30, stride: 12, format: "float32x3",
    }));
    const outer = cval<HashMap<string, aval<BufferView>>>(mapA);

    const obj: RenderObject = {
      effect: eff,
      pipelineState: PipelineState.constant({ rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } }),
      vertexAttributes: outer,
      uniforms: HashMap.empty(),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };

    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();
    const task = runtime.compile(AList.ofArray<Command>([
      { kind: "Render", output: fbo, tree: RenderTree.leaf(obj) },
    ]));
    task.run(AdaptiveToken.top);
    const pipelinesAfter1 = gpu.pipelines.length;
    expect(pipelinesAfter1).toBe(1);

    // Capture the buffers used in the first frame's setVertexBuffer
    // calls. Different identity expected after the flip.
    const setVB1 = gpu.renderPasses.flatMap(p => p.setVertexBufferCalls);
    expect(setVB1.length).toBeGreaterThan(0);
    const firstBuf = setVB1[setVB1.length - 1]!.buffer;

    // Flip the outer map → an entirely new HashMap with a new inner aval.
    transact(() => { outer.value = mapB; });
    task.run(AdaptiveToken.top);

    // Pipeline cache stays at 1 — buffer-identity swaps do not feed
    // the pipeline cache key.
    expect(gpu.pipelines.length).toBe(pipelinesAfter1);

    const setVB2 = gpu.renderPasses[gpu.renderPasses.length - 1]!.setVertexBufferCalls;
    expect(setVB2.length).toBeGreaterThan(0);
    const secondBuf = setVB2[setVB2.length - 1]!.buffer;
    // Distinct host buffers → distinct GPU buffers.
    expect(secondBuf).not.toBe(firstBuf);

    task.dispose();
    fbo.release();
  });

  it("set shrinks below shader's required vertex inputs → warn-once + skip", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device });
    const eff = singleAttribEffect();

    const mapOk = HashMap.empty<string, aval<BufferView>>().add("position", bv());
    const mapMissing = HashMap.empty<string, aval<BufferView>>(); // no "position"
    const outer = cval<HashMap<string, aval<BufferView>>>(mapOk);

    const obj: RenderObject = {
      effect: eff,
      pipelineState: PipelineState.constant({ rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } }),
      vertexAttributes: outer,
      uniforms: HashMap.empty(),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };

    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();
    const task = runtime.compile(AList.ofArray<Command>([
      { kind: "Render", output: fbo, tree: RenderTree.leaf(obj) },
    ]));
    task.run(AdaptiveToken.top);

    const drawsBefore = gpu.renderPasses[gpu.renderPasses.length - 1]!.drawCalls.length
      + gpu.renderPasses[gpu.renderPasses.length - 1]!.drawIndexedCalls.length;
    expect(drawsBefore).toBe(1);

    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    transact(() => { outer.value = mapMissing; });
    task.run(AdaptiveToken.top);
    task.run(AdaptiveToken.top);
    const lastPass = gpu.renderPasses[gpu.renderPasses.length - 1]!;
    expect(lastPass.drawCalls.length).toBe(0);
    expect(lastPass.drawIndexedCalls.length).toBe(0);
    expect(warn.mock.calls.length).toBe(1);
    warn.mockRestore();

    task.dispose();
    fbo.release();
  });

  it("set grows beyond shader's required inputs → ignored", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device });
    const eff = singleAttribEffect();

    const mapBase = HashMap.empty<string, aval<BufferView>>().add("position", bv());
    const mapExtra = mapBase.add("normal", bv()); // shader does not request "normal"
    const outer = cval<HashMap<string, aval<BufferView>>>(mapBase);

    const obj: RenderObject = {
      effect: eff,
      pipelineState: PipelineState.constant({ rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } }),
      vertexAttributes: outer,
      uniforms: HashMap.empty(),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();
    const task = runtime.compile(AList.ofArray<Command>([
      { kind: "Render", output: fbo, tree: RenderTree.leaf(obj) },
    ]));
    task.run(AdaptiveToken.top);
    const pipBefore = gpu.pipelines.length;

    transact(() => { outer.value = mapExtra; });
    task.run(AdaptiveToken.top);
    expect(gpu.pipelines.length).toBe(pipBefore);
    // Still drawing.
    const lastPass = gpu.renderPasses[gpu.renderPasses.length - 1]!;
    expect(lastPass.drawCalls.length + lastPass.drawIndexedCalls.length).toBe(1);

    task.dispose();
    fbo.release();
    void AVal;
  });
});
