// Per-attribute buffer reactivity: when an inner BufferView aval is
// flipped to a new buffer identity, the prepared-RO observes the new
// buffer without rebuilding the pipeline. The outer attribute map is
// structural (fixed key set) — only inner buffer values are reactive.

import { describe, expect, it } from "vitest";
import {
  AList, AdaptiveToken, HashMap, cval, transact, type aval,
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

describe("RenderObject vertexAttributes — per-buffer reactivity", () => {
  it("flipping an inner BufferView aval to a new buffer does not rebuild the pipeline", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device });
    const eff = singleAttribEffect();

    const viewA: BufferView = {
      buffer: IBuffer.fromHost(new ArrayBuffer(36)),
      offset: 0, count: 3, stride: 12, format: "float32x3",
    };
    // viewB has a LARGER buffer so the AdaptiveBuffer must reallocate
    // (different GPU handle observable in setVertexBufferCalls).
    const viewB: BufferView = {
      buffer: IBuffer.fromHost(new ArrayBuffer(360)),
      offset: 0, count: 30, stride: 12, format: "float32x3",
    };
    const positionView = cval<BufferView>(viewA);
    const map = HashMap.empty<string, aval<BufferView>>().add("position", positionView);

    const obj: RenderObject = {
      effect: eff,
      pipelineState: PipelineState.constant({ rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } }),
      vertexAttributes: map,
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

    const setVB1 = gpu.renderPasses.flatMap(p => p.setVertexBufferCalls);
    expect(setVB1.length).toBeGreaterThan(0);
    const firstBuf = setVB1[setVB1.length - 1]!.buffer;

    // Flip the inner per-attribute aval — same map, new BufferView.
    transact(() => { positionView.value = viewB; });
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
});
