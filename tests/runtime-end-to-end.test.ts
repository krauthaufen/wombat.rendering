// Runtime command-walker invariants: Clear+Render coalescing and
// Ordered tree batching. Both produce specific render-pass shapes
// the mock can inspect; real-GPU tests verify the same scenarios
// at the pixel level. Custom + Copy command paths live in
// tests-browser/runtime-real.test.ts.

import { describe, expect, it } from "vitest";
import {
  AList,
  AdaptiveToken,
    HashMap,
  cval,
  type aval, AVal
} from "@aardworx/wombat.adaptive";
import { V4f } from "@aardworx/wombat.base";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import {
  IBuffer,
  RenderTree,
  type BufferView,
  type ClearValues,
  type Command,
  type DrawCall,
  type RenderObject,
  PipelineState,
  ElementType,
} from "@aardworx/wombat.rendering/core";
import {
  allocateFramebuffer,
  createFramebufferSignature,
} from "@aardworx/wombat.rendering/resources";
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
      function fsMain(_input: {}): V4f {
        return new V4f(1.0, 0.0, 0.0, 1.0);
      }
    `,
    [
      {
        name: "vsMain", stage: "vertex",
        inputs: [
          { name: "position", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        ],
        outputs: [
          { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        ],
      },
      {
        name: "fsMain", stage: "fragment",
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ],
  );
}

function bv(): aval<BufferView> {
  return {
    buffer: AVal.constant(IBuffer.fromHost(new ArrayBuffer(36))),
    offset: 0, stride: 12, elementType: ElementType.V3f,
  };
}

function obj() {
  return {
    effect: singleAttribEffect(),
    pipelineState: PipelineState.constant({ rasterizer: { topology: "triangle-list" as const, cullMode: "none" as const, frontFace: "ccw" as const } }),
    vertexAttributes: HashMap.empty<string, BufferView>().add("position", bv()),
    uniforms: HashMap.empty(),
    textures: HashMap.empty(),
    samplers: HashMap.empty(),
    drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
  } satisfies RenderObject;
}

describe("Runtime.compile(alist<Command>)", () => {
  it("Clear + Render on same FBO coalesces into one pass with loadOp clear", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device, heapEnabled: cval(false) });
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();
    const tree = RenderTree.leaf(obj());
    const clearValues: ClearValues = {
      colors: HashMap.empty<string, V4f>().add("outColor", new V4f(0, 0, 0, 1)),
    };
    const commands = AList.ofArray<Command>([
      { kind: "Clear", values: clearValues },
      { kind: "Render",tree },
    ]);
    const task = runtime.compile(sig, commands);
    task.run(fbo.getValue(AdaptiveToken.top), AdaptiveToken.top);

    // Coalesced: one render pass that both clears and draws.
    expect(gpu.renderPasses).toHaveLength(1);
    const att = (gpu.renderPasses[0]!.desc.colorAttachments as GPURenderPassColorAttachment[])[0]!;
    expect(att.loadOp).toBe("clear");
    expect(att.clearValue).toEqual({ r: 0, g: 0, b: 0, a: 1 });
    expect(gpu.renderPasses[0]!.drawCalls).toHaveLength(1);

    task.dispose();
    fbo.release();
  });

  it("a linear Render command draws through the framebuffer's linear variant, with pipelines of its formats", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device, heapEnabled: cval(false) });
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();
    const plain = fbo.getValue(AdaptiveToken.top);
    const linearView = { __view: "srgb" } as unknown as GPUTextureView;
    const fb = {
      ...plain,
      linear: {
        ...plain,
        signature: { ...sig, colors: sig.colors.add("outColor", "rgba8unorm-srgb" as GPUTextureFormat) },
        colors: plain.colors.add("outColor", linearView),
      },
    };
    const cmds = AList.ofArray<Command>([
      { kind: "Render", tree: RenderTree.leaf(obj()) },
      { kind: "Render", tree: RenderTree.leaf(obj()), linear: true },
    ]);
    const task = runtime.compile(sig, cmds);
    task.run(fb, AdaptiveToken.top);
    expect(gpu.renderPasses).toHaveLength(2);
    const view = (i: number) => (gpu.renderPasses[i]!.desc.colorAttachments as GPURenderPassColorAttachment[])[0]!.view;
    expect(view(0)).toBe(plain.colors.tryFind("outColor"));
    expect(view(1)).toBe(linearView);
    const formats = gpu.pipelines.map((p) => [...(p.fragment!.targets as GPUColorTargetState[])][0]!.format);
    expect(formats).toEqual(["rgba8unorm", "rgba8unorm-srgb"]);
    // Without a linear variant the command draws as any other.
    gpu.renderPasses.length = 0;
    task.run(plain, AdaptiveToken.top);
    expect(view(1)).toBe(plain.colors.tryFind("outColor"));
    task.dispose();
    fbo.release();
  });

  it("Ordered tree of two leaves batches into one render pass", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device, heapEnabled: cval(false) });
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();
    const tree = RenderTree.ordered(RenderTree.leaf(obj()), RenderTree.leaf(obj()));
    const cmds = AList.ofArray<Command>([{ kind: "Render",tree }]);
    const task = runtime.compile(sig, cmds);
    task.run(fbo.getValue(AdaptiveToken.top), AdaptiveToken.top);
    expect(gpu.renderPasses).toHaveLength(1);
    expect(gpu.renderPasses[0]!.drawCalls).toHaveLength(2);
    task.dispose();
    fbo.release();
  });

});
