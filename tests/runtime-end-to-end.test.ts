// Runtime end-to-end: alist<Command> with Clear + Render + Copy +
// Custom commands → encoded into a single GPUCommandEncoder via
// the mock. Verifies RenderTree traversal, batching, command order.

import { describe, expect, it } from "vitest";
import {
  AList,
  AdaptiveToken,
  HashMap,
  cval,
  type aval,
} from "@aardworx/wombat.adaptive";
import { V4f } from "@aardworx/wombat.base";
import {
  IBuffer,
  RenderTree,
  type BufferView,
  type ClearValues,
  type Command,
  type CompiledEffect,
  type DrawCall,
  type RenderObject,
} from "@aardworx/wombat.rendering-core";
import {
  allocateFramebuffer,
  createFramebufferSignature,
} from "@aardworx/wombat.rendering-resources";
import { Runtime } from "@aardworx/wombat.rendering-runtime";
import { MockGPU } from "./_mockGpu.js";

function bv(bytes: number, format: GPUVertexFormat, count = 3): aval<BufferView> {
  return cval<BufferView>({
    buffer: IBuffer.fromHost(new ArrayBuffer(bytes)),
    offset: 0,
    count,
    stride: 12,
    format,
  });
}

import type { Type } from "@aardworx/wombat.shader-ir";

const f32:   Type = { kind: "Float", width: 32 };
const vec3f: Type = { kind: "Vector", element: f32, dim: 3 };
const vec4f: Type = { kind: "Vector", element: f32, dim: 4 };

function effect(): CompiledEffect {
  return {
    target: "wgsl",
    stages: [
      { stage: "vertex",   entryName: "main", source: "@vertex fn main(){}",  bindings: [] as never, meta: {} as never, sourceMap: null },
      { stage: "fragment", entryName: "main", source: "@fragment fn main(){}", bindings: [] as never, meta: {} as never, sourceMap: null },
    ],
    interface: {
      target: "wgsl",
      stages: [],
      attributes: [{ name: "position", location: 0, type: vec3f, format: "float32x3", components: 3, byteSize: 12 }],
      fragmentOutputs: [{ name: "color", location: 0, type: vec4f }],
      uniforms: [],
      uniformBlocks: [{
        name: "Globals", group: 0, slot: 0, size: 16,
        fields: [{ name: "tint", type: vec4f, offset: 0, size: 16, align: 16 }],
      }],
      samplers: [], textures: [], storageBuffers: [],
    },
    avalBindings: {},
  };
}

function obj(eff: CompiledEffect): RenderObject {
  return {
    effect: eff as unknown as RenderObject["effect"],
    pipelineState: {
      rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" },
    },
    vertexAttributes: HashMap.empty<string, aval<BufferView>>().add("position", bv(36, "float32x3")),
    uniforms:         HashMap.empty<string, aval<unknown>>().add("tint", cval(new V4f(1, 0, 0, 1))),
    textures: HashMap.empty(),
    samplers: HashMap.empty(),
    drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
  };
}

describe("Runtime.compile(alist<Command>)", () => {
  it("Clear + Render emit two render passes in order", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device, compileEffect: e => e as unknown as CompiledEffect });

    const sig = createFramebufferSignature({ colors: { color: "rgba8unorm" } });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();
    const ifb = fbo.getValue(AdaptiveToken.top);

    const eff = effect();
    const tree = RenderTree.leaf(obj(eff));
    const clearValues: ClearValues = {
      colors: HashMap.empty<string, V4f>().add("color", new V4f(0, 0, 0, 1)),
    };
    const commands = AList.ofArray<Command>([
      { kind: "Clear",  output: ifb, values: clearValues },
      { kind: "Render", output: ifb, tree },
    ]);

    const task = runtime.compile(commands);
    task.run(AdaptiveToken.top);

    expect(gpu.renderPasses).toHaveLength(2);
    // First pass = clear (loadOp:"clear", no draws).
    expect((gpu.renderPasses[0]!.desc.colorAttachments as GPURenderPassColorAttachment[])[0]!.loadOp).toBe("clear");
    expect(gpu.renderPasses[0]!.drawCalls).toHaveLength(0);
    // Second pass = render (loadOp:"load", one draw).
    expect((gpu.renderPasses[1]!.desc.colorAttachments as GPURenderPassColorAttachment[])[0]!.loadOp).toBe("load");
    expect(gpu.renderPasses[1]!.drawCalls).toHaveLength(1);

    task.dispose();
    fbo.release();
  });

  it("Ordered tree of two leaves batches into one render pass", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device, compileEffect: e => e as unknown as CompiledEffect });
    const sig = createFramebufferSignature({ colors: { color: "rgba8unorm" } });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();
    const ifb = fbo.getValue(AdaptiveToken.top);
    const eff = effect();
    const tree = RenderTree.ordered(RenderTree.leaf(obj(eff)), RenderTree.leaf(obj(eff)));
    const cmds = AList.ofArray<Command>([{ kind: "Render", output: ifb, tree }]);
    const task = runtime.compile(cmds);
    task.run(AdaptiveToken.top);
    expect(gpu.renderPasses).toHaveLength(1);
    expect(gpu.renderPasses[0]!.drawCalls).toHaveLength(2);
    task.dispose();
    fbo.release();
  });

  it("Custom command receives the encoder", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device, compileEffect: e => e as unknown as CompiledEffect });
    const seen: GPUCommandEncoder[] = [];
    const cmds = AList.ofArray<Command>([
      { kind: "Custom", encode: (enc) => { seen.push(enc); } },
    ]);
    const task = runtime.compile(cmds);
    task.run(AdaptiveToken.top);
    expect(seen).toHaveLength(1);
    task.dispose();
  });

  it("Copy command emits buffer-to-buffer transfer", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device, compileEffect: e => e as unknown as CompiledEffect });
    const src = { size: 64 } as GPUBuffer;
    const dst = { size: 64 } as GPUBuffer;
    const cmds = AList.ofArray<Command>([
      { kind: "Copy", copy: { kind: "buffer", src, dst, range: { srcOffset: 0, dstOffset: 0, size: 64 } } },
    ]);
    const task = runtime.compile(cmds);
    task.run(AdaptiveToken.top);
    expect(gpu.copyBufferCalls).toEqual([{ src, srcOffset: 0, dst, dstOffset: 0, size: 64 }]);
    task.dispose();
  });
});
