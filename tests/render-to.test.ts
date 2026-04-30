// renderTo end-to-end: a hidden RenderTask renders a scene into
// an off-screen FBO; the returned aval<ITexture> is consumed by a
// downstream RenderObject (whose textures HashMap binds it). When
// the outer task runs, the inner task encodes itself into the same
// command encoder, producing scene → texture data flow
// automatically.
//
// Lifetime is verified separately: acquiring/releasing the
// returned aval drives the underlying FBO + inner-task lifecycle.

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
  ISampler,
  ITexture,
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

function bv(format: GPUVertexFormat = "float32x3"): aval<BufferView> {
  return cval<BufferView>({
    buffer: IBuffer.fromHost(new ArrayBuffer(36)),
    offset: 0, count: 3, stride: 12, format,
  });
}

import type { Type } from "@aardworx/wombat.shader-ir";

const f32:   Type = { kind: "Float", width: 32 };
const vec3f: Type = { kind: "Vector", element: f32, dim: 3 };
const vec4f: Type = { kind: "Vector", element: f32, dim: 4 };
const tex2d: Type = { kind: "Texture", target: "2D", sampled: { kind: "Float" }, arrayed: false, multisampled: false };
const samp:  Type = { kind: "Sampler", target: "2D", sampled: { kind: "Float" }, comparison: false };

function passEffect(): CompiledEffect {
  return {
    target: "wgsl",
    stages: [
      { stage: "vertex",   entryName: "main", source: "@vertex fn main(){} /*pass*/",  bindings: [] as never, meta: {} as never, sourceMap: null },
      { stage: "fragment", entryName: "main", source: "@fragment fn main(){} /*pass*/", bindings: [] as never, meta: {} as never, sourceMap: null },
    ],
    interface: {
      target: "wgsl",
      stages: [],
      attributes: [{ name: "position", location: 0, type: vec3f, format: "float32x3", components: 3, byteSize: 12 }],
      fragmentOutputs: [{ name: "color", location: 0, type: vec4f }],
      uniforms: [], uniformBlocks: [],
      samplers: [], textures: [], storageBuffers: [],
    },
    avalBindings: {},
  };
}

function texturedEffect(): CompiledEffect {
  return {
    target: "wgsl",
    stages: [
      { stage: "vertex",   entryName: "main", source: "@vertex fn main(){} /*tex*/",  bindings: [] as never, meta: {} as never, sourceMap: null },
      { stage: "fragment", entryName: "main", source: "@fragment fn main(){} /*tex*/", bindings: [] as never, meta: {} as never, sourceMap: null },
    ],
    interface: {
      target: "wgsl",
      stages: [],
      attributes: [{ name: "position", location: 0, type: vec3f, format: "float32x3", components: 3, byteSize: 12 }],
      fragmentOutputs: [{ name: "color", location: 0, type: vec4f }],
      uniforms: [], uniformBlocks: [],
      textures: [{ name: "src",  group: 0, slot: 0, type: tex2d }],
      samplers: [{ name: "samp", group: 0, slot: 1, type: samp }],
      storageBuffers: [],
    },
    avalBindings: {},
  };
}

describe("renderTo", () => {
  it("compose: inner scene renders into FBO, outer scene samples its texture", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({
      device: gpu.device,
      compileEffect: e => e as unknown as CompiledEffect,
    });

    // Inner scene: one triangle, no textures, renders into the offscreen FBO.
    const innerEff = passEffect();
    const innerObj: RenderObject = {
      effect: innerEff as unknown as RenderObject["effect"],
      pipelineState: { rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } },
      vertexAttributes: HashMap.empty<string, aval<BufferView>>().add("position", bv()),
      uniforms: HashMap.empty(),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };
    const innerScene = RenderTree.leaf(innerObj);

    // renderTo → aval<ITexture> for "color" attachment.
    const offscreenSig = createFramebufferSignature({ colors: { color: "rgba8unorm" } });
    const result = runtime.renderTo(innerScene, {
      size: cval({ width: 32, height: 32 }),
      signature: offscreenSig,
      clear: { colors: HashMap.empty<string, V4f>().add("color", new V4f(0, 0, 0, 1)) } as ClearValues,
      label: "offscreen",
    });
    const offscreenColor = result.color("color");

    // Outer scene: one triangle that samples the offscreen color.
    const outerEff = texturedEffect();
    const outerObj: RenderObject = {
      effect: outerEff as unknown as RenderObject["effect"],
      pipelineState: { rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } },
      vertexAttributes: HashMap.empty<string, aval<BufferView>>().add("position", bv()),
      uniforms: HashMap.empty(),
      textures: HashMap.empty<string, aval<ITexture>>().add("src", offscreenColor),
      samplers: HashMap.empty<string, aval<ISampler>>().add("samp", cval(ISampler.fromDescriptor({ magFilter: "linear" }))),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };

    // Backbuffer for the outer scene.
    const outerSig = createFramebufferSignature({ colors: { color: "rgba8unorm" } });
    const backbuffer = allocateFramebuffer(gpu.device, outerSig, cval({ width: 64, height: 64 }));
    backbuffer.acquire();
    const ifb = backbuffer.getValue(AdaptiveToken.top);

    const outerCmds = AList.ofArray<Command>([
      { kind: "Render", output: ifb, tree: RenderTree.leaf(outerObj) },
    ]);
    const task = runtime.compile(outerCmds);
    task.run(AdaptiveToken.top);

    // Two render passes were encoded: first the inner (offscreen) clear,
    // then the inner scene render, then the outer scene render — but the
    // outer scene's render reads the inner's texture, so the inner
    // encoding happens *before* the outer draw.
    //
    // We expect at least 3 render passes:
    //   1. inner clear (offscreen FBO)
    //   2. inner render (offscreen FBO, 1 draw)
    //   3. outer render (backbuffer, 1 draw)
    expect(gpu.renderPasses.length).toBeGreaterThanOrEqual(3);
    const drawCounts = gpu.renderPasses.map(p => p.drawCalls.length);
    // The exact number of clear-only passes vs. render passes can shift
    // as we coalesce; minimal assertion: total draws across all passes
    // is exactly 2.
    expect(drawCounts.reduce((a, b) => a + b, 0)).toBe(2);
    // The pipeline cache built two pipelines (inner + outer).
    expect(gpu.pipelines).toHaveLength(2);

    task.dispose();
    backbuffer.release();
  });

  it("lifecycle: acquire on derived aval brings the FBO live; release tears it down", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({
      device: gpu.device,
      compileEffect: e => e as unknown as CompiledEffect,
    });

    const innerEff = passEffect();
    const innerObj: RenderObject = {
      effect: innerEff as unknown as RenderObject["effect"],
      pipelineState: { rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } },
      vertexAttributes: HashMap.empty<string, aval<BufferView>>().add("position", bv()),
      uniforms: HashMap.empty(),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };
    const result = runtime.renderTo(RenderTree.leaf(innerObj), {
      size: cval({ width: 16, height: 16 }),
      signature: createFramebufferSignature({ colors: { color: "rgba8unorm" } }),
    });

    const colorAval = result.color("color");
    expect(gpu.textures).toHaveLength(0);
    colorAval.acquire();
    // FBO not yet allocated — allocation happens on first compute.
    expect(gpu.textures).toHaveLength(0);
    // First read materialises the FBO texture.
    colorAval.getValue(AdaptiveToken.top);
    expect(gpu.textures).toHaveLength(1);

    colorAval.release();
    // Last release frees the FBO texture.
    expect(gpu.textures[0]!.destroyed).toBe(true);
  });
});
