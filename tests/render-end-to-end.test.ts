// End-to-end render slice: hand-built CompiledEffect → prepareRenderObject
// → render() → mock render-pass capture.
//
// Verifies:
//   - Pipeline compilation hits the cache for repeated `prepareRenderObject` calls
//   - Vertex buffers are bound to the right slots
//   - The single Globals UBO ends up in bind group 0, binding 0
//   - draw / drawIndexed routes correctly

import { describe, expect, it } from "vitest";
import { AdaptiveToken, HashMap, cval, type aval } from "@aardworx/wombat.adaptive";
import { V4f } from "@aardworx/wombat.base";
import {
  IBuffer,
  type BufferView,
  type CompiledEffect,
  type DrawCall,
  type RenderObject,
} from "@aardworx/wombat.rendering-core";
import {
  allocateFramebuffer,
  createFramebufferSignature,
  prepareRenderObject,
} from "@aardworx/wombat.rendering-resources";
import { render } from "@aardworx/wombat.rendering-commands";
import { MockGPU } from "./_mockGpu.js";

function bufferView(bytes: number, format: GPUVertexFormat, count: number): aval<BufferView> {
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

function fakeEffect(vsSrc: string, fsSrc: string): CompiledEffect {
  return {
    target: "wgsl",
    stages: [
      { stage: "vertex",   entryName: "main", source: vsSrc, bindings: [] as never, meta: {} as never, sourceMap: null },
      { stage: "fragment", entryName: "main", source: fsSrc, bindings: [] as never, meta: {} as never, sourceMap: null },
    ],
    interface: {
      target: "wgsl",
      stages: [],
      attributes: [
        { name: "position", location: 0, type: vec3f, format: "float32x3", components: 3, byteSize: 12 },
        { name: "normal",   location: 1, type: vec3f, format: "float32x3", components: 3, byteSize: 12 },
      ],
      fragmentOutputs: [
        { name: "color", location: 0, type: vec4f },
      ],
      uniforms: [],
      uniformBlocks: [
        {
          name: "Globals", group: 0, slot: 0, size: 16,
          fields: [{ name: "tint", type: vec4f, offset: 0, size: 16, align: 16 }],
        },
      ],
      samplers: [],
      textures: [],
      storageBuffers: [],
    },
    avalBindings: {},
  };
}

describe("end-to-end render", () => {
  it("compiles a pipeline, binds VBs + UBO, issues draw", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({ colors: { color: "rgba8unorm" } });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();

    const obj: RenderObject = {
      effect: {} as never,
      pipelineState: {
        rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" },
      },
      vertexAttributes: HashMap.empty<string, aval<BufferView>>()
        .add("position", bufferView(36, "float32x3", 3))
        .add("normal",   bufferView(36, "float32x3", 3)),
      uniforms: HashMap.empty<string, aval<unknown>>()
        .add("tint", cval(new V4f(1, 0.5, 0.25, 1))),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };

    const eff = fakeEffect("@vertex fn main(){}", "@fragment fn main(){}");
    const prepared = prepareRenderObject(gpu.device, obj, eff, sig, { label: "tri" });
    prepared.acquire();

    const enc = gpu.createCommandEncoder();
    render(enc, prepared, fbo.getValue(AdaptiveToken.top), AdaptiveToken.top);

    expect(gpu.renderPasses).toHaveLength(1);
    const pass = gpu.renderPasses[0]!;
    expect(pass.ended).toBe(true);
    expect(pass.setPipelineCalls).toHaveLength(1);
    expect(pass.setBindGroupCalls).toHaveLength(1);
    expect(pass.setBindGroupCalls[0]!.group).toBe(0);
    expect(pass.setVertexBufferCalls).toHaveLength(2);
    const slots = pass.setVertexBufferCalls.map(c => c.slot).sort();
    expect(slots).toEqual([0, 1]);
    expect(pass.drawCalls).toEqual([
      { vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 },
    ]);

    expect(gpu.pipelines).toHaveLength(1);
    expect(gpu.shaderModules).toHaveLength(2);                 // vs + fs distinct sources
    const pdesc = gpu.pipelines[0]!;
    expect((pdesc.fragment!.targets as GPUColorTargetState[])[0]!.format).toBe("rgba8unorm");

    prepared.release();
    fbo.release();
  });

  it("pipeline cache reuses on repeat", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({ colors: { color: "rgba8unorm" } });
    const eff = fakeEffect("@vertex fn main(){}", "@fragment fn main(){}");
    const make = () => {
      const obj: RenderObject = {
        effect: {} as never,
        pipelineState: { rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } },
        vertexAttributes: HashMap.empty<string, aval<BufferView>>()
          .add("position", bufferView(36, "float32x3", 3))
          .add("normal",   bufferView(36, "float32x3", 3)),
        uniforms: HashMap.empty<string, aval<unknown>>()
          .add("tint", cval(new V4f(0, 0, 0, 1))),
        textures: HashMap.empty(),
        samplers: HashMap.empty(),
        drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
      };
      return prepareRenderObject(gpu.device, obj, eff, sig);
    };
    const a = make();
    const b = make();
    expect(a.pipeline).toBe(b.pipeline);
    expect(gpu.pipelines).toHaveLength(1);
  });
});
