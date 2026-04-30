// End-to-end render slice with a real wombat.shader effect.
// Compiles vertex + fragment TS source, builds a RenderObject,
// and lowers through prepareRenderObject + render() against
// the mock device.

import { describe, expect, it } from "vitest";
import { AList, AdaptiveToken, HashMap, cval, type aval } from "@aardworx/wombat.adaptive";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader-ir";
import {
  IBuffer,
  RenderTree,
  type BufferView,
  type DrawCall,
  type RenderObject,
} from "@aardworx/wombat.rendering-core";
import {
  allocateFramebuffer,
  createFramebufferSignature,
  prepareRenderObject,
} from "@aardworx/wombat.rendering-resources";
import { Runtime } from "@aardworx/wombat.rendering-runtime";
import { makeEffect } from "./_makeEffect.js";
import { MockGPU } from "./_mockGpu.js";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

function twoAttribEffect() {
  return makeEffect(
    `
      function vsMain(input: { position: V3f; normal: V3f }): { gl_Position: V4f; v_normal: V3f } {
        return {
          gl_Position: new V4f(input.position.x, input.position.y, input.position.z, 1.0),
          v_normal: input.normal,
        };
      }
      function fsMain(input: { v_normal: V3f }): V4f {
        return new V4f(input.v_normal.x, input.v_normal.y, input.v_normal.z, 1.0);
      }
    `,
    [
      {
        name: "vsMain", stage: "vertex",
        inputs: [
          { name: "position", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
          { name: "normal",   type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
        ],
        outputs: [
          { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
          { name: "v_normal",    type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 0 }] },
        ],
      },
      {
        name: "fsMain", stage: "fragment",
        inputs:  [{ name: "v_normal",  type: Tvec3f, semantic: "Normal", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color",  decorations: [{ kind: "Location", value: 0 }] }],
      },
    ],
  );
}

function bv(bytes: number, format: GPUVertexFormat, count: number): aval<BufferView> {
  return cval<BufferView>({
    buffer: IBuffer.fromHost(new ArrayBuffer(bytes)),
    offset: 0, count, stride: 12, format,
  });
}

describe("end-to-end render", () => {
  it("compiles a pipeline, binds VBs, issues draw", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();

    const eff = twoAttribEffect();
    const compiled = eff.compile({ target: "wgsl" });
    const obj: RenderObject = {
      effect: eff,
      pipelineState: { rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } },
      vertexAttributes: HashMap.empty<string, aval<BufferView>>()
        .add("position", bv(36, "float32x3", 3))
        .add("normal",   bv(36, "float32x3", 3)),
      uniforms: HashMap.empty(),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };

    const prepared = prepareRenderObject(gpu.device, obj, compiled, sig, { label: "tri", effectId: eff.id });
    prepared.acquire();

    const runtime = new Runtime({ device: gpu.device });
    const task = runtime.compile(
      AList.ofArray([
        { kind: "Render" as const, output: fbo, tree: RenderTree.leaf(obj) },
      ]),
    );
    task.run(AdaptiveToken.top);

    expect(gpu.renderPasses).toHaveLength(1);
    const pass = gpu.renderPasses[0]!;
    expect(pass.ended).toBe(true);
    expect(pass.setPipelineCalls).toHaveLength(1);
    expect(pass.setVertexBufferCalls).toHaveLength(2);
    expect(pass.drawCalls).toEqual([
      { vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 },
    ]);
    expect(gpu.pipelines).toHaveLength(1);
    expect(gpu.shaderModules.length).toBeGreaterThan(0);
    const pdesc = gpu.pipelines[0]!;
    expect((pdesc.fragment!.targets as GPUColorTargetState[])[0]!.format).toBe("rgba8unorm");

    task.dispose();
    prepared.release();
    fbo.release();
  });

  it("(effect, signature) cache reuses pipelines across RenderObjects", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const eff = twoAttribEffect();
    const compiled = eff.compile({ target: "wgsl" });
    const make = () => prepareRenderObject(gpu.device, {
      effect: eff,
      pipelineState: { rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } },
      vertexAttributes: HashMap.empty<string, aval<BufferView>>()
        .add("position", bv(36, "float32x3", 3))
        .add("normal",   bv(36, "float32x3", 3)),
      uniforms: HashMap.empty(),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    }, compiled, sig, { effectId: eff.id });
    const a = make();
    const b = make();
    expect(a.pipeline).toBe(b.pipeline);
    expect(gpu.pipelines).toHaveLength(1);
  });
});
