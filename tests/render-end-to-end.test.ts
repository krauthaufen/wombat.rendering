// Pipeline cache invariant: two prepareRenderObject calls with the
// same (effect, signature, pipeline-state) share a GPURenderPipeline.
// Real-GPU coverage of "compile shader → encode draw" lives in
// tests-browser/render-real.test.ts.

import { describe, expect, it } from "vitest";
import { HashMap, cval, type aval } from "@aardworx/wombat.adaptive";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import {
  IBuffer,
  type BufferView,
  type DrawCall,
} from "@aardworx/wombat.rendering-core";
import {
  createFramebufferSignature,
  prepareRenderObject,
} from "@aardworx/wombat.rendering-resources";
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
      function fsMain(input: { v_normal: V3f }): { outColor: V4f } {
        return { outColor: new V4f(input.v_normal.x, input.v_normal.y, input.v_normal.z, 1.0) };
      }
    `,
    [
      { name: "vsMain", stage: "vertex",
        inputs: [
          { name: "position", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
          { name: "normal",   type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
        ],
        outputs: [
          { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
          { name: "v_normal",    type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 0 }] },
        ],
      },
      { name: "fsMain", stage: "fragment",
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

describe("prepareRenderObject", () => {
  it("(effect, signature) cache reuses pipelines across two prepares with identical state", () => {
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
