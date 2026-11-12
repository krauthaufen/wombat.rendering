// A draw whose linked shader reads an attribute it lacks
// (runtime/missingAttributes.ts): which attributes count, and the two
// throws a draw that bypasses compileHybridScene's check still meets,
// both naming the draw (`debugName`) and the attributes it has. The
// scene-level behaviour (left out, reported once, the rest draws) is in
// tests-browser/missing-attribute-real.test.ts.

import { describe, expect, it } from "vitest";
import { AdaptiveToken, HashMap, cval, AVal } from "@aardworx/wombat.adaptive";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import {
  IBuffer, type BufferView, type DrawCall, type RenderObject, PipelineState, ElementType,
} from "@aardworx/wombat.rendering/core";
import { createFramebufferSignature, prepareRenderObject } from "@aardworx/wombat.rendering/resources";
import { buildHeapScene, type HeapDrawSpec } from "../packages/rendering/src/runtime/heapScene.js";
import { missingAttributes, missingAttributesMessage } from "../packages/rendering/src/runtime/missingAttributes.js";
import { makeEffect } from "./_makeEffect.js";
import { MockGPU } from "./_mockGpu.js";
import { makeHeapTestEffect } from "./_heapTestEffect.js";

if (typeof (globalThis as { GPUTextureUsage?: unknown }).GPUTextureUsage === "undefined") {
  (globalThis as Record<string, unknown>).GPUTextureUsage = {
    COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04,
    STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10,
  };
}
if (typeof (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage === "undefined") {
  (globalThis as Record<string, unknown>).GPUBufferUsage = {
    MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
    INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
    INDIRECT: 0x0100, QUERY_RESOLVE: 0x0200,
  };
}
if (typeof (globalThis as { GPUShaderStage?: unknown }).GPUShaderStage === "undefined") {
  (globalThis as Record<string, unknown>).GPUShaderStage = {
    VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4,
  };
}

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

function bv(bytes: number, elementType: ElementType): BufferView {
  return {
    buffer: AVal.constant(IBuffer.fromHost(new ArrayBuffer(bytes))),
    elementType,
  };
}

const layout = { locations: new Map([["outColor", 0]]) };
const ro = (effect: RenderObject["effect"], names: string[], debugName?: string): RenderObject => {
  let m = HashMap.empty<string, BufferView>();
  for (const n of names) m = m.add(n, bv(36, ElementType.V3f));
  return {
    effect,
    ...(debugName !== undefined ? { debugName } : {}),
    pipelineState: PipelineState.constant({ rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } }),
    vertexAttributes: m,
    uniforms: HashMap.empty(),
    textures: HashMap.empty(),
    samplers: HashMap.empty(),
    drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
  } as RenderObject;
};

describe("missing attributes", () => {
  it("counts what the linked shader reads, not what a stage declares", () => {
    expect(missingAttributes(ro(twoAttribEffect(), ["position", "normal"]), layout)).toEqual([]);
    expect(missingAttributes(ro(twoAttribEffect(), ["position"]), layout)).toEqual(["normal"]);
    // the vertex stage passes `Normals` on to a varying the fragment stage never reads
    expect(missingAttributes(ro(makeHeapTestEffect(), ["Positions"]), layout)).toEqual([]);
    expect(missingAttributes(ro(makeHeapTestEffect(), ["Normals"]), layout)).toEqual(["Positions"]);
  });

  it("the message names the draw, the missing attribute and those it has", () => {
    const r = ro(twoAttribEffect(), ["position"], "mesh at App.fs:12");
    expect(missingAttributesMessage(r, ["normal"])).toBe(
      `mesh at App.fs:12: its shader reads attribute "normal", which it does not have (it has position); not drawn`);
  });

  it("legacy: prepareRenderObject's throw names the draw and its attributes", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const eff = twoAttribEffect();
    expect(() => prepareRenderObject(gpu.device, ro(eff, ["position"], "bad leaf"),
      eff.compile({ target: "wgsl" }), sig, { effectId: eff.id }))
      .toThrow(`prepareRenderObject: bad leaf: missing vertex attribute "normal" (it has position)`);
  });

  it("heap: addDraw's throw names the draw and the attributes given", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const spec: HeapDrawSpec = {
      effect: twoAttribEffect(),
      debugName: "bad leaf",
      inputs: { position: bv(36, ElementType.V3f) },
      vertexCount: 3,
    };
    expect(() => buildHeapScene(gpu.device, sig, [spec], { fragmentOutputLayout: layout }).update(AdaptiveToken.top))
      .toThrow(/heapScene: bad leaf: spec\.inputs missing required entry 'normal' .*attributes given: position\)/);
  });
});
