// Verifies the `family-member` emission mode of `compileHeapEffectIR`.
//
// In family-member mode the wrapper module is responsible for the
// megacall binary search; per-effect VS becomes a regular function
// that takes (heap_drawIdx, instId, vid) as plain u32 parameters.
// Standalone mode (the default) is exercised by
// `heap-megacall-ir.test.ts` and must keep working unchanged.

import { describe, expect, it } from "vitest";
import { Tu32, Tf32, Vec, Mat, type Type, type ValueDef } from "@aardworx/wombat.shader/ir";
import { compileHeapEffectIR } from "../packages/rendering/src/runtime/heapEffectIR.js";
import { buildBucketLayout } from "../packages/rendering/src/runtime/heapEffect.js";
import type { HeapEffectSchema } from "../packages/rendering/src/runtime/heapEffect.js";
import { makeEffect } from "./_makeEffect.js";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);
const Tmat4: Type = Mat(Tf32, 4, 4);

void Tu32;

function buildEffectAndLayout() {
  const source = `
    function vsMain(input: { Positions: V4f; Normals: V3f }): {
      gl_Position: V4f; worldPos: V3f; normal: V3f; color: V4f;
    } {
      const wp = uModelTrafo.mul(input.Positions);
      const n4 = uModelTrafo.mul(new V4f(input.Normals, 0.0));
      return {
        gl_Position: uViewProjTrafo.mul(wp),
        worldPos: wp.xyz, normal: n4.xyz, color: uColor,
      };
    }
    function fsMain(input: { color: V4f }): { outColor: V4f } {
      return { outColor: input.color };
    }
  `;

  const extraValues: ValueDef[] = [
    { kind: "Uniform", binding: { group: 0, slot: 99 }, name: "U", uniforms: [
      { name: "uModelTrafo", type: Tmat4 },
      { name: "uViewProjTrafo", type: Tmat4 },
      { name: "uColor", type: Tvec4f },
    ] },
  ];

  const effect = makeEffect(source, [
    {
      name: "vsMain", stage: "vertex",
      inputs: [
        { name: "Positions", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "Normals",   type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
      ],
      outputs: [
        { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "worldPos",    type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "normal",      type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
        { name: "color",       type: Tvec4f, semantic: "Color",    decorations: [{ kind: "Location", value: 2 }] },
      ],
    },
    {
      name: "fsMain", stage: "fragment",
      inputs: [
        { name: "color", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 2 }] },
      ],
      outputs: [
        { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
  ], { extraValues });

  const schema: HeapEffectSchema = {
    attributes: [
      { name: "Positions", wgslType: "vec4<f32>", byteSize: 16, location: 0 },
      { name: "Normals",   wgslType: "vec3<f32>", byteSize: 12, location: 1 },
    ],
    uniforms: [
      { name: "uModelTrafo",    wgslType: "mat4x4<f32>", byteSize: 64 },
      { name: "uViewProjTrafo", wgslType: "mat4x4<f32>", byteSize: 64 },
      { name: "uColor",         wgslType: "vec4<f32>",   byteSize: 16 },
    ],
    varyings: [
      { name: "clipPos",  wgslType: "vec4<f32>", builtin: "position" },
      { name: "worldPos", wgslType: "vec3<f32>", location: 0 },
      { name: "normal",   wgslType: "vec3<f32>", location: 1 },
      { name: "color",    wgslType: "vec4<f32>", location: 2 },
    ],
    fragmentOutputs: [{ name: "outColor", location: 0, wgslType: "vec4<f32>" }],
    textures: [],
    samplers: [],
  };

  const layout = buildBucketLayout(schema, false, {});
  return { effect, layout };
}

describe("compileHeapEffectIR family-member mode", () => {
  it("emits a VS callable from a family wrapper", () => {
    const { effect, layout } = buildEffectAndLayout();
    const ir = compileHeapEffectIR(effect, layout, { target: "wgsl" }, "family-member");

    // Entry signature carries the three plain u32 params, no @builtin.
    expect(ir.vs).toMatch(/@vertex\s+fn\s+\w+\s*\([^)]*heap_drawIdx:\s*u32[^)]*instId:\s*u32[^)]*vid:\s*u32[^)]*\)/);
    expect(ir.vs).not.toMatch(/@builtin\(\s*vertex_index\s*\)/);
    expect(ir.vs).not.toMatch(/@builtin\(\s*instance_index\s*\)/);
    // No `emitIdx` (that's the standalone-mode prelude name).
    expect(ir.vs).not.toMatch(/\bemitIdx\b/);

    // Megacall storage-buffer bindings live on the wrapper, not here.
    expect(ir.vs).not.toMatch(/var<storage,\s*read>\s+drawTable\b/);
    expect(ir.vs).not.toMatch(/var<storage,\s*read>\s+indexStorage\b/);
    expect(ir.vs).not.toMatch(/var<storage,\s*read>\s+firstDrawInTile\b/);
    // No binary-search loop fragments.
    expect(ir.vs).not.toMatch(/\b_tileIdx\b/);
    expect(ir.vs).not.toMatch(/\bfirstDrawInTile\b/);

    // In the decoder-composition path the decoder synthesises the
    // per-input/uniform loads using `instId`/`vid` directly via Var
    // expressions; the IR doesn't emit `let vertex_index = vid;`
    // aliases that the legacy substitute path used. The body's
    // megacall identifiers ARE `instId` / `vid` (no `vertex_index` /
    // `instance_index` aliases needed).
    expect(ir.vs).not.toMatch(/let\s+instance_index\b/);
    expect(ir.vs).not.toMatch(/let\s+vertex_index\b/);

    // Heap arena bindings (0..3) ARE present — the per-effect VS still
    // reads heapU32/headersU32/heapF32/heapV4f directly.
    expect(ir.vs).toMatch(/var<storage,\s*read>\s+heapU32\b/);
    expect(ir.vs).toMatch(/var<storage,\s*read>\s+headersU32\b/);
  });

  it("standalone mode (default) keeps the megacall prelude and bindings", () => {
    const { effect, layout } = buildEffectAndLayout();
    const ir = compileHeapEffectIR(effect, layout, { target: "wgsl" });
    // Standalone shape: vertex_index builtin renamed to emitIdx, prelude
    // injected, drawTable/indexStorage/firstDrawInTile bindings appended.
    expect(ir.vs).toContain("@builtin(vertex_index) vertex_index: u32");
    expect(ir.vs).toMatch(/drawTable:\s+array<u32>/);
    expect(ir.vs).toMatch(/firstDrawInTile:\s+array<u32>/);
    expect(ir.vs).toMatch(/let _tileIdx\b/);
  });
});
