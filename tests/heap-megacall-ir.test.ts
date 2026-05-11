// Inspect the WGSL produced by the megacall IR rewrite pipeline
// against a representative DSL effect (the heap-demo's defaultVs +
// flatFs shape — VS reads ModelTrafo / ViewProjTrafo / Color +
// per-vertex Positions / Normals; FS just writes Color through).
//
// We don't need a real GPU; this is a pure compile-time inspection
// to flush out structural bugs in `applyMegacallToEmittedVs` /
// `compileHeapEffectIR`.

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

describe("megacall IR WGSL emission", () => {
  it("emits a compileable VS with binary search prelude (DSL path)", () => {
    const source = `
      function vsMain(input: { Positions: V4f; Normals: V3f }): {
        gl_Position: V4f; worldPos: V3f; normal: V3f; color: V4f;
      } {
        // Read uniforms; the heapScene IR rewrite replaces these reads
        // with heap-load expressions before WGSL emit.
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
    const ir = compileHeapEffectIR(effect, layout, { target: "wgsl" });
    // eslint-disable-next-line no-console
    console.log("=== VS ===\n" + ir.vs + "\n=== FS ===\n" + ir.fs);

    // Sanity checks: the decoder VS has the megacall builtin, the
    // heap arena bindings, and the binary-search-derived locals.
    expect(ir.vs).toContain("@builtin(vertex_index) vertex_index: u32");
    expect(ir.vs).toMatch(/drawTable:\s+array<u32>/);
    expect(ir.vs).toMatch(/indexStorage:\s+array<u32>/);
    // No `var<private>` for the megacall shared values — module-scope
    // mutable state is rejected by some WGSL parsers (Safari/WebKit)
    // when read from helper fn bodies.
    expect(ir.vs).not.toMatch(/var<private>\s+heap_drawIdx/);
    expect(ir.vs).not.toMatch(/var<private>\s+instId/);
    expect(ir.vs).not.toMatch(/var<private>\s+vid/);
    // Per-vertex `vid` is always live (heap attribute loads depend on
    // it). `heap_drawIdx` is always live (every per-RO load reads
    // headers via it). `instId` lives iff the effect reads
    // instance_index; this effect doesn't, so DCE drops it.
    expect(ir.vs).toMatch(/let heap_drawIdx:\s*u32\s*=\s*drawTable\[/);
    expect(ir.vs).toMatch(/let vid:\s*u32\s*=\s*indexStorage\[/);
    expect(ir.vs).not.toMatch(/let __heap_drawIdx\b/);
    // No leftover @builtin(instance_index) — the decoder owns the
    // megacall search and consumers receive `instId` as a local.
    expect(ir.vs).not.toMatch(/@builtin\(\s*instance_index\s*\)/);
  });
});
