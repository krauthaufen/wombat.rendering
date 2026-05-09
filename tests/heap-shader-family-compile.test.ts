// Tests for `compileShaderFamily` — slice 3b WGSL synthesis.
//
// Builds a `ShaderFamilySchema` from one or two effects, runs
// `compileShaderFamily`, and asserts the resulting `{ vs, fs }` pair
// has the expected wrapper shape (per-effect helpers as plain `fn`
// bodies, single `@vertex fn family_vs_main` / `@fragment fn
// family_fs_main`, megacall search prelude present, switch-on-layoutId
// dispatch with N cases). Real WebGPU validation lives in slice 3c;
// here we rely on textual assertions plus a structural sanity check
// that no decl appears twice in the merged output.

import { describe, expect, it } from "vitest";
import {
  Tu32, Tf32, Vec, Mat,
  type Type, type ValueDef,
} from "@aardworx/wombat.shader/ir";
import {
  buildShaderFamily,
  compileShaderFamily,
} from "../packages/rendering/src/runtime/heapShaderFamily.js";
import { makeEffect } from "./_makeEffect.js";
import type { Effect } from "@aardworx/wombat.shader";

void Tu32;

const Tvec2f: Type = Vec(Tf32, 2);
const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);
const Tmat4: Type = Mat(Tf32, 4, 4);

// ─── Effect fixtures (same shape as heap-shader-family.test.ts) ─────

function makeSurfaceEffect(): Effect {
  const source = `
    function vsMain(input: { Positions: V4f; Normals: V3f }): {
      gl_Position: V4f; WorldPositions: V3f; Normals: V3f; Colors: V4f;
    } {
      const wp = ModelTrafo.mul(input.Positions);
      return {
        gl_Position: ViewProjTrafo.mul(wp),
        WorldPositions: wp.xyz,
        Normals: input.Normals,
        Colors: SurfaceColor,
      };
    }
    function fsMain(input: { WorldPositions: V3f; Normals: V3f; Colors: V4f }): { outColor: V4f } {
      const w = new V4f(input.WorldPositions, 0.0);
      const n = new V4f(input.Normals, 0.0);
      return { outColor: input.Colors.add(w).add(n) };
    }
  `;
  const extras: ValueDef[] = [{
    kind: "Uniform", binding: { group: 0, slot: 99 }, name: "U", uniforms: [
      { name: "ModelTrafo", type: Tmat4 },
      { name: "ViewProjTrafo", type: Tmat4 },
      { name: "SurfaceColor", type: Tvec4f },
    ],
  }];
  return makeEffect(source, [
    { name: "vsMain", stage: "vertex",
      inputs: [
        { name: "Positions", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "Normals",   type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
      ],
      outputs: [
        { name: "gl_Position",     type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "WorldPositions",  type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "Normals",         type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
        { name: "Colors",          type: Tvec4f, semantic: "Color",    decorations: [{ kind: "Location", value: 2 }] },
      ],
    },
    { name: "fsMain", stage: "fragment",
      inputs: [
        { name: "WorldPositions", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "Normals",        type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
        { name: "Colors",         type: Tvec4f, semantic: "Color",    decorations: [{ kind: "Location", value: 2 }] },
      ],
      outputs: [
        { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
  ], { extraValues: extras });
}

function makeTexturedEffect(): Effect {
  const source = `
    function vsMain(input: { Positions: V4f; Normals: V3f; UvIn: V2f }): {
      gl_Position: V4f; WorldPositions: V3f; Normals: V3f; Colors: V4f; Uvs: V2f;
    } {
      const wp = ModelTrafo.mul(input.Positions);
      return {
        gl_Position: ViewProjTrafo.mul(wp),
        WorldPositions: wp.xyz,
        Normals: input.Normals,
        Colors: SurfaceColor,
        Uvs: input.UvIn,
      };
    }
    function fsMain(input: { WorldPositions: V3f; Normals: V3f; Colors: V4f; Uvs: V2f }): { outColor: V4f } {
      const w = new V4f(input.WorldPositions, 0.0);
      const n = new V4f(input.Normals, 0.0);
      const u = new V4f(input.Uvs, 0.0, 0.0);
      return { outColor: input.Colors.add(w).add(n).add(u) };
    }
  `;
  const extras: ValueDef[] = [{
    kind: "Uniform", binding: { group: 0, slot: 99 }, name: "U", uniforms: [
      { name: "ModelTrafo", type: Tmat4 },
      { name: "ViewProjTrafo", type: Tmat4 },
      { name: "SurfaceColor", type: Tvec4f },
    ],
  }];
  return makeEffect(source, [
    { name: "vsMain", stage: "vertex",
      inputs: [
        { name: "Positions", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "Normals",   type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
        { name: "UvIn",      type: Tvec2f, semantic: "TexCoord", decorations: [{ kind: "Location", value: 2 }] },
      ],
      outputs: [
        { name: "gl_Position",    type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "WorldPositions", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "Normals",        type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
        { name: "Colors",         type: Tvec4f, semantic: "Color",    decorations: [{ kind: "Location", value: 2 }] },
        { name: "Uvs",            type: Tvec2f, semantic: "TexCoord", decorations: [{ kind: "Location", value: 3 }] },
      ],
    },
    { name: "fsMain", stage: "fragment",
      inputs: [
        { name: "WorldPositions", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "Normals",        type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
        { name: "Colors",         type: Tvec4f, semantic: "Color",    decorations: [{ kind: "Location", value: 2 }] },
        { name: "Uvs",            type: Tvec2f, semantic: "TexCoord", decorations: [{ kind: "Location", value: 3 }] },
      ],
      outputs: [
        { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
  ], { extraValues: extras });
}

// ─── Tests ──────────────────────────────────────────────────────────

describe("compileShaderFamily", () => {
  it("single-effect family — non-empty vs + fs strings", () => {
    const family = buildShaderFamily([makeSurfaceEffect()]);
    const out = compileShaderFamily(family);
    expect(out.vs.length).toBeGreaterThan(0);
    expect(out.fs.length).toBeGreaterThan(0);
  });

  it("single-effect family — exactly one @vertex / @fragment entry, helper present", () => {
    const family = buildShaderFamily([makeSurfaceEffect()]);
    const out = compileShaderFamily(family);

    // Exactly one @vertex / @fragment in the merged output (the wrappers).
    expect(out.vs.match(/@vertex\s+fn\b/g)).toHaveLength(1);
    expect(out.vs).toMatch(/@vertex\s+fn\s+family_vs_main\b/);
    expect(out.fs.match(/@fragment\s+fn\b/g)).toHaveLength(1);
    expect(out.fs).toMatch(/@fragment\s+fn\s+family_fs_main\b/);

    // Per-effect helpers exist as regular fns (no @vertex / @fragment).
    expect(out.vs).toMatch(/\bfn\s+family_vs_0\s*\(/);
    expect(out.fs).toMatch(/\bfn\s+family_fs_0\s*\(/);
  });

  it("two-effect family — two helpers and switch with two cases", () => {
    const surface = makeSurfaceEffect();
    const textured = makeTexturedEffect();
    const family = buildShaderFamily([surface, textured]);
    const out = compileShaderFamily(family);

    // Both helper functions present (as plain fns, no stage decoration).
    expect(out.vs).toMatch(/\bfn\s+family_vs_0\s*\(/);
    expect(out.vs).toMatch(/\bfn\s+family_vs_1\s*\(/);
    // The helpers must NOT carry @vertex (only family_vs_main does).
    expect(out.vs).not.toMatch(/@vertex\s+fn\s+family_vs_0\b/);
    expect(out.vs).not.toMatch(/@vertex\s+fn\s+family_vs_1\b/);

    // FS side same.
    expect(out.fs).toMatch(/\bfn\s+family_fs_0\s*\(/);
    expect(out.fs).toMatch(/\bfn\s+family_fs_1\s*\(/);
    expect(out.fs).not.toMatch(/@fragment\s+fn\s+family_fs_0\b/);

    // Switch with two cases on layoutId.
    expect(out.vs).toMatch(/switch\s*\(\s*layoutId\s*\)/);
    expect(out.vs.match(/case\s+0u\s*:/g)).toHaveLength(1);
    expect(out.vs.match(/case\s+1u\s*:/g)).toHaveLength(1);
    expect(out.fs).toMatch(/switch\s*\(\s*in\.layoutIdIn\s*\)/);
    expect(out.fs.match(/case\s+0u\s*:/g)).toHaveLength(1);
    expect(out.fs.match(/case\s+1u\s*:/g)).toHaveLength(1);
  });

  it("VS contains the megacall search prelude — exactly once", () => {
    const family = buildShaderFamily([makeSurfaceEffect(), makeTexturedEffect()]);
    const out = compileShaderFamily(family);
    // Prelude markers.
    expect(out.vs.match(/let _tileIdx\b/g)).toHaveLength(1);
    expect(out.vs).toContain("firstDrawInTile[_tileIdx]");
    // Megacall storage-buffer bindings declared exactly once each.
    expect(out.vs.match(/\bdrawTable\s*:\s+array<u32>/g)).toHaveLength(1);
    expect(out.vs.match(/\bindexStorage\s*:\s+array<u32>/g)).toHaveLength(1);
    expect(out.vs.match(/\bfirstDrawInTile\s*:\s+array<u32>/g)).toHaveLength(1);
  });

  it("module-scope decl dedup — no duplicate `var` or `struct` decls", () => {
    const family = buildShaderFamily([makeSurfaceEffect(), makeTexturedEffect()]);
    const out = compileShaderFamily(family);
    // For each storage/uniform binding we expect a single declaration.
    // The IR currently emits heapU32/headersU32/heapF32/heapV4f bindings;
    // they should appear once each in both VS and FS.
    for (const name of ["heapU32", "headersU32", "heapF32", "heapV4f"]) {
      const re = new RegExp(`var<storage,\\s*read>\\s+${name}\\b`, "g");
      const vsMatches = out.vs.match(re);
      const fsMatches = out.fs.match(re);
      // Defensive: the per-effect emit may or may not include each one
      // depending on which loaders the IR chose. If it's there, it's
      // there exactly once.
      if (vsMatches !== null) expect(vsMatches).toHaveLength(1);
      if (fsMatches !== null) expect(fsMatches).toHaveLength(1);
    }
  });

  it("layoutId is read from headersU32 at the right offset", () => {
    const family = buildShaderFamily([makeSurfaceEffect(), makeTexturedEffect()]);
    const out = compileShaderFamily(family);
    // The wrapper should compute layoutId from the family stride and
    // the __layoutId field's offset.
    const stride = family.drawHeaderUnion.strideU32;
    const fld = family.drawHeaderUnion.drawHeaderFields.find(f => f.name === "__layoutId")!;
    const offU32 = fld.byteOffset / 4;
    const re = new RegExp(`headersU32\\[\\(heap_drawIdx\\s*\\*\\s*${stride}u\\)\\s*\\+\\s*${offU32}u\\]`);
    expect(out.vs).toMatch(re);
  });

  it("FamilyVsOut has one Varying<i> location per slot plus flat HeapVarying<k> + layoutIdOut", () => {
    const family = buildShaderFamily([makeSurfaceEffect(), makeTexturedEffect()]);
    const out = compileShaderFamily(family);
    for (let i = 0; i < family.varyingSlots; i++) {
      const re = new RegExp(`@location\\(${i}\\)\\s+Varying${i}\\s*:\\s*vec4<f32>`);
      expect(out.vs).toMatch(re);
      expect(out.fs).toMatch(re);
    }
    // Heap-injected slots (vec4<u32>, flat) sit immediately after the
    // user vec4<f32> slots. For these fixtures (no FS uniform reads /
    // no atlas) the count is 0; if the IR rewrite adds threading, the
    // count grows and this loop catches it.
    for (let i = 0; i < family.heapVaryingSlots; i++) {
      const loc = family.varyingSlots + i;
      const re = new RegExp(`@interpolate\\(flat\\)\\s+@location\\(${loc}\\)\\s+HeapVarying${i}\\s*:\\s*vec4<u32>`);
      expect(out.vs).toMatch(re);
      expect(out.fs).toMatch(re);
    }
    // layoutIdOut sits at @location(N + M) flat-interpolated.
    const N = family.varyingSlots + family.heapVaryingSlots;
    expect(out.vs).toMatch(new RegExp(`@interpolate\\(flat\\)\\s+@location\\(${N}\\)\\s+layoutIdOut\\s*:\\s*u32`));
    expect(out.fs).toMatch(new RegExp(`@interpolate\\(flat\\)\\s+@location\\(${N}\\)\\s+layoutIdIn\\s*:\\s*u32`));
  });

  it("idempotent — same family schema produces same merged output", () => {
    const a = makeSurfaceEffect();
    const b = makeTexturedEffect();
    const f1 = buildShaderFamily([a, b]);
    const f2 = buildShaderFamily([b, a]); // input order shouldn't matter
    const o1 = compileShaderFamily(f1);
    const o2 = compileShaderFamily(f2);
    expect(o1.vs).toEqual(o2.vs);
    expect(o1.fs).toEqual(o2.fs);
  });
});
