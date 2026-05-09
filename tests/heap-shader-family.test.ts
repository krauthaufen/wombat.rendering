// Tests for `buildShaderFamily` — the §6 family-merge analysis pass.
//
// Synthesises effects via `_makeEffect`, runs `buildShaderFamily`, and
// asserts the resulting `ShaderFamilySchema` shape (id, layoutIdOf,
// per-effect slot maps, drawHeaderUnion, type-conflict throws,
// determinism under input reordering, custom slotAssigner honored).

import { describe, expect, it } from "vitest";
import {
  Tu32, Tf32, Vec, Mat,
  type Type, type ValueDef,
} from "@aardworx/wombat.shader/ir";
import {
  buildShaderFamily,
  type FamilySlot,
} from "../packages/rendering/src/runtime/heapShaderFamily.js";
import { makeEffect } from "./_makeEffect.js";
import type { Effect } from "@aardworx/wombat.shader";

void Tu32;

const Tvec2f: Type = Vec(Tf32, 2);
const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);
const Tmat4: Type = Mat(Tf32, 4, 4);

// ─── Effect fixtures ────────────────────────────────────────────────

/** Surface effect: ModelTrafo+ViewProjTrafo uniforms; WorldPositions, Normals, Colors varyings. */
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
    {
      name: "vsMain", stage: "vertex",
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
    {
      name: "fsMain", stage: "fragment",
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

/** Like surface, but adds a `Uvs: vec2<f32>` varying for a textured path. */
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
    {
      name: "vsMain", stage: "vertex",
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
    {
      name: "fsMain", stage: "fragment",
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

/** Effect declaring `Colors` as a vec3 (mismatched against surface's vec4). */
function makeConflictingColorsEffect(): Effect {
  const source = `
    function vsMain(input: { Positions: V4f }): {
      gl_Position: V4f; Colors: V3f;
    } {
      return {
        gl_Position: ViewProjTrafo.mul(input.Positions),
        Colors: SurfaceColor3,
      };
    }
    function fsMain(input: { Colors: V3f }): { outColor: V4f } {
      return { outColor: new V4f(input.Colors, 1.0) };
    }
  `;
  const extras: ValueDef[] = [{
    kind: "Uniform", binding: { group: 0, slot: 99 }, name: "U", uniforms: [
      { name: "ViewProjTrafo", type: Tmat4 },
      { name: "SurfaceColor3", type: Tvec3f },
    ],
  }];
  return makeEffect(source, [
    {
      name: "vsMain", stage: "vertex",
      inputs: [
        { name: "Positions", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
      ],
      outputs: [
        { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "Colors",      type: Tvec3f, semantic: "Color",    decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
    {
      name: "fsMain", stage: "fragment",
      inputs: [
        { name: "Colors", type: Tvec3f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
      ],
      outputs: [
        { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
  ], { extraValues: extras });
}

/**
 * Effect with same uniform NAME `ModelTrafo` but as vec4 instead of mat4.
 * The drawHeader entry is keyed by uniform name → wgslType conflict
 * across effects → throw.
 */
function makeConflictingUniformEffect(): Effect {
  const source = `
    function vsMain(input: { Positions: V4f }): {
      gl_Position: V4f; Colors: V4f;
    } {
      const p = input.Positions.add(ModelTrafo);
      return {
        gl_Position: p,
        Colors: SurfaceColor.add(p),
      };
    }
    function fsMain(input: { Colors: V4f }): { outColor: V4f } {
      return { outColor: input.Colors };
    }
  `;
  const extras: ValueDef[] = [{
    kind: "Uniform", binding: { group: 0, slot: 99 }, name: "U", uniforms: [
      { name: "ModelTrafo",   type: Tvec4f }, // <-- conflicts with surface's mat4
      { name: "SurfaceColor", type: Tvec4f },
    ],
  }];
  return makeEffect(source, [
    {
      name: "vsMain", stage: "vertex",
      inputs: [
        { name: "Positions", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
      ],
      outputs: [
        { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "Colors",      type: Tvec4f, semantic: "Color",    decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
    {
      name: "fsMain", stage: "fragment",
      inputs: [
        { name: "Colors", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
      ],
      outputs: [
        { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
  ], { extraValues: extras });
}

// ─── Tests ──────────────────────────────────────────────────────────

describe("buildShaderFamily", () => {
  it("single-effect family — schema is just the one effect's", () => {
    const surface = makeSurfaceEffect();
    const family = buildShaderFamily([surface]);

    expect(family.effects).toHaveLength(1);
    expect(family.effects[0]).toBe(surface);
    expect(family.layoutIdOf.get(surface)).toBe(0);
    expect(family.id).toBeTypeOf("string");
    expect(family.id.length).toBeGreaterThan(0);

    // varyingSlots covers the surviving non-builtin VS outputs
    // (WorldPositions vec3, Normals vec3, Colors vec4). With greedy
    // packing and exactOptionalPropertyTypes, slot 0 holds vec3 +
    // nothing extra (next vec3 won't fit), slot 1 holds vec3, slot 2
    // holds vec4 → 3 slots.
    expect(family.varyingSlots).toBe(3);

    const slotMap = family.perEffectSlotMap.get(surface)!;
    expect(slotMap.size).toBe(3);
    const wp = slotMap.get("WorldPositions")!;
    expect(wp).toEqual({ slot: 0, offset: 0, size: 3 } as FamilySlot);
    const nrm = slotMap.get("Normals")!;
    expect(nrm).toEqual({ slot: 1, offset: 0, size: 3 } as FamilySlot);
    const col = slotMap.get("Colors")!;
    expect(col).toEqual({ slot: 2, offset: 0, size: 4 } as FamilySlot);

    // drawHeaderUnion: surface's fields plus __layoutId. Only the
    // surface uniforms are in here (no per-vertex attrs go through
    // the IR-rewrite path for this synthetic effect; `attributes` are
    // populated from the schema and DO show up as attribute-refs).
    const fieldNames = family.drawHeaderUnion.drawHeaderFields.map(f => f.name);
    expect(fieldNames).toContain("__layoutId");
    expect(fieldNames).toContain("ModelTrafo");
    expect(fieldNames).toContain("ViewProjTrafo");
    expect(fieldNames).toContain("SurfaceColor");

    // layoutId is a u32 at the end.
    const layoutIdField = family.drawHeaderUnion.drawHeaderFields.find(
      f => f.name === "__layoutId",
    )!;
    expect(layoutIdField.wgslType).toBe("u32");
    expect(layoutIdField.byteSize).toBe(4);
  });

  it("two effects with disjoint extra varyings — slots accommodate the larger one", () => {
    const surface = makeSurfaceEffect();
    const textured = makeTexturedEffect();
    const family = buildShaderFamily([surface, textured]);

    expect(family.effects).toHaveLength(2);
    // Sorted by id; whatever the sorted order is, both layoutIds are
    // assigned 0 and 1 distinctly.
    const ids = new Set([
      family.layoutIdOf.get(surface),
      family.layoutIdOf.get(textured),
    ]);
    expect(ids).toEqual(new Set([0, 1]));

    // textured has 4 non-builtin varyings (vec3 + vec3 + vec4 + vec2);
    // greedy pack → slots [vec3 | -, vec3 | -, vec4, vec2 | -] = 4 slots
    // (vec2 fits into a fresh slot since slot 0 has 1 float remaining
    // but vec2 needs 2 — actually fits! Let me recompute: slot 0 holds
    // vec3 (3 used, 1 left). vec2 needs 2 floats — does NOT fit in
    // slot 0, opens slot 3 instead). Actually: order is WorldPositions
    // (vec3, slot 0), Normals (vec3, slot 1), Colors (vec4, slot 2),
    // Uvs (vec2, slot 3). varyingSlots = 4.
    expect(family.varyingSlots).toBe(4);

    const surfMap = family.perEffectSlotMap.get(surface)!;
    expect(surfMap.size).toBe(3);
    expect(surfMap.has("Uvs")).toBe(false);

    const texMap = family.perEffectSlotMap.get(textured)!;
    expect(texMap.size).toBe(4);
    expect(texMap.get("Uvs")).toEqual({ slot: 3, offset: 0, size: 2 } as FamilySlot);

    // Same-named varyings have independent slot positions per effect
    // (they happen to coincide here because greedy packing is the
    // same up to that point — and that's fine).
    expect(surfMap.get("WorldPositions")).toEqual(texMap.get("WorldPositions"));
  });

  it("same-name same-type varying — works, slot maps are independent", () => {
    const surface = makeSurfaceEffect();
    const surface2 = makeSurfaceEffect(); // distinct identity, same shape
    const family = buildShaderFamily([surface, surface2]);

    expect(family.effects).toHaveLength(2);
    // Same-name same-type Colors → no throw; both effects get their own slot maps.
    const m1 = family.perEffectSlotMap.get(surface)!;
    const m2 = family.perEffectSlotMap.get(surface2)!;
    expect(m1).not.toBe(m2);
    expect(m1.get("Colors")).toEqual(m2.get("Colors"));
  });

  it("same-name DIFFERENT type varying → throws with clear error", () => {
    const surface = makeSurfaceEffect();           // Colors: vec4
    const conflicting = makeConflictingColorsEffect(); // Colors: vec3

    expect(() => buildShaderFamily([surface, conflicting])).toThrowError(
      /Colors.*conflicting type/,
    );
  });

  it("same-name DIFFERENT type drawHeader uniform → throws with clear error", () => {
    const surface = makeSurfaceEffect();           // ModelTrafo: mat4
    const conflicting = makeConflictingUniformEffect(); // ModelTrafo: vec4

    expect(() => buildShaderFamily([surface, conflicting])).toThrowError(
      /ModelTrafo.*conflicting WGSL type/,
    );
  });

  it("determinism: building from [A, B] and [B, A] produces same id and same layoutIdOf", () => {
    const a = makeSurfaceEffect();
    const b = makeTexturedEffect();
    const f1 = buildShaderFamily([a, b]);
    const f2 = buildShaderFamily([b, a]);

    expect(f1.id).toBe(f2.id);
    expect(f1.layoutIdOf.get(a)).toBe(f2.layoutIdOf.get(a));
    expect(f1.layoutIdOf.get(b)).toBe(f2.layoutIdOf.get(b));
    expect(f1.varyingSlots).toBe(f2.varyingSlots);

    // Sorted-effects order matches across both calls.
    expect(f1.effects.map(e => e.id)).toEqual(f2.effects.map(e => e.id));
  });

  it("custom slotAssigner is honored when supplied", () => {
    const surface = makeSurfaceEffect();
    // Force WorldPositions into slot 5, offset 1, size 3 — way past
    // the natural greedy choice.
    const calls: string[] = [];
    const family = buildShaderFamily([surface], undefined, (_e, name, _ty) => {
      calls.push(name);
      if (name === "WorldPositions") return { slot: 5, offset: 1, size: 3 };
      return undefined; // fall back to greedy for the rest
    });

    expect(calls).toContain("WorldPositions");
    const slotMap = family.perEffectSlotMap.get(surface)!;
    expect(slotMap.get("WorldPositions")).toEqual({ slot: 5, offset: 1, size: 3 });
    // varyingSlots reflects the override (slot 5 → at least 6 slots).
    expect(family.varyingSlots).toBeGreaterThanOrEqual(6);
  });
});
