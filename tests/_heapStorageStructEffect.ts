// A hand-built effect whose fragment stage reads a storage array of
// STRUCTS: one element whole into a local, one member of another element
// on its own. (The inline-marker frontend cannot declare a struct-typed
// storage array, so the declaration is supplied as IR.)
//   Lamp (WGSL layout): Position vec3 @0, Strength f32 @12, Tint vec4 @16 — 32 bytes

import { Tf32, Vec, Mat, type Type, type ValueDef } from "@aardworx/wombat.shader/ir";
import type { Effect } from "@aardworx/wombat.shader";
import { makeEffect } from "./_makeEffect.js";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);
const Tmat4: Type = Mat(Tf32, 4, 4);
export const Lamp: Type = {
  kind: "Struct", name: "Lamp",
  fields: [{ name: "Position", type: Tvec3f }, { name: "Strength", type: Tf32 }, { name: "Tint", type: Tvec4f }],
} as Type;

/** `helper`: the storage and a uniform are read in a function the fragment stage calls, not in its own body. */
export function makeLampEffect(helper = false): Effect {
  const source = (helper ? `
    function lampStrength(i: i32): f32 {
      if (i < 0) { return 0.0; }
      let sum = 0.0;
      for (let k = 0; k < 2; k++) { sum = sum + Lamps[i + k].Strength * uColor.y; }
      return sum;
    }
    function fsMain(input: { color: V4f }): { outColor: V4f } {
      const idx = input.color.x as i32;
      return { outColor: new V4f(lampStrength(idx), lampStrength(0), 0.0, 1.0) };
    }
  ` : "") + `
    function vsMain(input: { Positions: V4f }): { gl_Position: V4f; color: V4f } {
      return { gl_Position: uViewProjTrafo.mul(input.Positions), color: uColor };
    }
    ` + (helper ? "" : `function fsMain(input: { color: V4f }): { outColor: V4f } {
      const idx = input.color.x as i32;
      const lamp = Lamps[idx];
      const dim = Lamps[0].Strength;
      return { outColor: new V4f(
        lamp.Tint.x * lamp.Strength + lamp.Position.x,
        lamp.Tint.y * lamp.Strength + lamp.Position.y,
        lamp.Tint.z * lamp.Strength + lamp.Position.z + dim,
        1.0) };
    }
  `);
  const extraValues: ValueDef[] = [
    { kind: "Uniform", binding: { group: 0, slot: 99 }, name: "U", uniforms: [
      { name: "uViewProjTrafo", type: Tmat4 }, { name: "uColor", type: Tvec4f },
    ] } as ValueDef,
    { kind: "StorageBuffer", binding: { group: 0, slot: 98 }, name: "Lamps",
      layout: { kind: "Array", element: Lamp, length: "runtime" } as Type, access: "read" },
  ];
  return makeEffect(source, [
    {
      name: "vsMain", stage: "vertex",
      inputs: [{ name: "Positions", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [
        { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "color", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
    {
      name: "fsMain", stage: "fragment",
      inputs: [{ name: "color", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    },
  ], { extraValues, ...(helper ? { helpers: ["lampStrength"] } : {}) });
}
