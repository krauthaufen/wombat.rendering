// Shared minimal heap-test Effect: lambert-style VS reading
// Positions (vec4) + Normals (vec3) attributes + ModelTrafo/ViewProjTrafo
// (mat4) + Color (vec4) + LightLocation (vec3) uniforms; FS outputs a
// shaded color so the heap path covers attribute decode, mat4/vec4/vec3
// uniform decodes, and a writable @location(0) outColor.

import { parseShader } from "@aardworx/wombat.shader/frontend";
import { stage, type Effect } from "@aardworx/wombat.shader";
import {
  Tf32, Vec, Mat,
  type Module, type Type, type ValueDef,
} from "@aardworx/wombat.shader/ir";

const TSampler2D: Type = {
  kind: "Sampler", target: "2D",
  sampled: { kind: "Float" }, comparison: false,
};

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);
const Tmat4: Type = Mat(Tf32, 4, 4);

// Every uniform/attribute is wired into outColor so the IR's DCE
// keeps them in the final ProgramInterface — the heap layout reflects
// what was in the source.

const SOURCE_PLAIN = `
  function vsMain(input: { Positions: V4f; Normals: V3f }): {
    clipPos: V4f; worldPos: V3f; normal: V3f; color: V4f; lightLoc: V3f;
  } {
    const wp = ModelTrafo.mul(input.Positions);
    const n4 = ModelTrafo.mul(new V4f(input.Normals, 0.0));
    return {
      clipPos: ViewProjTrafo.mul(wp),
      worldPos: wp.xyz, normal: n4.xyz,
      color: Color, lightLoc: LightLocation,
    };
  }
  function fsMain(input: { color: V4f }): { outColor: V4f } {
    return { outColor: input.color };
  }
`;

const SOURCE_TEXTURED = `
  function vsMain(input: { Positions: V4f; Normals: V3f }): {
    clipPos: V4f; worldPos: V3f; normal: V3f; color: V4f; lightLoc: V3f;
  } {
    const wp = ModelTrafo.mul(input.Positions);
    const n4 = ModelTrafo.mul(new V4f(input.Normals, 0.0));
    return {
      clipPos: ViewProjTrafo.mul(wp),
      worldPos: wp.xyz, normal: n4.xyz,
      color: Color, lightLoc: LightLocation,
    };
  }
  function fsMain(input: { color: V4f }): { outColor: V4f } {
    const tex = texture(checker, input.color.xy);
    return { outColor: input.color.mul(tex) };
  }
`;

const UNIFORMS_VALUE: ValueDef = {
  kind: "Uniform", uniforms: [
    { name: "ModelTrafo",    type: Tmat4 },
    { name: "ViewProjTrafo", type: Tmat4 },
    { name: "Color",         type: Tvec4f },
    { name: "LightLocation", type: Tvec3f },
  ],
};

const SAMPLER_VALUE: ValueDef = {
  kind: "Sampler", binding: { group: 0, slot: 4 }, name: "checker", type: TSampler2D,
};

function build(source: string, extras: readonly ValueDef[]): Effect {
  const externalTypes = new Map<string, Type>();
  for (const v of extras) {
    if (v.kind === "Uniform") for (const u of v.uniforms) externalTypes.set(u.name, u.type);
    else if (v.kind === "Sampler") externalTypes.set(v.name, v.type);
  }
  const parsed = parseShader({
    source, externalTypes,
    entries: [
      {
        name: "vsMain", stage: "vertex",
        inputs: [
          { name: "Positions", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
          { name: "Normals",   type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
        ],
        outputs: [
          { name: "clipPos",  type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
          { name: "worldPos", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
          { name: "normal",   type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
          { name: "color",    type: Tvec4f, semantic: "Color",    decorations: [{ kind: "Location", value: 2 }] },
          { name: "lightLoc", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 3 }] },
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
    ],
  });
  const merged: Module = { ...parsed, values: [...extras, ...parsed.values] };
  return stage(merged);
}

export function makeHeapTestEffect(): Effect {
  return build(SOURCE_PLAIN, [UNIFORMS_VALUE]);
}

export function makeHeapTestEffectTextured(): Effect {
  return build(SOURCE_TEXTURED, [UNIFORMS_VALUE, SAMPLER_VALUE]);
}
