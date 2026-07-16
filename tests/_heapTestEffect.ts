// Shared heap-test Effect built via the inline-marker plugin —
// same authoring path as production code in heap-demo and wombat.dom.
// The wombat.shader-vite + boperators plugins lower each
// `vertex(...) / fragment(...)` call to IR at build time; uniform
// reads become `ReadInput("Uniform", name)` nodes, sampler refs
// become module-scope Sampler bindings the heap-atlas path can
// substitute. No hand-constructed `ValueDef.Uniform` ever enters
// the module.

import { effect, vertex, fragment, type Effect } from "@aardworx/wombat.shader";
import { type Sampler2D, texture, textureSize } from "@aardworx/wombat.shader/types";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { V3f, V4f } from "@aardworx/wombat.base";

// `ModelTrafo`, `ViewProjTrafo`, `LightLocation` are in the default
// `UniformScope`; we augment only for the test-specific `Color`.
declare module "@aardworx/wombat.shader/uniforms" {
  interface UniformScope {
    readonly Color: V4f;
  }
}

const checker: Sampler2D = null as unknown as Sampler2D;

// VS reads Positions + Normals attributes and 4 uniforms (3 built-in,
// 1 augmented). Writes 5 varyings, every uniform/attribute wired
// into the live data so DCE keeps them in the ProgramInterface.

const baseVS = vertex((v: {
  Positions: V4f;
  Normals:   V3f;
}) => {
  const wp = uniform.ModelTrafo.mul(v.Positions);
  const n4 = uniform.ModelTrafo.mul(new V4f(v.Normals, 0.0));
  return {
    gl_Position: uniform.ViewProjTrafo.mul(wp),
    worldPos:    wp.xyz,
    normal:      n4.xyz,
    color:       uniform.Color,
    lightLoc:    uniform.LightLocation,
  };
});

const plainFS = fragment((v: { color: V4f }) => ({
  outColor: v.color,
}));

const texturedFS = fragment((v: { color: V4f }) => {
  const tex = texture(checker, v.color.xy);
  return { outColor: v.color.mul(tex) };
});

// Samples AND reads the texture's size — exercises the heap-atlas
// `textureSize` rewrite (per-draw size from the draw header).
const texturedSizeFS = fragment((v: { color: V4f }) => {
  const tex = texture(checker, v.color.xy);
  const ts = textureSize(checker, 0);
  return { outColor: tex.mul(new V4f(v.color.x * ts.x, v.color.y * ts.y, 1.0, 1.0)) };
});

export function makeHeapTestEffect(): Effect {
  return effect(baseVS, plainFS);
}

export function makeHeapTestEffectTextured(): Effect {
  return effect(baseVS, texturedFS);
}

export function makeHeapTestEffectTexturedSize(): Effect {
  return effect(baseVS, texturedSizeFS);
}
