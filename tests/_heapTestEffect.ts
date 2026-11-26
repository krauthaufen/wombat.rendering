// Shared heap-test Effect built via the inline-marker plugin —
// same authoring path as production code in heap-demo and wombat.dom.
// The wombat.shader-vite + boperators plugins lower each
// `vertex(...) / fragment(...)` call to IR at build time; uniform
// reads become `ReadInput("Uniform", name)` nodes, sampler refs
// become module-scope Sampler bindings the heap-atlas path can
// substitute. No hand-constructed `ValueDef.Uniform` ever enters
// the module.

import { effect, vertex, fragment, type Effect } from "@aardworx/wombat.shader";
import { type Sampler2D, type SamplerCube, type Storage, type i32, texelFetch, texture, textureLod, textureSize } from "@aardworx/wombat.shader/types";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { V2i, V3f, V4f } from "@aardworx/wombat.base";

// `ModelTrafo`, `ViewProjTrafo`, `LightLocation` are in the default
// `UniformScope`; we augment only for the test-specific `Color`.
declare module "@aardworx/wombat.shader/uniforms" {
  interface UniformScope {
    readonly Color: V4f;
  }
}

const checker: Sampler2D = null as unknown as Sampler2D;
const envCube: SamplerCube = null as unknown as SamplerCube;

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

// Samples a samplerCube with a world-space direction — exercises the
// heap cube→2D-grid rewrite (textureSample(cube, dir) → sample the
// packed 3×2 face grid via cubeProjectZup).
const cubeFS = fragment((v: { worldPos: V3f }) => {
  const c = texture(envCube, v.worldPos);
  return { outColor: new V4f(c.x, c.y, c.z, 1.0) };
});

export function makeHeapTestEffect(): Effect {
  return effect(baseVS, plainFS);
}

export function makeHeapTestEffectCube(): Effect {
  return effect(baseVS, cubeFS);
}

// Three textures in one effect (a material: colour, detail, mask), each
// sampled on its own: every one is an atlas lookup with its own placement.
const detail: Sampler2D = null as unknown as Sampler2D;
const mask: Sampler2D = null as unknown as Sampler2D;
const threeTexturesFS = fragment((v: { color: V4f }) => {
  const a = texture(checker, v.color.xy);
  const b = texture(detail, v.color.xy);
  const c = texture(mask, v.color.xy);
  return { outColor: a.mul(b).add(c) };
});
export function makeHeapTestEffectThreeTextures(): Effect {
  return effect(baseVS, threeTexturesFS);
}

// The other ways of reading a texture: an explicit level, and a single texel.
const explicitReadsFS = fragment((v: { color: V4f }) => {
  const a = textureLod(checker, v.color.xy, 2.0);
  const b = texelFetch(detail, new V2i(3, 4), 1);
  return { outColor: a.add(b) };
});
export function makeHeapTestEffectExplicitReads(): Effect {
  return effect(baseVS, explicitReadsFS);
}

export function makeHeapTestEffectTextured(): Effect {
  return effect(baseVS, texturedFS);
}

export function makeHeapTestEffectTexturedSize(): Effect {
  return effect(baseVS, texturedSizeFS);
}

// ─── heap-managed storage buffers ──────────────────────────────────────
// Two read-only storage arrays, read in BOTH stages:
//   Offsets (vec3 elements — 16-byte stride in WGSL's layout) moves the
//   vertices by its SECOND element; Palette (vec4 elements) gives the
//   fragment colour, picked by `Color.x`.
declare const Offsets: Storage<V3f[], "read">;
declare const Palette: Storage<V4f[], "read">;

const storageVS = vertex((v: { Positions: V4f }) => {
  const I1: i32 = 1 as i32;
  const o = Offsets[I1];
  const wp = uniform.ModelTrafo.mul(new V4f(v.Positions.x + o.x, v.Positions.y + o.y, v.Positions.z + o.z, 1.0));
  return {
    gl_Position: uniform.ViewProjTrafo.mul(wp),
    color:       uniform.Color,
  };
});

const storageFS = fragment((v: { color: V4f }) => {
  const idx: i32 = (v.color.x + 0.5) as i32;
  return { outColor: Palette[idx] };
});

export function makeHeapTestEffectStorage(): Effect {
  return effect(storageVS, storageFS);
}

// Reads ONLY the derived ModelViewProjTrafo (the heap computes it on the
// GPU from ModelTrafo / ViewTrafo / ProjTrafo) and the colour.
const mvpVS = vertex((v: { Positions: V4f }) => ({
  gl_Position: uniform.ModelViewProjTrafo.mul(v.Positions),
  color:       uniform.Color,
}));

export function makeHeapTestEffectMvp(): Effect {
  return effect(mvpVS, plainFS);
}
