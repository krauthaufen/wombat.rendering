// Minimal trafo + Lambert effect for the heap-demo. Identical
// shape to `wombat.dom/scene/defaultSurfaces.ts` (trafo + simpleLighting)
// but trimmed to what the demo's geometry uses (Positions + Normals
// + Colors). The wombat.shader-vite plugin lifts the inline
// `vertex(...) / fragment(...)` calls into an `Effect` at build time.

import { effect, vertex, fragment } from "@aardworx/wombat.shader";
import { abs, type Sampler2D, texture } from "@aardworx/wombat.shader/types";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { V2f, V3f, V4f } from "@aardworx/wombat.base";

// Sampler2D capture. The wombat.shader-vite plugin classifies any
// non-ambient free identifier whose TS type maps to a Sampler IR
// type as a `uniform-sampler` capture and emits a Sampler ValueDef
// with the same name into the IR. At draw time the runtime matches
// the name against `RenderObject.textures` + `RenderObject.samplers`.
// The runtime value is never read (samplers are opaque texture
// handles bound by name).
const albedo: Sampler2D = null as unknown as Sampler2D;

// Split into modelVS (object→world) + clipVS (world→clip) so an
// instance-offset modifier can be inserted between them without
// duplicating either piece. Same observable behaviour as the old
// monolithic `trafoVS`; same-stage fusion (`composeStages`) recombines
// them via `extractFusedEntry`.

export const modelVS = vertex((v: {
  Positions: V4f;
  Normals:   V3f;
  Colors:    V4f;
}) => {
  const wp = uniform.ModelTrafo.mul(v.Positions);
  const n4 = new V4f(v.Normals.xyz, 0.0);   // raw object-space normal, no matrix
  return {
    WorldPositions: wp,
    Normals:        n4.xyz,
    Colors:         v.Colors,
  };
});

// Reads the current `WorldPositions` (whatever upstream stage produced
// it) and adds the per-instance offset attribute. Pass-through for
// Normals + Colors — they're untouched by the offset.
export const instanceOffsetVS = vertex((v: {
  WorldPositions: V4f;
  Normals:        V3f;
  Colors:         V4f;
  InstanceOffset: V3f;
}) => {
  return {
    WorldPositions: new V4f(v.WorldPositions.xyz.add(v.InstanceOffset), v.WorldPositions.w),
    Normals:        v.Normals,
    Colors:         v.Colors,
  };
});

export const clipVS = vertex((v: {
  WorldPositions: V4f;
  Normals:        V3f;
  Colors:         V4f;
}) => {
  return {
    gl_Position:    uniform.ViewProjTrafo.mul(v.WorldPositions),
    WorldPositions: v.WorldPositions,
    Normals:        v.Normals,
    Colors:         v.Colors,
  };
});

export const lambertFS = fragment((v: {
  Normals:        V3f;
  Colors:         V4f;
  WorldPositions: V4f;
}) => {
  const n  = v.Normals.normalize();
  const wp = v.WorldPositions.xyz;
  // Camera-headlight: light follows the camera.
  const l  = uniform.LightLocation.sub(wp).normalize();
  const ambient = 0.2;
  const diffuse = abs(l.dot(n));
  const k = ambient + (1.0 - ambient) * diffuse;
  return {
    outColor: new V4f(v.Colors.xyz.mul(k), v.Colors.w),
  };
});

export const surface = effect(modelVS, clipVS, lambertFS);

// ─── Textured variant ─────────────────────────────────────────────────
// Same lambert lighting as `lambertFS`, with the base surface colour
// modulated by `sample(albedo, Uvs)`. Per-vertex UVs come from the
// geometry: planar per-face UVs for box, lat/long parameterization
// for sphere, side wraps + cap projection for cylinder.

export const trafoTexturedVS = vertex((v: {
  Positions: V4f;
  Normals:   V3f;
  Colors:    V4f;
  Uvs:       V2f;
}) => {
  const wp = uniform.ModelTrafo.mul(v.Positions);
  const n4 = new V4f(v.Normals.xyz, 0.0);
  return {
    gl_Position:    uniform.ViewProjTrafo.mul(wp),
    WorldPositions: wp,
    Normals:        n4.xyz,
    Colors:         v.Colors,
    Uvs:            v.Uvs,
  };
});

export const lambertTexturedFS = fragment((v: {
  Normals:        V3f;
  Colors:         V4f;
  WorldPositions: V4f;
  Uvs:            V2f;
}) => {
  const n  = v.Normals.normalize();
  const wp = v.WorldPositions.xyz;
  const l  = uniform.LightLocation.sub(wp).normalize();
  const ambient = 0.2;
  const diffuse = abs(l.dot(n));
  const k = ambient + (1.0 - ambient) * diffuse;
  const tex = texture(albedo, v.Uvs);
  const lit = v.Colors.xyz.mul(tex.xyz).mul(k);
  return {
    outColor: new V4f(lit, v.Colors.w),
  };
});

export const texturedSurface = effect(trafoTexturedVS, lambertTexturedFS);

// Tiny passthrough stage that declares `Uvs` as both a vertex input
// and an inter-stage carrier. Composed-VS chains pick it up and
// surface Uvs in the merged outputs; the textured FS reads it. No
// math — the optimizer DCEs it down to a wire when present, and
// drops it entirely when no FS consumes Uvs.
export const uvsVS = vertex((v: { Uvs: V2f }) => ({ Uvs: v.Uvs }));

// ─── Instanced variants ───────────────────────────────────────────────
// Proper composition: model → instanceOffset → clip → lambert. No
// duplication; `composeStages` fuses the VS stages via
// `extractFusedEntry`, and the per-stage emit's `pruneToStage` keeps
// the FS module free of VS-only helper bodies.

export const instancedSurface = effect(modelVS, instanceOffsetVS, clipVS, lambertFS);

// Textured instanced variant — same VS chain plus a `uvsVS` stage
// that surfaces the per-vertex Uvs into the inter-stage carrier so
// `lambertTexturedFS` can sample the atlas.
export const instancedTexturedSurface = effect(
  modelVS, uvsVS, instanceOffsetVS, clipVS, lambertTexturedFS,
);
