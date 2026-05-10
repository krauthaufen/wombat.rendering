// Trafo + Lambert effects for the heap-demo.
//
// Non-instanced: object → view via ModelViewTrafo (single derived
//   uniform), normals via ModelViewTrafoInv.transpose() — both
//   §7-derived in df32 from Model/View constituents.
// Instanced: object → world via ModelTrafo, world += offset, then
//   world → view via ViewTrafo (so the per-instance translation is
//   applied in world-space). Normals: ModelTrafoInv.transpose() then
//   ViewTrafo (orthonormal) — equivalent to (M·V)⁻ᵀ on view-space.
//
// Lighting is in view-space: camera at origin, l = −normalize(viewPos).

import { effect, vertex, fragment } from "@aardworx/wombat.shader";
import { abs, sin, cos, type Sampler2D, texture } from "@aardworx/wombat.shader/types";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { V2f, V3f, V4f } from "@aardworx/wombat.base";

declare module "@aardworx/wombat.shader/uniforms" {
  interface UniformScope {
    readonly Time: number;
    readonly Tint: V4f;
  }
}

const albedo: Sampler2D = null as unknown as Sampler2D;

// ─── Object → View, via ModelViewTrafo (non-instanced) ────────────────

export const modelViewVS = vertex((v: {
  Positions: V4f;
  Normals:   V3f;
  Colors:    V4f;
}) => {
  const vp = uniform.ModelViewTrafo.mul(v.Positions);
  const n4 = uniform.ModelViewTrafoInv.transpose().mul(new V4f(v.Normals.xyz, 0.0));
  return {
    ViewPositions: vp,
    Normals:       n4.xyz,
    Colors:        v.Colors,
  };
});

// ─── Object → World, via ModelTrafo (instanced chain) ─────────────────

export const modelVS = vertex((v: {
  Positions: V4f;
  Normals:   V3f;
  Colors:    V4f;
}) => {
  const wp = uniform.ModelTrafo.mul(v.Positions);
  const n4 = uniform.ModelTrafoInv.transpose().mul(new V4f(v.Normals.xyz, 0.0));
  return {
    WorldPositions: wp,
    Normals:        n4.xyz,
    Colors:         v.Colors,
  };
});

// ─── World += InstanceOffset (world-space) ────────────────────────────

// InstanceOffset declared as V4f (.w padding) for 16-byte stride —
// avoids vec3-tight (12B) storage layout that may interact badly
// with iOS Safari's MSL backend at certain instanceCount values.
export const instanceOffsetVS = vertex((v: {
  WorldPositions: V4f;
  Normals:        V3f;
  Colors:         V4f;
  InstanceOffset: V4f;
}) => {
  return {
    WorldPositions: new V4f(v.WorldPositions.xyz.add(v.InstanceOffset.xyz), v.WorldPositions.w),
    Normals:        v.Normals,
    Colors:         v.Colors,
  };
});

// ─── World → View, via ViewTrafo (instanced chain) ────────────────────
// ViewTrafo is orthonormal (rotation+translation), so it's its own
// inverse-transpose for direction vectors.

export const viewVS = vertex((v: {
  WorldPositions: V4f;
  Normals:        V3f;
  Colors:         V4f;
}) => {
  const vp = uniform.ViewTrafo.mul(v.WorldPositions);
  const n4 = uniform.ViewTrafo.mul(new V4f(v.Normals.xyz, 0.0));
  return {
    ViewPositions: vp,
    Normals:       n4.xyz,
    Colors:        v.Colors,
  };
});

// ─── View → Clip, via ProjTrafo ───────────────────────────────────────

export const clipVS = vertex((v: {
  ViewPositions: V4f;
  Normals:       V3f;
  Colors:        V4f;
}) => {
  return {
    gl_Position:   uniform.ProjTrafo.mul(v.ViewPositions),
    ViewPositions: v.ViewPositions,
    Normals:       v.Normals,
    Colors:        v.Colors,
  };
});

// ─── View-space Lambert ───────────────────────────────────────────────

export const lambertFS = fragment((v: {
  Normals:       V3f;
  Colors:        V4f;
  ViewPositions: V4f;
}) => {
  const n = v.Normals.normalize();
  const l = v.ViewPositions.xyz.normalize().mul(-1.0);
  const ambient = 0.2;
  const diffuse = abs(l.dot(n));
  const k = ambient + (1.0 - ambient) * diffuse;
  return {
    outColor: new V4f(v.Colors.xyz.mul(k), v.Colors.w),
  };
});

export const surface = effect(modelViewVS, clipVS, lambertFS);

// ─── Textured variant ─────────────────────────────────────────────────

export const uvsVS = vertex((v: { Uvs: V2f }) => ({ Uvs: v.Uvs }));

export const lambertTexturedFS = fragment((v: {
  Normals:       V3f;
  Colors:        V4f;
  ViewPositions: V4f;
  Uvs:           V2f;
}) => {
  const n = v.Normals.normalize();
  const l = v.ViewPositions.xyz.normalize().mul(-1.0);
  const ambient = 0.2;
  const diffuse = abs(l.dot(n));
  const k = ambient + (1.0 - ambient) * diffuse;
  const tex = texture(albedo, v.Uvs);
  const lit = v.Colors.xyz.mul(tex.xyz).mul(k);
  return {
    outColor: new V4f(lit, v.Colors.w),
  };
});

export const texturedSurface = effect(modelViewVS, uvsVS, clipVS, lambertTexturedFS);

// ─── Instanced variants ───────────────────────────────────────────────

export const instancedSurface = effect(modelVS, instanceOffsetVS, viewVS, clipVS, lambertFS);

export const instancedTexturedSurface = effect(
  modelVS, uvsVS, instanceOffsetVS, viewVS, clipVS, lambertTexturedFS,
);

// ─── Time-driven and tinted variants ──────────────────────────────────

// Z-wobble in world-space (between modelVS and viewVS).
export const wobbleVS = vertex((v: {
  WorldPositions: V4f;
  Normals:        V3f;
  Colors:         V4f;
}) => {
  const t = uniform.Time;
  const wob = sin(t.mul(0.002).add(v.WorldPositions.x)).mul(0.3);
  return {
    WorldPositions: new V4f(
      v.WorldPositions.x,
      v.WorldPositions.y,
      v.WorldPositions.z + wob,
      v.WorldPositions.w,
    ),
    Normals: v.Normals,
    Colors:  v.Colors,
  };
});

// UV swirl. Pure UV rewrite.
export const swirlUvsVS = vertex((v: {
  Uvs: V2f;
}) => {
  const t = uniform.Time.mul(0.0007);
  const c = cos(t);
  const s = sin(t);
  const u = v.Uvs.x - 0.5;
  const w = v.Uvs.y - 0.5;
  return {
    Uvs: new V2f(c.mul(u).sub(s.mul(w)).add(0.5), s.mul(u).add(c.mul(w)).add(0.5)),
  };
});

export const tintFS = fragment((v: { outColor: V4f }) => ({
  outColor: new V4f(v.outColor.xyz.mul(uniform.Tint.xyz), v.outColor.w),
}));

export const pulseFS = fragment((v: { outColor: V4f }) => {
  const t = uniform.Time.mul(0.003);
  const k = sin(t).mul(0.4).add(0.6);
  return {
    outColor: new V4f(v.outColor.xyz.mul(k), v.outColor.w),
  };
});

export const tintedSurface  = effect(modelViewVS, clipVS, lambertFS, tintFS);
export const pulsingSurface = effect(modelViewVS, clipVS, lambertFS, pulseFS);
export const wobblingInstancedSurface = effect(
  modelVS, wobbleVS, instanceOffsetVS, viewVS, clipVS, lambertFS,
);
export const swirlingTexturedSurface = effect(
  modelViewVS, uvsVS, swirlUvsVS, clipVS, lambertTexturedFS,
);
