// Minimal trafo + Lambert effect for the heap-demo. Identical
// shape to `wombat.dom/scene/defaultSurfaces.ts` (trafo + simpleLighting)
// but trimmed to what the demo's geometry uses (Positions + Normals
// + Colors). The wombat.shader-vite plugin lifts the inline
// `vertex(...) / fragment(...)` calls into an `Effect` at build time.

import { effect, vertex, fragment } from "@aardworx/wombat.shader";
import { abs } from "@aardworx/wombat.shader/types";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { V3f, V4f } from "@aardworx/wombat.base";

export const trafoVS = vertex((v: {
  Positions: V4f;
  Normals:   V3f;
  Colors:    V4f;
}) => {
  const wp = uniform.ModelTrafo.mul(v.Positions);
  // Inv-transpose normal trick — same as defaultSurfaces.trafo.
  const n4 = new V4f(v.Normals.xyz, 0.0).mul(uniform.ModelTrafoInv);
  return {
    gl_Position:    uniform.ViewProjTrafo.mul(wp),
    WorldPositions: wp,
    Normals:        n4.xyz,
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

export const surface = effect(trafoVS, lambertFS);
