// renderbench effects — VERBATIM ports of aardvark.rendering-heap's
// HeapSpike RenderBench.Sh shaders (heapVert / bakedVert / lit), so the
// heap-vs-baked gpuMs ratio is comparable across the two engines.
//
//   heapVert:  wp = ModelTrafo · pos;  out = ViewProjTrafo · wp;
//              n  = ModelTrafo.TransformDir n     (per-object matrix gather)
//   bakedVert: out = ViewProjTrafo · pos          (geometry pre-baked to world)
//   lit:       l = normalize(1,2,3); d = 0.25 + 0.75·max(0, n̂·l);  rgb·d, α=1
//
// Colors ride per-vertex (Vienna colors.bin, repacked to tight RGB —
// wombat stores vec4 attrs as tight vec3 with w assembled to 1, and the
// classic vertex fetch fills missing components with (0,0,0,1) too).

import { effect, vertex, fragment } from "@aardworx/wombat.shader";
import { max } from "@aardworx/wombat.shader/types";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { V3f, V4f } from "@aardworx/wombat.base";

// aardvark: Sh.heapVert
const heapVS = vertex((v: {
  Positions: V4f;
  Normals:   V3f;
  Colors:    V4f;
}) => {
  const wp = uniform.ModelTrafo.mul(v.Positions);
  const n4 = uniform.ModelTrafo.mul(new V4f(v.Normals.xyz, 0.0)); // TransformDir
  return {
    gl_Position: uniform.ViewProjTrafo.mul(wp),
    Normals:     n4.xyz,
    Colors:      v.Colors,
  };
});

// aardvark: Sh.bakedVert — VP only, world-baked geometry
const bakedVS = vertex((v: {
  Positions: V4f;
  Normals:   V3f;
  Colors:    V4f;
}) => {
  return {
    gl_Position: uniform.ViewProjTrafo.mul(v.Positions),
    Normals:     v.Normals,
    Colors:      v.Colors,
  };
});

// aardvark: Sh.lit
const litFS = fragment((v: { Normals: V3f; Colors: V4f }) => {
  const l = new V3f(1.0, 2.0, 3.0).normalize();
  const d = max(0.0, v.Normals.normalize().dot(l)) * 0.75 + 0.25;
  return {
    outColor: new V4f(v.Colors.xyz.mul(d), 1.0),
  };
});

export const heapSurface  = effect(heapVS, litFS);
export const bakedSurface = effect(bakedVS, litFS);
