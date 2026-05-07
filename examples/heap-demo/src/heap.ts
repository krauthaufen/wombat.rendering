// heap.ts — thin demo wrapper around the package's heapScene API.
// All the architecture lives in
// `@aardworx/wombat.rendering.experimental/runtime` now; the demo
// just supplies its two fragment shaders ("lambert" / "flat") and
// adapts its local GeometryData type to the package's HeapGeometry
// shape.

import {
  buildHeapScene,
  type HeapDrawSpec as PkgHeapDrawSpec,
  type HeapGeometry,
  type HeapScene,
} from "@aardworx/wombat.rendering.experimental/runtime";
import type { CanvasAttachment } from "@aardworx/wombat.rendering.experimental/window";
import type { aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V4f } from "@aardworx/wombat.base";
import type { GeometryData } from "./geometry.js";

export type ShaderKind = "lambert" | "flat";

const FS_LAMBERT = /* wgsl */`
@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let n = normalize(in.normal);
  let l = normalize(in.lightLoc - in.worldPos);
  let ambient = 0.2;
  let diffuse = abs(dot(l, n));
  let k = ambient + (1.0 - ambient) * diffuse;
  return vec4<f32>(in.color.xyz * k, in.color.w);
}
`;

const FS_FLAT = /* wgsl */`
@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  return in.color;
}
`;

export interface HeapDrawSpec {
  readonly geo: GeometryData;
  readonly modelTrafo: aval<Trafo3d> | Trafo3d;
  readonly color: aval<V4f> | V4f;
  readonly kind: ShaderKind;
}

export type HeapRenderer = HeapScene;

export function buildHeapRenderer(
  device: GPUDevice,
  attach: CanvasAttachment,
  draws: readonly HeapDrawSpec[],
): HeapRenderer {
  const pkgDraws: PkgHeapDrawSpec[] = draws.map(d => ({
    geo: { positions: d.geo.positions, normals: d.geo.normals, indices: d.geo.indices } satisfies HeapGeometry,
    modelTrafo: d.modelTrafo,
    color: d.color,
    groupKey: d.kind,
  }));
  return buildHeapScene(device, attach, pkgDraws, {
    fragmentShaders: new Map<string, string>([
      ["lambert", FS_LAMBERT],
      ["flat",    FS_FLAT],
    ]),
  });
}
