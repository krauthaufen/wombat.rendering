// heap.ts — thin demo wrapper around the package's heapScene API.
// All the architecture lives in
// `@aardworx/wombat.rendering.experimental/runtime` now; the demo
// supplies its three fragment shaders ("lambert" / "flat" /
// "textured"), adapts the local GeometryData to HeapGeometry, and
// builds a small synthetic checker texture for the "textured"
// group.

import {
  buildHeapScene,
  type HeapDrawSpec as PkgHeapDrawSpec,
  type HeapGeometry,
  type HeapScene,
  type HeapTextureSet,
} from "@aardworx/wombat.rendering.experimental/runtime";
import type { CanvasAttachment } from "@aardworx/wombat.rendering.experimental/window";
import type { aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V4f } from "@aardworx/wombat.base";
import type { GeometryData } from "./geometry.js";

export type ShaderKind = "lambert" | "flat" | "textured";

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

// Texture sample uses planar projection on the world-space normal —
// crude but enough to read pixels off the texture and prove the
// per-group bind-layout extension works without UVs in the heap.
const FS_TEXTURED = /* wgsl */`
@group(0) @binding(4) var checker:    texture_2d<f32>;
@group(0) @binding(5) var checkerSmp: sampler;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let n = normalize(in.normal);
  let uv = vec2<f32>(in.worldPos.x, in.worldPos.y) * 0.25 + vec2<f32>(0.5, 0.5);
  let tex = textureSample(checker, checkerSmp, uv).rgb;
  let l = normalize(in.lightLoc - in.worldPos);
  let k = 0.3 + 0.7 * abs(dot(l, n));
  return vec4<f32>(tex * in.color.xyz * k, in.color.w);
}
`;

export interface HeapDrawSpec {
  readonly geo: GeometryData;
  readonly modelTrafo: aval<Trafo3d> | Trafo3d;
  readonly color: aval<V4f> | V4f;
  readonly kind: ShaderKind;
}

export type HeapRenderer = HeapScene;

/** Build a small RGBA8 checkerboard texture for the "textured" group. */
function buildCheckerTexture(device: GPUDevice): HeapTextureSet {
  const size = 64;
  const bytes = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cell = ((x >> 3) ^ (y >> 3)) & 1;
      const v = cell !== 0 ? 220 : 60;
      const idx = (y * size + x) * 4;
      bytes[idx + 0] = v;
      bytes[idx + 1] = v;
      bytes[idx + 2] = v;
      bytes[idx + 3] = 255;
    }
  }
  const texture = device.createTexture({
    size: { width: size, height: size, depthOrArrayLayers: 1 },
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: "heap-demo: checker",
  });
  device.queue.writeTexture(
    { texture },
    bytes,
    { bytesPerRow: size * 4, rowsPerImage: size },
    { width: size, height: size, depthOrArrayLayers: 1 },
  );
  const sampler = device.createSampler({
    magFilter: "nearest", minFilter: "nearest",
    addressModeU: "repeat", addressModeV: "repeat",
    label: "heap-demo: checker sampler",
  });
  return { texture, sampler };
}

export function buildHeapRenderer(
  device: GPUDevice,
  attach: CanvasAttachment,
  draws: readonly HeapDrawSpec[],
): HeapRenderer {
  const checker = buildCheckerTexture(device);
  const pkgDraws: PkgHeapDrawSpec[] = draws.map(d => ({
    geo: { positions: d.geo.positions, normals: d.geo.normals, indices: d.geo.indices } satisfies HeapGeometry,
    modelTrafo: d.modelTrafo,
    color: d.color,
    groupKey: d.kind,
  }));
  return buildHeapScene(device, attach, pkgDraws, {
    fragmentShaders: new Map<string, string>([
      ["lambert",  FS_LAMBERT],
      ["flat",     FS_FLAT],
      ["textured", FS_TEXTURED],
    ]),
    texturesByGroupKey: new Map<string, HeapTextureSet>([
      ["textured", checker],
    ]),
  });
}
