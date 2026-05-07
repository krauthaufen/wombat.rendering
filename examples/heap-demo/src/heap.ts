// heap.ts — thin demo wrapper around the package's heapScene API.
// Phase 9: fragment shaders are now written in the wombat.shader
// DSL and compiled via `compileHeapFragment`. The DSL effect's
// fragment input fields use the heap-scene VsOut names directly
// (worldPos, normal, color, lightLoc); the adapter rewraps the
// emitted WGSL into the heap-scene's FS shape.

import {
  buildHeapScene,
  compileHeapFragment,
  type HeapDrawSpec as PkgHeapDrawSpec,
  type HeapGeometry,
  type HeapScene,
  type HeapTextureSet,
} from "@aardworx/wombat.rendering.experimental/runtime";
import type { CanvasAttachment } from "@aardworx/wombat.rendering.experimental/window";
import type { aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3f, V4f } from "@aardworx/wombat.base";
import { fragment } from "@aardworx/wombat.shader";
import { abs } from "@aardworx/wombat.shader/types";
import type { GeometryData } from "./geometry.js";

export type ShaderKind = "lambert" | "flat" | "textured";

// ─── DSL fragment effects ────────────────────────────────────────────
//
// Inputs use the heap-scene VsOut field names directly. Output is a
// single V4f field (any name; the adapter rewraps to bare
// @location(0) vec4<f32>).

const lambertEffect = fragment((v: {
  worldPos: V3f; normal: V3f; color: V4f; lightLoc: V3f;
}) => {
  const n = v.normal.normalize();
  const l = v.lightLoc.sub(v.worldPos).normalize();
  const ambient = 0.2;
  const diffuse = abs(l.dot(n));
  const k = ambient + (1.0 - ambient) * diffuse;
  return { outColor: new V4f(v.color.xyz.mul(k), v.color.w) };
});

const flatEffect = fragment((v: { color: V4f }) => ({
  outColor: v.color,
}));

// "textured" still uses raw WGSL because it samples a binding (4/5)
// that's outside the DSL's surface — the adapter doesn't (yet)
// model heap-scene's texture-set bindings. Demo this as the
// "escape hatch" path.
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

function buildCheckerTexture(device: GPUDevice): HeapTextureSet {
  const size = 64;
  const bytes = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cell = ((x >> 3) ^ (y >> 3)) & 1;
      const v = cell !== 0 ? 220 : 60;
      const idx = (y * size + x) * 4;
      bytes[idx + 0] = v; bytes[idx + 1] = v; bytes[idx + 2] = v; bytes[idx + 3] = 255;
    }
  }
  const texture = device.createTexture({
    size: { width: size, height: size, depthOrArrayLayers: 1 },
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: "heap-demo: checker",
  });
  device.queue.writeTexture(
    { texture }, bytes,
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
      ["lambert",  compileHeapFragment(lambertEffect)],
      ["flat",     compileHeapFragment(flatEffect)],
      ["textured", FS_TEXTURED],
    ]),
    texturesByGroupKey: new Map<string, HeapTextureSet>([
      ["textured", checker],
    ]),
  });
}
