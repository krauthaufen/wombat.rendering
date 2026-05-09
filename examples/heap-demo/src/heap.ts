// heap.ts — thin demo wrapper around the package's heapScene API.
// Each group key gets a single composed Effect (VS+FS authored in
// the wombat.shader DSL). The rendering package's `compileHeapEffect`
// runs the DSL emit and rewires the resulting WGSL to fit the
// heap-scene prelude (storage-buffer uniforms, vertex pulling).

import {
  buildHeapScene,
  type HeapDrawSpec as PkgHeapDrawSpec,
  type HeapScene,
  type HeapTextureSet,
} from "@aardworx/wombat.rendering.experimental/runtime";
import { ITexture, ISampler } from "@aardworx/wombat.rendering.experimental/core";
import type { CanvasAttachment } from "@aardworx/wombat.rendering.experimental/window";
import type { aval } from "@aardworx/wombat.adaptive";
import { cset, cval, transact, AdaptiveToken } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V3f, V4f } from "@aardworx/wombat.base";
import { effect, fragment, vertex } from "@aardworx/wombat.shader";
import { abs, sin } from "@aardworx/wombat.shader/types";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import type { GeometryData } from "./geometry.js";

// Extend the wombat.shader uniform namespace so `uniform.Color`
// type-checks and the inline-marker plugin recognises it. heap-scene
// resolves `Color` at WGSL-rewrite time from `draws[drawIdx].color`.
declare module "@aardworx/wombat.shader/uniforms" {
  interface UniformScope {
    readonly Color: V4f;
  }
}

export type ShaderKind = "lambert" | "flat" | "textured" | "wave";

// ─── DSL stage markers ───────────────────────────────────────────────
//
// Inputs use the heap-scene VsOut field names directly. Output is a
// single V4f field (any name; the adapter rewraps to bare
// @location(0) vec4<f32>).

const lambertFs = fragment((v: {
  worldPos: V3f; normal: V3f; color: V4f; lightLoc: V3f;
}) => {
  const n = v.normal.normalize();
  const l = v.lightLoc.sub(v.worldPos).normalize();
  const ambient = 0.2;
  const diffuse = abs(l.dot(n));
  const k = ambient + (1.0 - ambient) * diffuse;
  return { outColor: new V4f(v.color.xyz.mul(k), v.color.w) };
});

const flatFs = fragment((v: { color: V4f }) => ({
  outColor: v.color,
}));

// Standard transform VS — read positions/normals from the per-vertex
// inputs, apply ModelTrafo, project via ViewProjTrafo, thread Color
// and LightLocation through to the FS via VsOut.
const defaultVs = vertex((v: { Positions: V4f; Normals: V3f }) => {
  const wp = uniform.ModelTrafo.mul(v.Positions);
  const n4 = uniform.ModelTrafo.mul(new V4f(v.Normals.xyz, 0.0));
  return {
    gl_Position: uniform.ViewProjTrafo.mul(wp),
    worldPos:    wp.xyz,
    normal:      n4.xyz,
    color:       uniform.Color,
    lightLoc:    uniform.LightLocation,
  };
});

// Wave-displacement VS — same model trafo, then a vertical sine
// modulation in world space. Demonstrates that the user-supplied VS
// body actually reaches the GPU and produces pixel-different output.
const waveVs = vertex((v: { Positions: V4f; Normals: V3f }) => {
  const wp = uniform.ModelTrafo.mul(v.Positions);
  // The DSL types V4f.x as `number` for ergonomics, but at runtime the
  // value is a proxy with `.mul`/`.add` that builds shader IR. TS sees
  // a plain number; the runtime sees the proxy. (No scalar wrapper
  // type exists in @aardworx/wombat.shader yet.)
  // @ts-expect-error DSL scalars are proxied at runtime
  const dz = sin(wp.x.mul(0.5)).mul(0.4);
  // @ts-expect-error DSL scalars are proxied at runtime
  const wpw = new V4f(wp.x, wp.y, wp.z.add(dz), wp.w);
  const n4 = uniform.ModelTrafo.mul(new V4f(v.Normals.xyz, 0.0));
  return {
    gl_Position: uniform.ViewProjTrafo.mul(wpw),
    worldPos:    wpw.xyz,
    normal:      n4.xyz,
    color:       uniform.Color,
    lightLoc:    uniform.LightLocation,
  };
});

// Composed effects — one per group key. The DSL cross-stage linker
// aligns the VS output struct to the FS input struct and DCEs unread
// VS outputs.
const lambertEffect = effect(defaultVs, lambertFs);
const waveEffect    = effect(waveVs,    lambertFs);
const flatEffect    = effect(defaultVs, flatFs);

// "textured" still rides the raw-WGSL escape hatch — the DSL doesn't
// model heap-scene's per-group texture set bindings (4/5) yet. The
// VS half is the same body the DSL emits for `defaultVs`; the FS
// samples a checker texture and modulates by the lambert kernel.
// Raw-WGSL escape hatch: this shader must read directly from the
// arena's typed views (no per-bucket prelude helpers any more —
// step 7 inlined them). The schema baked into RAW_DEFAULT_SCHEMA
// gives us the layout: stride = 8 u32s, refs at u32 offsets:
//   0 = ModelTrafoRef, 1 = ColorRef, 2 = ViewProjTrafoRef,
//   3 = LightLocationRef, 4 = PositionsRef, 5 = NormalsRef.
const TEXTURED_VS_WGSL = /* wgsl */`
@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) drawIdx: u32) -> VsOut {
  let mtRef  = headersU32[drawIdx * 8u + 0u];
  let cRef   = headersU32[drawIdx * 8u + 1u];
  let vpRef  = headersU32[drawIdx * 8u + 2u];
  let llRef  = headersU32[drawIdx * 8u + 3u];
  let posRef = headersU32[drawIdx * 8u + 4u];
  let norRef = headersU32[drawIdx * 8u + 5u];

  let posBase = (posRef + 16u) / 4u + vid * 3u;
  let pos = vec3<f32>(heapF32[posBase], heapF32[posBase + 1u], heapF32[posBase + 2u]);
  let norBase = (norRef + 16u) / 4u + vid * 3u;
  let nor = vec3<f32>(heapF32[norBase], heapF32[norBase + 1u], heapF32[norBase + 2u]);

  let mtB = (mtRef + 16u) / 16u;
  let M   = mat4x4<f32>(heapV4f[mtB], heapV4f[mtB + 1u], heapV4f[mtB + 2u], heapV4f[mtB + 3u]);
  let vpB = (vpRef + 16u) / 16u;
  let VP  = mat4x4<f32>(heapV4f[vpB], heapV4f[vpB + 1u], heapV4f[vpB + 2u], heapV4f[vpB + 3u]);
  let color = heapV4f[(cRef + 16u) / 16u];
  let llB = (llRef + 16u) / 4u;
  let lightLoc = vec3<f32>(heapF32[llB], heapF32[llB + 1u], heapF32[llB + 2u]);

  let wp = vec4<f32>(pos, 1.0) * M;
  let n  = (vec4<f32>(nor, 0.0) * M).xyz;
  var out: VsOut;
  out.clipPos  = wp * VP;
  out.worldPos = wp.xyz;
  out.normal   = n;
  out.color    = color;
  out.lightLoc = lightLoc;
  return out;
}
`;

const TEXTURED_FS_WGSL = /* wgsl */`
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

const texturedShader = { vs: TEXTURED_VS_WGSL, fs: TEXTURED_FS_WGSL };

export interface HeapDrawSpec {
  readonly geo: GeometryData;
  readonly modelTrafo: aval<Trafo3d> | Trafo3d;
  readonly color: aval<V4f> | V4f;
  readonly kind: ShaderKind;
}

export type HeapRenderer = Omit<HeapScene, "frame"> & {
  /** Demo wrapper: pushes viewProj/light cvals then runs scene.frame internally. */
  frame(viewProj: Trafo3d, lightLocation: V3d): void;
};

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
  return {
    kind: "standalone",
    texture: ITexture.fromGPU(texture),
    sampler: ISampler.fromGPU(sampler),
  };
}

function specOf(
  d: HeapDrawSpec, checker: HeapTextureSet,
  viewProj: aval<Trafo3d>, light: aval<V3d>,
): PkgHeapDrawSpec {
  // One generic inputs map covering both vertex attributes
  // (Positions, Normals) and uniforms (ModelTrafo, Color, …). The
  // runtime walks the effect's schema and pulls each by name.
  const inputs = {
    Positions:     d.geo.positions,
    Normals:       d.geo.normals,
    ModelTrafo:    d.modelTrafo,
    Color:         d.color,
    ViewProjTrafo: viewProj,
    LightLocation: light,
  };
  const base = { inputs, indices: d.geo.indices };
  switch (d.kind) {
    case "lambert":  return { ...base, effect: lambertEffect };
    case "wave":     return { ...base, effect: waveEffect };
    case "flat":     return { ...base, effect: flatEffect };
    case "textured": return { ...base, effect: texturedShader, textures: checker };
  }
}

/**
 * Build an instanced render-object: one draw, N transforms, sharing
 * the same geometry + color. Demonstrates the runtime's shape-2 path
 * (`spec.instances`) — the GPU sees a single `drawIndexed(_, N, …)`
 * and the WGSL reads `ModelTrafo` per-instance from a packed array.
 */
export interface HeapInstancedSpec {
  readonly geo: GeometryData;
  /** One transform per instance; length must be ≥ `count`. */
  readonly transforms: readonly Trafo3d[];
  readonly count: number;
  readonly color: V4f;
  readonly kind: "lambert" | "flat";
}

export function buildHeapRenderer(
  device: GPUDevice,
  attach: CanvasAttachment,
  draws: readonly HeapDrawSpec[],
  instanced?: readonly HeapInstancedSpec[],
): HeapRenderer {
  const checker = buildCheckerTexture(device);

  // Per-frame uniforms are now per-draw avals shared by every spec.
  // One cval each → one heap allocation each, regardless of N draws.
  // The wrapper's `frame()` pushes new values via transact() and
  // the renderer's pool repacks the single allocation once.
  const viewProjCval = cval<Trafo3d>(Trafo3d.identity);
  const lightCval    = cval<V3d>(new V3d(0, 0, 0));

  const initial: PkgHeapDrawSpec[] = draws.map(d => specOf(d, checker, viewProjCval, lightCval));

  // Shape-2 instanced draws — each becomes ONE PkgHeapDrawSpec with
  // an `instances` map. ModelTrafo is per-instance; the rest (color,
  // viewProj, light, attribs) stay per-draw and share via aval id.
  if (instanced !== undefined) {
    for (const i of instanced) {
      const eff = i.kind === "flat" ? flatEffect : lambertEffect;
      initial.push({
        effect: eff,
        inputs: {
          Positions:     i.geo.positions,
          Normals:       i.geo.normals,
          Color:         i.color,
          ViewProjTrafo: viewProjCval,
          LightLocation: lightCval,
        },
        indices: i.geo.indices,
        instances: {
          count: i.count,
          values: { ModelTrafo: i.transforms },
        },
      });
    }
  }

  // Build via a `cset` so the renderer is driven from an adaptive
  // set; the static list is just the initial population.
  // The fragmentOutputLayout pins our effects' single `outColor`
  // output to the framebuffer's color0 slot. Outputs the effect
  // could declare but the FB doesn't accept would be DCE'd by
  // `linkFragmentOutputs`, dragging any uniforms that fed only those
  // dead outputs out of the schema too.
  const drawSet = cset<PkgHeapDrawSpec>(initial);
  const scene = buildHeapScene(device, attach.signature, drawSet, {
    fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
  });

  // ─── aset-driven mutation smoke tests ────────────────────────────
  if (draws.length > 0) {
    const probe = draws[0]!;
    const probeSpec = specOf(probe, checker, viewProjCval, lightCval);
    const before = scene.stats.totalDraws;
    transact(() => { drawSet.add(probeSpec); });
    scene.update(AdaptiveToken.top);
    console.log(`[heap-demo] aset add probe: before=${before} after=${scene.stats.totalDraws}`);
    transact(() => { drawSet.remove(probeSpec); });
    scene.update(AdaptiveToken.top);
    console.log(`[heap-demo] aset remove probe: after=${scene.stats.totalDraws}`);

    // Direct addDraw/removeDraw still works alongside the aset path.
    const slot = scene.addDraw(probeSpec);
    console.log(`[heap-demo] direct addDraw probe: slot=${slot} count=${scene.stats.totalDraws}`);
    scene.removeDraw(slot);
    console.log(`[heap-demo] direct removeDraw probe: count=${scene.stats.totalDraws}`);
  }

  // Wrap scene.frame so main.ts's existing call-shape (frame(viewProj,
  // light)) keeps working. Behind the scenes we push the values into
  // the shared cvals via transact(), force the framebuffer aval, then
  // hand both off to the package's frame(framebuffer, token).
  const fbAval = attach.framebuffer as aval<import("@aardworx/wombat.rendering.experimental/core").IFramebuffer>;
  // Spread scene minus its `frame` so the wrapper's frame signature
  // (viewProj, light) doesn't collide with HeapScene's (framebuffer,
  // token). Destructure rather than `...scene` to avoid TS picking up
  // the package's frame as the wrapper's type.
  const { frame: _packageFrame, ...rest } = scene;
  void _packageFrame;
  return {
    ...rest,
    frame(viewProj: Trafo3d, lightLocation: V3d): void {
      transact(() => {
        viewProjCval.value = viewProj;
        lightCval.value    = lightLocation;
      });
      scene.frame(fbAval.force(/* allow-force */), AdaptiveToken.top);
    },
  };
}
