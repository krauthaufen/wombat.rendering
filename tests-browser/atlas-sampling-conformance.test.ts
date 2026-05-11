// Conformance test for the heap atlas sample math.
//
// Black-box: place a non-pow2 source texture inside a larger "atlas"
// at a deliberate non-aligned offset, with the gutter layout the
// shader expects (2 px per side: inner clamp ring + outer wrap ring).
// In one compute pass, for each pixel of a dense UV sweep, sample the
// source via two paths:
//   - reference: native hardware bilinear on the source's own texture
//   - atlas:    our atlasSample() function reading from the atlas.
// Read back the per-pixel absolute difference; assert ≤ 1/255.
//
// Drives across { addressModeU/V } × { LOD 0 } for now; mips come
// once the compute mip+gutter kernel lands. dpdx/dpdy LOD is tested
// separately under a render pass — here we pin LOD with
// textureSampleLevel to isolate the sample math.

import { describe, expect, it } from "vitest";
import { mipOffsetInPyramid, defaultMipCount } from "../packages/rendering/src/runtime/textureAtlas/atlasPool.js";
import { requestRealDevice } from "./_realGpu.js";

const SRC_W = 127;
const SRC_H = 93;
const ATLAS_SIZE = 512;
const DST_X = 211; // non-aligned offset inside the atlas
const DST_Y = 137;
const SWEEP = 256; // output texture dimension (UV grid)

type WrapMode = "clamp" | "repeat" | "mirror";
const wrapCode = (m: WrapMode): number => m === "clamp" ? 0 : m === "repeat" ? 1 : 2;
const wrapGpu = (m: WrapMode): GPUAddressMode =>
  m === "clamp" ? "clamp-to-edge" : m === "repeat" ? "repeat" : "mirror-repeat";

// 2×2 box-filter downscale: src (w×h) → dst (max(1,w>>1) × max(1,h>>1)).
// Matches the canvas-2d 'high'-quality downscale closely enough for an
// atlas mip pyramid; deterministic regardless of the platform.
function downscaleBoxFilter(src: Uint8Array, w: number, h: number): { data: Uint8Array; w: number; h: number } {
  const dw = Math.max(1, w >> 1);
  const dh = Math.max(1, h >> 1);
  const out = new Uint8Array(dw * dh * 4);
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      const sx = x * 2;
      const sy = y * 2;
      const sx2 = Math.min(sx + 1, w - 1);
      const sy2 = Math.min(sy + 1, h - 1);
      const i00 = (sy * w + sx) * 4;
      const i10 = (sy * w + sx2) * 4;
      const i01 = (sy2 * w + sx) * 4;
      const i11 = (sy2 * w + sx2) * 4;
      const o = (y * dw + x) * 4;
      for (let c = 0; c < 4; c++) {
        out[o + c] = Math.round((src[i00 + c] + src[i10 + c] + src[i01 + c] + src[i11 + c]) * 0.25);
      }
    }
  }
  return { data: out, w: dw, h: dh };
}

// Synthetic source: gradient × checkerboard. High-frequency content
// in the green channel so bilinear filtering is visible.
function makeSource(w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      out[i + 0] = Math.floor((x / (w - 1)) * 255);        // R: horizontal gradient
      out[i + 1] = ((x + y) & 1) === 0 ? 255 : 0;          // G: checker
      out[i + 2] = Math.floor((y / (h - 1)) * 255);        // B: vertical gradient
      out[i + 3] = 255;
    }
  }
  return out;
}

// Place `src` (srcW × srcH) inside `atlas` (atlasW × atlasH) at
// (dstX, dstY), filling a 2-px gutter around it: inner ring = clamp-
// replicate (nearest edge), outer ring = wrap (opposite edge).
function placeWithGutter(
  atlas: Uint8Array, atlasW: number, atlasH: number,
  src: Uint8Array, srcW: number, srcH: number,
  dstX: number, dstY: number,
): void {
  type Mode = "clamp" | "wrap";
  const pickX = (dx: number): { x: number; mode: Mode } => {
    if (dx === -2) return { x: srcW - 1, mode: "wrap" };  // outer ring: wrap → opposite edge
    if (dx === -1) return { x: 0, mode: "clamp" };        // inner ring: clamp → first interior col
    if (dx === srcW) return { x: srcW - 1, mode: "clamp" };
    if (dx === srcW + 1) return { x: 0, mode: "wrap" };
    return { x: dx, mode: "clamp" };
  };
  const pickY = (dy: number): { y: number; mode: Mode } => {
    if (dy === -2) return { y: srcH - 1, mode: "wrap" };
    if (dy === -1) return { y: 0, mode: "clamp" };
    if (dy === srcH) return { y: srcH - 1, mode: "clamp" };
    if (dy === srcH + 1) return { y: 0, mode: "wrap" };
    return { y: dy, mode: "clamp" };
  };
  for (let dy = -2; dy < srcH + 2; dy++) {
    for (let dx = -2; dx < srcW + 2; dx++) {
      const { x: sx } = pickX(dx);
      const { y: sy } = pickY(dy);
      const ax = dstX + dx;
      const ay = dstY + dy;
      if (ax < 0 || ax >= atlasW || ay < 0 || ay >= atlasH) continue;
      const si = (sy * srcW + sx) * 4;
      const ai = (ay * atlasW + ax) * 4;
      atlas[ai + 0] = src[si + 0];
      atlas[ai + 1] = src[si + 1];
      atlas[ai + 2] = src[si + 2];
      atlas[ai + 3] = src[si + 3];
    }
  }
}

const COMPARE_WGSL = /* wgsl */`
struct Params {
  origin_px: vec2<f32>,
  size_px:   vec2<f32>,
  page_size: f32,
  addrU:     u32,
  addrV:     u32,
  lod:       f32,
  sweep:     u32,
  _pad:      vec3<u32>,
};

@group(0) @binding(0) var atlasTex: texture_2d<f32>;
@group(0) @binding(1) var atlasSamp: sampler;
@group(0) @binding(2) var refTex:   texture_2d<f32>;
@group(0) @binding(3) var refSamp:  sampler;
@group(0) @binding(4) var diffOut:  texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var<uniform> P: Params;

fn mirrorUv(u: f32) -> f32 {
  let t = u - floor(u * 0.5) * 2.0;
  return 1.0 - abs(t - 1.0);
}

// Hardware-faithful formula: atlas_p = origin + processed_uv * size.
// Texel-k center lands at atlas_p = origin + k + 0.5 (matching native).
// No half-texel inset; FP jitter at the boundary is absorbed by the
// clamp-gutter cells (col origin-1 etc.) which contain edge-replicate
// data — that's why every mode relies on the inner gutter ring being
// populated.
//
// For repeat: shift atlas_p by ∓1 inside the 0.5-px seam region so
// hardware bilinear straddles the wrap-gutter ring (col origin-2 =
// opposite-edge texel) and the clamp-gutter cell — giving correct
// seam interpolation between T[s-1] and T[0].
fn atlasAxis(uv: f32, origin: f32, size: f32, mode: u32) -> f32 {
  if (mode == 0u) {           // clamp
    let c = clamp(uv, 0.0, 1.0);
    return origin + c * size;
  }
  if (mode == 2u) {           // mirror
    let m = mirrorUv(uv);
    return origin + m * size;
  }
  // repeat
  let f = uv - floor(uv);
  var p = origin + f * size;
  let dL = p - origin;
  let dR = (origin + size) - p;
  if (dL < 0.5) { p = p - 1.0; }
  else if (dR < 0.5) { p = p + 1.0; }
  return p;
}

fn atlasSample(uv: vec2<f32>) -> vec4<f32> {
  let px = atlasAxis(uv.x, P.origin_px.x, P.size_px.x, P.addrU);
  let py = atlasAxis(uv.y, P.origin_px.y, P.size_px.y, P.addrV);
  // Manual bilinear via textureLoad. Bypasses the hardware filter unit
  // (whose weight precision is implementation-defined per WebGPU spec)
  // so the result is bit-deterministic against textureSampleLevel on
  // the reference path, with the only remaining slack being whatever
  // the hardware filter does on the reference side.
  let p = vec2<f32>(px - 0.5, py - 0.5);
  let lo = vec2<i32>(i32(floor(p.x)), i32(floor(p.y)));
  let fr = p - vec2<f32>(floor(p.x), floor(p.y));
  // Atlas always samples mip-0 of the page texture — the mip pyramid
  // is embedded as sub-rects inside mip-0 of the atlas (caller passes
  // mip-k's origin/size for the LOD being tested). The reference path
  // uses its texture's actual mip level via P.lod.
  let t00 = textureLoad(atlasTex, lo + vec2<i32>(0, 0), 0);
  let t10 = textureLoad(atlasTex, lo + vec2<i32>(1, 0), 0);
  let t01 = textureLoad(atlasTex, lo + vec2<i32>(0, 1), 0);
  let t11 = textureLoad(atlasTex, lo + vec2<i32>(1, 1), 0);
  let tx0 = mix(t00, t10, fr.x);
  let tx1 = mix(t01, t11, fr.x);
  return mix(tx0, tx1, fr.y);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= P.sweep || id.y >= P.sweep) { return; }
  // Sweep uv ∈ [-0.25, 1.25]² so we exercise wrap behaviour outside [0,1].
  let s = f32(P.sweep - 1u);
  let u = mix(-0.25, 1.25, f32(id.x) / s);
  let v = mix(-0.25, 1.25, f32(id.y) / s);
  let uv = vec2<f32>(u, v);

  let a = atlasSample(uv);
  let r = textureSampleLevel(refTex, refSamp, uv, P.lod);
  let d = abs(a - r);
  textureStore(diffOut, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(d.rgb, 1.0));
}
`;

async function readbackRgba(device: GPUDevice, tex: GPUTexture): Promise<Uint8Array> {
  const w = tex.width, h = tex.height;
  const bpr = Math.ceil((w * 4) / 256) * 256;
  const staging = device.createBuffer({
    size: bpr * h,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer(
    { texture: tex },
    { buffer: staging, bytesPerRow: bpr, rowsPerImage: h },
    { width: w, height: h, depthOrArrayLayers: 1 },
  );
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    out.set(padded.subarray(y * bpr, y * bpr + w * 4), y * w * 4);
  }
  return out;
}

async function runCompareAt(
  device: GPUDevice,
  atlasTex: GPUTexture,
  refTex: GPUTexture,
  wrapU: WrapMode,
  wrapV: WrapMode,
  lod: number,
  originPx: { x: number; y: number },
  sizePx: { w: number; h: number },
  // The "logical" source dimensions used by wrap/border classification
  // — at mip-k these are the mip-k pixel dims (NOT mip-0's).
  srcW: number,
  srcH: number,
): Promise<{
  histAll: number[]; histInterior: number[]; histBorder: number[];
  interiorCount: number; borderCount: number;
  meanAbs: [number, number, number];
  firstBorderFails: { x: number; y: number; d: number }[];
}> {
  // Diff output
  const diff = device.createTexture({
    size: { width: SWEEP, height: SWEEP },
    format: "rgba8unorm",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  // Samplers: both use the same filter setup; the wrap mode only
  // matters for the reference texture (atlas does wrap in shader).
  const atlasSamp = device.createSampler({
    magFilter: "linear", minFilter: "linear", mipmapFilter: "linear",
    addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
  });
  const refSamp = device.createSampler({
    magFilter: "linear", minFilter: "linear", mipmapFilter: "linear",
    addressModeU: wrapGpu(wrapU),
    addressModeV: wrapGpu(wrapV),
  });
  // Params UBO
  const params = new ArrayBuffer(64);
  const pf = new Float32Array(params);
  const pu = new Uint32Array(params);
  pf[0] = originPx.x; pf[1] = originPx.y;
  pf[2] = sizePx.w;   pf[3] = sizePx.h;
  pf[4] = ATLAS_SIZE;
  pu[5] = wrapCode(wrapU);
  pu[6] = wrapCode(wrapV);
  pf[7] = lod;
  pu[8] = SWEEP;
  const ubo = device.createBuffer({
    size: params.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(ubo, 0, params);
  // Pipeline
  const mod = device.createShaderModule({ code: COMPARE_WGSL });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: mod, entryPoint: "main" },
  });
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: atlasTex.createView() },
      { binding: 1, resource: atlasSamp },
      { binding: 2, resource: refTex.createView() },
      { binding: 3, resource: refSamp },
      { binding: 4, resource: diff.createView() },
      { binding: 5, resource: { buffer: ubo } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(SWEEP / 8), Math.ceil(SWEEP / 8), 1);
  pass.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();

  const image = await readbackRgba(device, diff);
  // Partition pixels into interior vs border. Border = sample's bilinear
  // footprint straddles a source edge (texel_coord floor ∈ {-1, 0, s-2,
  // s-1} on either axis). The gutter, wrap-shift, mirror-fold, and
  // clamp-saturate logic all converge here, so a separate histogram
  // catches systematic edge bugs that the interior bulk would mask.
  const histAll: number[] = [];
  const histInterior: number[] = [];
  const histBorder: number[] = [];
  // Per-channel signed sums for systematic-offset detection. (atlas - ref)
  // — but we only have abs(d) from the diff texture, so we approximate
  // by tracking the abs(d) totals; a true signed mean would need another
  // pass that emits signed deltas in a separate rg16f target. For v1 the
  // abs-mean is sufficient to flag "is the tail symmetric or skewed."
  const signedAbs: [number, number, number] = [0, 0, 0];
  let interiorCount = 0, borderCount = 0;
  const firstBorderFails: { x: number; y: number; d: number }[] = [];

  const wrap = (u: number, mode: WrapMode): number => {
    if (mode === "clamp") return Math.max(0, Math.min(1, u));
    if (mode === "repeat") return u - Math.floor(u);
    // mirror: 1 - abs((u - floor(u/2)*2) - 1)
    const t = u - Math.floor(u * 0.5) * 2;
    return 1 - Math.abs(t - 1);
  };
  const isBorderAxis = (procUv: number, size: number): boolean => {
    const tc = procUv * size - 0.5;
    const lo = Math.floor(tc);
    return lo <= 0 || lo >= size - 2;
  };

  for (let py = 0; py < SWEEP; py++) {
    for (let px = 0; px < SWEEP; px++) {
      const i = (py * SWEEP + px) * 4;
      const dR = image[i], dG = image[i + 1], dB = image[i + 2];
      const d = Math.max(dR, dG, dB);
      histAll[d] = (histAll[d] ?? 0) + 1;
      signedAbs[0] += dR; signedAbs[1] += dG; signedAbs[2] += dB;

      // Recover uv for this sweep pixel (matches compute kernel mapping).
      const u = -0.25 + 1.5 * (px / (SWEEP - 1));
      const v = -0.25 + 1.5 * (py / (SWEEP - 1));
      const pu = wrap(u, wrapU);
      const pv = wrap(v, wrapV);
      const onBorder = isBorderAxis(pu, srcW) || isBorderAxis(pv, srcH);
      if (onBorder) {
        borderCount++;
        histBorder[d] = (histBorder[d] ?? 0) + 1;
        if (d > 0 && firstBorderFails.length < 8) {
          firstBorderFails.push({ x: px, y: py, d });
        }
      } else {
        interiorCount++;
        histInterior[d] = (histInterior[d] ?? 0) + 1;
      }
    }
  }
  const totalPx = SWEEP * SWEEP;
  const meanAbs: [number, number, number] = [
    signedAbs[0] / totalPx, signedAbs[1] / totalPx, signedAbs[2] / totalPx,
  ];
  diff.destroy();
  ubo.destroy();
  return {
    histAll, histInterior, histBorder,
    interiorCount, borderCount,
    meanAbs, firstBorderFails,
  };
}

/** Cumulative-fraction-at-most-d for a histogram of integer diffs. */
function cdfAtMost(hist: number[], d: number, total: number): number {
  if (total === 0) return 1;
  let acc = 0;
  for (let i = 0; i <= d; i++) acc += hist[i] ?? 0;
  return acc / total;
}

function formatHist(hist: number[], total: number): string {
  return hist
    .map((c, d) => c > 0 ? `${d}:${c}(${((c / total) * 100).toFixed(2)}%)` : "")
    .filter(Boolean)
    .slice(0, 8)
    .join(" ");
}

describe("atlas-sampling conformance", () => {
  it("matches native hardware sampling within 1 LSB across clamp / repeat / mirror at LOD 0", async () => {
    const device = await requestRealDevice();

    // Build source + atlas pixel data
    const src = makeSource(SRC_W, SRC_H);
    const atlas = new Uint8Array(ATLAS_SIZE * ATLAS_SIZE * 4);
    // Fill atlas with magenta first so gutter mistakes are visible.
    for (let i = 0; i < atlas.length; i += 4) {
      atlas[i + 0] = 255; atlas[i + 1] = 0; atlas[i + 2] = 255; atlas[i + 3] = 255;
    }
    placeWithGutter(atlas, ATLAS_SIZE, ATLAS_SIZE, src, SRC_W, SRC_H, DST_X, DST_Y);

    // GPU textures
    const atlasTex = device.createTexture({
      size: { width: ATLAS_SIZE, height: ATLAS_SIZE },
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: atlasTex },
      atlas,
      { bytesPerRow: ATLAS_SIZE * 4, rowsPerImage: ATLAS_SIZE },
      { width: ATLAS_SIZE, height: ATLAS_SIZE, depthOrArrayLayers: 1 },
    );

    const refTex = device.createTexture({
      size: { width: SRC_W, height: SRC_H },
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: refTex },
      src,
      { bytesPerRow: SRC_W * 4, rowsPerImage: SRC_H },
      { width: SRC_W, height: SRC_H, depthOrArrayLayers: 1 },
    );

    const cases: [WrapMode, WrapMode][] = [
      ["clamp",  "clamp"],
      ["repeat", "repeat"],
      ["mirror", "mirror"],
      ["repeat", "clamp"],
      ["clamp",  "mirror"],
    ];
    // Atlas path uses manual bilinear (textureLoad + mix), so we
    // expect bit-exact agreement with native textureSampleLevel on
    // the reference — at minimum on the interior bulk. The border
    // partition is reported separately so a future hardware-filter
    // tolerance can be applied there without masking interior bugs.
    const failures: string[] = [];
    for (const [u, v] of cases) {
      const res = await runCompareAt(
        device, atlasTex, refTex, u, v, 0,
        { x: DST_X, y: DST_Y }, { w: SRC_W, h: SRC_H }, SRC_W, SRC_H,
      );
      const intTotal = res.interiorCount;
      const borTotal = res.borderCount;
      const intCdf0 = cdfAtMost(res.histInterior, 0, intTotal);
      const intCdf1 = cdfAtMost(res.histInterior, 1, intTotal);
      const borCdf0 = cdfAtMost(res.histBorder, 0, borTotal);
      const borCdf1 = cdfAtMost(res.histBorder, 1, borTotal);
      const borCdf2 = cdfAtMost(res.histBorder, 2, borTotal);
      const meanOk = res.meanAbs.every(m => m < 0.05); // < 0.05 LSB

      const caseFails: string[] = [];
      if (intCdf0 < 1.0)                caseFails.push(`interior diff=0 frac ${(intCdf0*100).toFixed(3)}% < 100%`);
      if (intCdf1 < 1.0)                caseFails.push(`interior diff≤1 frac ${(intCdf1*100).toFixed(3)}% < 100%`);
      if (borTotal > 0 && borCdf0 < 1.0) caseFails.push(`border diff=0 frac ${(borCdf0*100).toFixed(3)}% < 100%`);
      if (borTotal > 0 && borCdf1 < 1.0) caseFails.push(`border diff≤1 frac ${(borCdf1*100).toFixed(3)}% < 100%`);
      if (borTotal > 0 && borCdf2 < 1.0) caseFails.push(`border diff≤2 frac ${(borCdf2*100).toFixed(3)}% < 100%`);
      if (!meanOk)                       caseFails.push(`mean(|atlas-ref|) per channel = [${res.meanAbs.map(m => m.toFixed(3)).join(",")}] (R G B), expected < 0.05`);

      if (caseFails.length > 0) {
        failures.push(
          `addrU=${u} addrV=${v}: ${caseFails.join(" | ")}\n` +
          `    interior hist (${intTotal}px): ${formatHist(res.histInterior, intTotal)}\n` +
          `    border   hist (${borTotal}px): ${formatHist(res.histBorder, borTotal)}\n` +
          `    sample border fails: ${res.firstBorderFails.map(f => `(${f.x},${f.y})=${f.d}`).join(" ")}`,
        );
      }
    }
    if (failures.length > 0) {
      throw new Error("Atlas sampling diverges from native:\n  " + failures.join("\n  "));
    }
    atlasTex.destroy();
    refTex.destroy();
  });

  it("matches native at every mip level (Iliffe pyramid layout)", async () => {
    const device = await requestRealDevice();

    // Build the full mip chain CPU-side.
    const mip0 = makeSource(SRC_W, SRC_H);
    const numMips = defaultMipCount(SRC_W, SRC_H);
    const mips: { data: Uint8Array; w: number; h: number }[] = [
      { data: mip0, w: SRC_W, h: SRC_H },
    ];
    for (let k = 1; k < numMips; k++) {
      const prev = mips[k - 1]!;
      mips.push(downscaleBoxFilter(prev.data, prev.w, prev.h));
    }

    // Atlas: place each mip k at (DST + mipOffsetInPyramid(SRC_W, SRC_H, k))
    // with its own 2-px gutter ring. Fill the atlas with magenta first
    // so any uncovered gutter cells show up obviously in a debug capture.
    const atlas = new Uint8Array(ATLAS_SIZE * ATLAS_SIZE * 4);
    for (let i = 0; i < atlas.length; i += 4) {
      atlas[i + 0] = 255; atlas[i + 1] = 0; atlas[i + 2] = 255; atlas[i + 3] = 255;
    }
    for (let k = 0; k < numMips; k++) {
      const m = mips[k]!;
      const off = mipOffsetInPyramid(SRC_W, SRC_H, k);
      placeWithGutter(atlas, ATLAS_SIZE, ATLAS_SIZE, m.data, m.w, m.h,
                      DST_X + off.x, DST_Y + off.y);
    }

    const atlasTex = device.createTexture({
      size: { width: ATLAS_SIZE, height: ATLAS_SIZE },
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: atlasTex },
      atlas,
      { bytesPerRow: ATLAS_SIZE * 4, rowsPerImage: ATLAS_SIZE },
      { width: ATLAS_SIZE, height: ATLAS_SIZE, depthOrArrayLayers: 1 },
    );

    // Reference: multi-level mipmapped GPUTexture, every level uploaded.
    const refTex = device.createTexture({
      size: { width: SRC_W, height: SRC_H },
      format: "rgba8unorm",
      mipLevelCount: numMips,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    for (let k = 0; k < numMips; k++) {
      const m = mips[k]!;
      device.queue.writeTexture(
        { texture: refTex, mipLevel: k },
        m.data,
        { bytesPerRow: m.w * 4, rowsPerImage: m.h },
        { width: m.w, height: m.h, depthOrArrayLayers: 1 },
      );
    }

    // For each LOD k and each wrap-mode pair, sample with the mip-k
    // sub-rect's atlas-pixel origin/size; the kernel's manual bilinear
    // reads from those coords and compares against textureSampleLevel
    // at LOD k on the reference. Bit-exact agreement expected.
    const cases: [WrapMode, WrapMode][] = [
      ["clamp",  "clamp"],
      ["repeat", "repeat"],
      ["mirror", "mirror"],
      ["repeat", "mirror"],
    ];
    const failures: string[] = [];
    for (let k = 0; k < numMips; k++) {
      const m = mips[k]!;
      const off = mipOffsetInPyramid(SRC_W, SRC_H, k);
      const mipOriginPx = { x: DST_X + off.x + 2, y: DST_Y + off.y + 2 };
      const mipSizePx = { w: m.w, h: m.h };
      for (const [u, v] of cases) {
        const res = await runCompareAt(
          device, atlasTex, refTex, u, v, k,
          mipOriginPx, mipSizePx, SRC_W, SRC_H,
        );
        const intTotal = res.interiorCount;
        const borTotal = res.borderCount;
        const intCdf0 = cdfAtMost(res.histInterior, 0, intTotal);
        const borCdf0 = cdfAtMost(res.histBorder, 0, borTotal);
        const caseFails: string[] = [];
        if (intCdf0 < 1.0) caseFails.push(`interior diff=0 frac ${(intCdf0*100).toFixed(3)}% < 100%`);
        if (borTotal > 0 && borCdf0 < 1.0) caseFails.push(`border diff=0 frac ${(borCdf0*100).toFixed(3)}% < 100%`);
        if (caseFails.length > 0) {
          failures.push(
            `LOD ${k} (mip ${m.w}×${m.h}) addrU=${u} addrV=${v}: ${caseFails.join(" | ")}\n` +
            `    interior hist (${intTotal}px): ${formatHist(res.histInterior, intTotal)}\n` +
            `    border   hist (${borTotal}px): ${formatHist(res.histBorder, borTotal)}`,
          );
        }
      }
    }
    if (failures.length > 0) {
      throw new Error("Atlas sampling diverges from native at mip > 0:\n  " + failures.join("\n  "));
    }
    atlasTex.destroy();
    refTex.destroy();
  });
});
