// Compute kernel that builds an atlas sub-rect's mip pyramid AND fills
// both gutter rings entirely on GPU, using only core WebGPU 1.0
// features (no `readonly-and-readwrite-storage-textures`).
//
// Architecture
// ============
// The whole operation runs inside a per-acquire **scratch buffer**
// laid out exactly like the destination region of the atlas page
// (bounding rect of the Iliffe pyramid + gutter, in atlas-pixel
// coordinates relative to the sub-rect origin). Each pixel is one
// packed `u32` (rgba8). Rows are padded to 256-byte alignment so the
// final `copyBufferToTexture` is straight-through.
//
// Per acquire:
//   1. CPU writes the raw source pixels into the scratch buffer at
//      the mip-0 interior offset (via `mappedAtCreation`). One JS
//      loop, no upload validation pitfalls.
//   2. For each mip k > 0: dispatch the `interior` kernel, which
//      reads 2×2 from mip-(k-1) in the buffer and writes the average
//      to mip-k interior, all in-buffer. Each mip lives in its own
//      compute pass — the pass boundary acts as a barrier so
//      mip-(k-1) is fully written before mip-k reads from it.
//   3. For each mip k: dispatch the `gutter` kernel, which copies
//      the appropriate clamp / wrap interior cell into each gutter
//      cell. Also one compute pass per mip.
//   4. `copyBufferToTexture` writes the full scratch buffer to the
//      atlas page at the sub-rect origin.
//
// The kernel reads and writes ONE storage buffer through a
// `read_write` binding. WebGPU validates a single binding (even with
// read_write access) as one usage entry per resource — no hazard.
//
// No device features required. No same-texture binding pitfalls. The
// scratch buffer is per-acquire and freed after submission completes.

const WGSL = /* wgsl */`
struct Params {
  buf_stride_u32:    u32,
  _pad0:             u32,
  src_origin_in_buf: vec2<u32>,
  src_size:          vec2<u32>,
  dst_origin_in_buf: vec2<u32>,
  dst_size:          vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> P: Params;
// Optional mip-0 source — bound only by the interiorFromTexture
// entry point (the staging-texture upload path). The buffer-only
// entry points (interior, gutter) don't reference it, so their
// pipeline layouts omit it.
@group(0) @binding(2) var srcTex: texture_2d<f32>;

fn loadRgba(x: u32, y: u32) -> vec4<f32> {
  let idx = y * P.buf_stride_u32 + x;
  return unpack4x8unorm(buf[idx]);
}

fn storeRgba(x: u32, y: u32, v: vec4<f32>) {
  let idx = y * P.buf_stride_u32 + x;
  buf[idx] = pack4x8unorm(v);
}

// Copy the staging texture's pixels into the buffer at mip-0's
// interior offset (dst_origin_in_buf). src_size = staging size.
// Used by the external-source upload path so the source decode
// happens GPU-side (copyExternalImageToTexture -> staging) instead
// of via a main-thread canvas-2d getImageData round-trip.
@compute @workgroup_size(8, 8, 1)
fn interiorFromTexture(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.src_size.x || gid.y >= P.src_size.y) { return; }
  let v = textureLoad(srcTex, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
  storeRgba(P.dst_origin_in_buf.x + gid.x, P.dst_origin_in_buf.y + gid.y, v);
}

@compute @workgroup_size(8, 8, 1)
fn interior(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.dst_size.x || gid.y >= P.dst_size.y) { return; }
  let sx0 = P.src_origin_in_buf.x + gid.x * 2u;
  let sy0 = P.src_origin_in_buf.y + gid.y * 2u;
  let sxMax = P.src_origin_in_buf.x + P.src_size.x - 1u;
  let syMax = P.src_origin_in_buf.y + P.src_size.y - 1u;
  let sx1 = min(sx0 + 1u, sxMax);
  let sy1 = min(sy0 + 1u, syMax);
  let a = loadRgba(sx0, sy0);
  let b = loadRgba(sx1, sy0);
  let c = loadRgba(sx0, sy1);
  let d = loadRgba(sx1, sy1);
  let avg = (a + b + c + d) * 0.25;
  storeRgba(P.dst_origin_in_buf.x + gid.x, P.dst_origin_in_buf.y + gid.y, avg);
}

@compute @workgroup_size(8, 8, 1)
fn gutter(@builtin(global_invocation_id) gid: vec3<u32>) {
  let totW = P.dst_size.x + 4u;
  let totH = P.dst_size.y + 4u;
  if (gid.x >= totW || gid.y >= totH) { return; }
  let dx = i32(gid.x) - 2;
  let dy = i32(gid.y) - 2;
  let dw = i32(P.dst_size.x);
  let dh = i32(P.dst_size.y);
  if (dx >= 0 && dx < dw && dy >= 0 && dy < dh) { return; }
  var sx: i32;
  var sy: i32;
       if (dx == -2)     { sx = dw - 1; }
  else if (dx == -1)     { sx = 0; }
  else if (dx == dw)     { sx = dw - 1; }
  else if (dx == dw + 1) { sx = 0; }
  else                   { sx = dx; }
       if (dy == -2)     { sy = dh - 1; }
  else if (dy == -1)     { sy = 0; }
  else if (dy == dh)     { sy = dh - 1; }
  else if (dy == dh + 1) { sy = 0; }
  else                   { sy = dy; }
  let v = loadRgba(u32(i32(P.dst_origin_in_buf.x) + sx),
                   u32(i32(P.dst_origin_in_buf.y) + sy));
  storeRgba(u32(i32(P.dst_origin_in_buf.x) + dx),
            u32(i32(P.dst_origin_in_buf.y) + dy),
            v);
}
`;

interface KernelCache {
  /** bindings 0 (storage buf) + 1 (uniform params) — buffer-only passes. */
  bufBindGroupLayout: GPUBindGroupLayout;
  /** bindings 0 + 1 + 2 (texture_2d src) — the mip-0-from-texture pass. */
  texBindGroupLayout: GPUBindGroupLayout;
  interiorPipeline:           GPUComputePipeline;
  gutterPipeline:             GPUComputePipeline;
  interiorFromTexturePipeline: GPUComputePipeline;
}

const caches = new WeakMap<GPUDevice, KernelCache>();

function getKernel(device: GPUDevice): KernelCache {
  const cached = caches.get(device);
  if (cached !== undefined) return cached;
  const module = device.createShaderModule({ code: WGSL, label: "atlas/mipGutterKernel" });
  const bufBindGroupLayout = device.createBindGroupLayout({
    label: "atlas/mipGutterKernel/bgl.buf",
    entries: [
      { binding: 0, visibility: 0x4 /* COMPUTE */, buffer: { type: "storage" } },
      { binding: 1, visibility: 0x4 /* COMPUTE */, buffer: { type: "uniform" } },
    ],
  });
  const texBindGroupLayout = device.createBindGroupLayout({
    label: "atlas/mipGutterKernel/bgl.tex",
    entries: [
      { binding: 0, visibility: 0x4 /* COMPUTE */, buffer: { type: "storage" } },
      { binding: 1, visibility: 0x4 /* COMPUTE */, buffer: { type: "uniform" } },
      { binding: 2, visibility: 0x4 /* COMPUTE */, texture: { sampleType: "float", viewDimension: "2d" } },
    ],
  });
  const bufLayout = device.createPipelineLayout({ bindGroupLayouts: [bufBindGroupLayout] });
  const texLayout = device.createPipelineLayout({ bindGroupLayouts: [texBindGroupLayout] });
  const interiorPipeline = device.createComputePipeline({
    layout: bufLayout, compute: { module, entryPoint: "interior" },
    label: "atlas/mipGutterKernel/interior",
  });
  const gutterPipeline = device.createComputePipeline({
    layout: bufLayout, compute: { module, entryPoint: "gutter" },
    label: "atlas/mipGutterKernel/gutter",
  });
  const interiorFromTexturePipeline = device.createComputePipeline({
    layout: texLayout, compute: { module, entryPoint: "interiorFromTexture" },
    label: "atlas/mipGutterKernel/interiorFromTexture",
  });
  const entry: KernelCache = {
    bufBindGroupLayout, texBindGroupLayout,
    interiorPipeline, gutterPipeline, interiorFromTexturePipeline,
  };
  caches.set(device, entry);
  return entry;
}

/**
 * Per-mip slot description. `origin`/`size` are in *bounding-rect-
 * relative pixel coordinates* (not atlas-page coordinates) — the
 * scratch buffer is laid out as the bounding rect with row stride
 * `bufStrideU32` u32s per row. Mip 0's interior origin matches
 * `acq.origin` − sub-rect origin.
 */
export interface MipSlot {
  readonly origin: { x: number; y: number };
  readonly size:   { w: number; h: number };
}

/** How mip-0's interior pixels get into the scratch buffer. */
type Mip0Source =
  // The caller already filled the buffer's mip-0 region at creation
  // (via `mappedAtCreation`) — no extra pass needed.
  | { readonly kind: "prefilled" }
  // Add an `interiorFromTexture` compute pass that reads the staging
  // texture and writes mip-0's interior into the buffer.
  | { readonly kind: "texture"; readonly tex: GPUTexture };

/**
 * Shared orchestration: allocate a scratch buffer laid out as the
 * sub-rect's bounding box, initialise mip-0 (per `mip0Src`), run the
 * per-mip interior-downscale + gutter compute passes over the buffer,
 * then `copyBufferToTexture` into `page` at (subRectX, subRectY). The
 * scratch buffer + per-pass UBOs are freed once the submission
 * completes. Caller owns `mip0Src.tex` (if any) and its lifetime.
 *
 * `makeScratch` lets the `prefilled` path create the buffer with
 * `mappedAtCreation` and fill it before we run; the `texture` path
 * passes a plain (already-created) buffer.
 */
function runMipGutter(
  device: GPUDevice,
  page: GPUTexture,
  subRectX: number, subRectY: number,
  boundsW: number, boundsH: number,
  rowBytes: number,
  scratch: GPUBuffer,
  srcW: number, srcH: number,
  mips: readonly MipSlot[],
  mip0Src: Mip0Source,
): void {
  const k = getKernel(device);
  const bufStrideU32 = rowBytes / 4;
  const mip0 = mips[0]!;

  const enc = device.createCommandEncoder({ label: "atlas/mipGutterKernel" });

  const ubos: GPUBuffer[] = [];
  const makeUbo = (
    srcO: { x: number; y: number }, srcS: { w: number; h: number },
    dstO: { x: number; y: number }, dstS: { w: number; h: number },
  ): GPUBuffer => {
    const buf = device.createBuffer({
      size: 48, // 12 u32, padded
      usage: 0x40 /* UNIFORM */ | 0x08 /* COPY_DST */,
      label: "atlas/mipGutter/params",
    });
    const u = new Uint32Array(12);
    u[0] = bufStrideU32;
    u[1] = 0;
    u[2] = srcO.x; u[3] = srcO.y;
    u[4] = srcS.w; u[5] = srcS.h;
    u[6] = dstO.x; u[7] = dstO.y;
    u[8] = dstS.w; u[9] = dstS.h;
    device.queue.writeBuffer(buf, 0, u);
    ubos.push(buf);
    return buf;
  };
  const makeBufBg = (ubo: GPUBuffer): GPUBindGroup =>
    device.createBindGroup({
      layout: k.bufBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: scratch } },
        { binding: 1, resource: { buffer: ubo } },
      ],
    });

  // Mip-0 init pass — only the staging-texture path needs one. Reads
  // `srcTex[gid]` for gid in [0,srcW)×[0,srcH), writes into the buffer
  // at mip0.origin. Its own pass so the write is visible to the
  // downstream interior/gutter passes (the pass boundary is a barrier).
  if (mip0Src.kind === "texture") {
    const ubo = makeUbo(mip0.origin, { w: srcW, h: srcH }, mip0.origin, { w: srcW, h: srcH });
    const bg = device.createBindGroup({
      layout: k.texBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: scratch } },
        { binding: 1, resource: { buffer: ubo } },
        { binding: 2, resource: mip0Src.tex.createView() },
      ],
    });
    const pass = enc.beginComputePass({ label: "atlas/mipGutter/interiorFromTexture" });
    pass.setPipeline(k.interiorFromTexturePipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(srcW / 8), Math.ceil(srcH / 8), 1);
    pass.end();
  }

  // Interior passes: one compute pass per mip > 0 (pass boundary
  // gives us the barrier ensuring mip-(k-1) is fully written before
  // mip-k reads it).
  for (let m = 1; m < mips.length; m++) {
    const src = mips[m - 1]!;
    const dst = mips[m]!;
    const ubo = makeUbo(src.origin, src.size, dst.origin, dst.size);
    const pass = enc.beginComputePass({ label: `atlas/mipGutter/interior/${m}` });
    pass.setPipeline(k.interiorPipeline);
    pass.setBindGroup(0, makeBufBg(ubo));
    pass.dispatchWorkgroups(Math.ceil(dst.size.w / 8), Math.ceil(dst.size.h / 8), 1);
    pass.end();
  }

  // Gutter passes: one per mip (each pass reads mip-m interior, which
  // was either filled at buffer creation / by the texture pass for
  // m=0, or by the interior pass for m>0).
  for (let m = 0; m < mips.length; m++) {
    const dst = mips[m]!;
    const ubo = makeUbo(dst.origin, dst.size, dst.origin, dst.size);
    const pass = enc.beginComputePass({ label: `atlas/mipGutter/gutter/${m}` });
    pass.setPipeline(k.gutterPipeline);
    pass.setBindGroup(0, makeBufBg(ubo));
    pass.dispatchWorkgroups(
      Math.ceil((dst.size.w + 4) / 8),
      Math.ceil((dst.size.h + 4) / 8),
      1,
    );
    pass.end();
  }

  // Final upload: buffer → page.
  enc.copyBufferToTexture(
    { buffer: scratch, bytesPerRow: rowBytes, rowsPerImage: boundsH },
    { texture: page, origin: { x: subRectX, y: subRectY, z: 0 } },
    { width: boundsW, height: boundsH, depthOrArrayLayers: 1 },
  );

  device.queue.submit([enc.finish()]);

  // Lifetime: destroy scratch + per-pass UBOs once submitted work is
  // done. The submit holds a ref until completion.
  void device.queue.onSubmittedWorkDone().then(() => {
    scratch.destroy();
    for (const b of ubos) b.destroy();
  });
}

/** Scratch-buffer geometry for a sub-rect's bounding box. Rows are
 *  padded to 256 bytes for the final `copyBufferToTexture`. */
function scratchGeom(boundsW: number, boundsH: number): { rowBytes: number; bufSize: number } {
  const rowBytes = Math.max(256, Math.ceil(boundsW * 4 / 256) * 256);
  return { rowBytes, bufSize: rowBytes * boundsH };
}

/**
 * Build the mip pyramid + gutters on GPU from a **CPU pixel array**
 * and upload the result to a sub-rect of `page`. `srcPixels` is the
 * raw RGBA8 source for mip 0 (`srcW × srcH` pixels). The scratch
 * buffer's mip-0 region is filled at creation (`mappedAtCreation`),
 * then the compute passes downscale the remaining mips and fill every
 * mip's 2-px gutter ring. Used for `raw` host sources (where the
 * pixels are already in memory — no extraction cost) and as the
 * fallback for `external` sources when the GPU staging path is
 * unavailable. Caller must ensure `page` has COPY_DST usage.
 */
export function buildMipsAndGutterOnGpu(
  device: GPUDevice,
  page: GPUTexture,
  subRectX: number, subRectY: number,
  boundsW: number, boundsH: number,
  srcPixels: Uint8Array,
  srcW: number, srcH: number,
  mips: readonly MipSlot[],
): void {
  if (mips.length === 0) return;
  const { rowBytes, bufSize } = scratchGeom(boundsW, boundsH);
  const bufStrideU32 = rowBytes / 4;

  const scratch = device.createBuffer({
    label: `atlas/mipGutter/scratch(${boundsW}x${boundsH})`,
    size: bufSize,
    usage: 0x80 /* STORAGE */ | 0x04 /* COPY_SRC */ | 0x08 /* COPY_DST */,
    mappedAtCreation: true,
  });

  // CPU init: write mip-0 source pixels at the mip-0 interior offset
  // inside the buffer. mips[0].origin is the interior offset.
  const mip0 = mips[0]!;
  {
    const mapped = new Uint32Array(scratch.getMappedRange());
    for (let y = 0; y < srcH; y++) {
      for (let x = 0; x < srcW; x++) {
        const si = (y * srcW + x) * 4;
        const r = srcPixels[si]!;
        const g = srcPixels[si + 1]!;
        const b = srcPixels[si + 2]!;
        const a = srcPixels[si + 3]!;
        // little-endian rgba8: byte0=r, byte1=g, byte2=b, byte3=a.
        const packed = (a << 24) | (b << 16) | (g << 8) | r;
        mapped[(mip0.origin.y + y) * bufStrideU32 + (mip0.origin.x + x)] = packed >>> 0;
      }
    }
  }
  scratch.unmap();

  runMipGutter(
    device, page, subRectX, subRectY, boundsW, boundsH,
    rowBytes, scratch, srcW, srcH, mips, { kind: "prefilled" },
  );
}

/**
 * Build the mip pyramid + gutters on GPU from a **staging texture**
 * and upload the result to a sub-rect of `page`. `srcTex` holds the
 * decoded mip-0 pixels (`srcW × srcH`) — typically just filled by
 * `copyExternalImageToTexture`, so no main-thread pixel readback is
 * involved. An `interiorFromTexture` compute pass copies `srcTex` into
 * the scratch buffer; the rest is identical to `buildMipsAndGutterOnGpu`
 * (downscale every further mip, fill every mip's 2-px gutter ring,
 * single `copyBufferToTexture` into the page).
 *
 * Caller owns `srcTex` and is responsible for destroying it after the
 * submission completes (`device.queue.onSubmittedWorkDone()`); this
 * function only reads it. `srcTex` must have TEXTURE_BINDING usage.
 */
export function buildMipsAndGutterFromTexture(
  device: GPUDevice,
  page: GPUTexture,
  subRectX: number, subRectY: number,
  boundsW: number, boundsH: number,
  srcTex: GPUTexture,
  srcW: number, srcH: number,
  mips: readonly MipSlot[],
): void {
  if (mips.length === 0) return;
  const { rowBytes, bufSize } = scratchGeom(boundsW, boundsH);
  const scratch = device.createBuffer({
    label: `atlas/mipGutter/scratch(${boundsW}x${boundsH})`,
    size: bufSize,
    usage: 0x80 /* STORAGE */ | 0x04 /* COPY_SRC */ | 0x08 /* COPY_DST */,
  });
  runMipGutter(
    device, page, subRectX, subRectY, boundsW, boundsH,
    rowBytes, scratch, srcW, srcH, mips, { kind: "texture", tex: srcTex },
  );
}
