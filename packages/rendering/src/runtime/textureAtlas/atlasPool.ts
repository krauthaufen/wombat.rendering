// AtlasPool — aval-keyed refcounted sub-rect allocator over shared
// 4096×4096 atlas pages.
//
// Mirrors the UniformPool / IndexPool pattern in heapScene: identity
// keyed on the source `aval<ITexture>`; multiple ROs referencing the
// same texture share one sub-rect; refcount drives free-on-release.
//
// Storage shape: per format, N independent 2D `GPUTexture` "pages",
// each its own 4096² image. The shader binds N consecutive single-
// texture BGL slots per format (linear/srgb) and switches on
// `pageRef` (0..N-1). Adding a new page = allocate a fresh
// GPUTexture and slot it into the next bind-group entry — no GPU-
// side `copyTextureToTexture` of the array, unlike a `texture_2d_array`
// design.
//
// `binding_array<texture_2d<f32>, N>` would be simpler in the shader
// but requires Chrome's experimental `bindless-textures` feature;
// WebGPU 1.0's `GPUBindGroupEntry.resource` doesn't accept arrays of
// views. The N-consecutive-bindings approach is core WebGPU 1.0.
//
// MVP scope (per docs/heap-textures-plan.md):
//   - Tier S/M: source ≤ 1024×1024.
//   - Two formats: rgba8unorm, rgba8unorm-srgb.
//   - No eviction. New pages are allocated lazily up to
//     ATLAS_MAX_PAGES_PER_FORMAT; allocations past that throw.
//
// Mipped textures use the classic Iliffe / id Tech 1.5×1 layout: the
// packer reserves a `1.5W × H` rect, mip 0 occupies the left W×H block
// and mips 1..N stack vertically in the right W/2-wide column. CPU
// mip generation (canvas-2d) for v1; compute-shader path is future.

import { V2f, V2i } from "@aardworx/wombat.base";
import type { aval } from "@aardworx/wombat.adaptive";
import type { ITexture, HostTextureSource } from "../../core/texture.js";
import { TexturePacking } from "./packer.js";

export const ATLAS_PAGE_SIZE = 4096;
/** Tier S/M source-side dimension cap. */
export const ATLAS_MAX_DIM = 1024;
/**
 * Maximum number of independent pages allowed per format. The shader's
 * `switch pageRef` runs an N-way ladder; the BGL declares N consecutive
 * texture slots per format. Allocations beyond this throw.
 */
export const ATLAS_MAX_PAGES_PER_FORMAT = 8;

export type AtlasPageFormat = "rgba8unorm" | "rgba8unorm-srgb";

export const ATLAS_PAGE_FORMATS: readonly AtlasPageFormat[] = ["rgba8unorm", "rgba8unorm-srgb"];

/** Format index used in the drawHeader: 0 = linear, 1 = srgb. */
export const atlasFormatIndex = (f: AtlasPageFormat): number =>
  f === "rgba8unorm-srgb" ? 1 : 0;

export interface AtlasPage {
  readonly format: AtlasPageFormat;
  /**
   * The page's own independent `GPUTexture` (dimension: "2d", single
   * 4096² image). Stable for the page's lifetime — never reallocated.
   */
  readonly texture: GPUTexture;
  /** Immutable packer state — `tryAdd`/`remove` produce new instances. */
  packing: TexturePacking<number>;
  /**
   * Page slot index in the format's binding sequence (0..N-1). Same
   * value flows into the drawHeader `pageRef` field; the shader's
   * `switch pageRef` selects the matching `atlasLinear<i>` /
   * `atlasSrgb<i>` declaration.
   */
  readonly pageId: number;
}

export interface AtlasAcquisition {
  /** Page slot index within the format's binding sequence (0..N-1). */
  readonly pageId: number;
  /** Top-left of mip 0 in the atlas, in normalized [0,1] coords. */
  readonly origin: V2f;
  /** Size of mip 0 in the atlas, in normalized [0,1] coords. */
  readonly size: V2f;
  /** Number of mip levels stored for this acquisition (1 = no pyramid). */
  readonly numMips: number;
  /** Stable refcount handle: pass back to `release`. */
  readonly ref: number;
  /** The page's GPUTexture (so the caller can pin it in the bind group). */
  readonly page: AtlasPage;
}

interface AtlasEntry {
  readonly key: aval<ITexture>;
  readonly format: AtlasPageFormat;
  readonly pageId: number;
  /** Reserved rect in the atlas (covers the full 1.5W×H pyramid for mipped). */
  readonly subRect: { x: number; y: number; w: number; h: number };
  /** Mip-0 size in atlas pixels (== logical source W×H). */
  readonly mip0: { w: number; h: number };
  readonly numMips: number;
  refcount: number;
  /** Stable handle returned to callers. */
  readonly ref: number;
}

/** Source descriptor accepted by `acquire` (small enough not to need a full ITexture). */
export interface AtlasSource {
  readonly width: number;
  readonly height: number;
  /**
   * The host source for upload. The pool calls
   * `device.queue.copyExternalImageToTexture` (external) or
   * `writeTexture` (raw) into the page region the first time the
   * sub-rect is allocated. For mipped uploads the pool downscales
   * via canvas-2d and uploads each level into the pyramid sub-region.
   */
  readonly host?: HostTextureSource;
}

export interface AtlasAcquireOptions {
  readonly source?: AtlasSource;
  /** When true, reserve a 1.5W×H rect and store an embedded mip pyramid. */
  readonly wantsMips?: boolean;
  /**
   * Number of mip levels to store. Defaults to
   * `floor(log2(max(w,h))) + 1` (full chain to 1×1).
   */
  readonly numMips?: number;
}

/**
 * Default mip count for a `w×h` source: full chain down to 1×1.
 */
export const defaultMipCount = (w: number, h: number): number =>
  Math.floor(Math.log2(Math.max(w, h))) + 1;

/**
 * Mip-k pixel size given mip-0 size `(w,h)`. Halves each level and
 * floors at 1×1.
 */
export const mipPixelSize = (w: number, h: number, k: number): { w: number; h: number } => ({
  w: Math.max(1, w >> k),
  h: Math.max(1, h >> k),
});

/**
 * Mip-k offset within the embedded 1.5×1 pyramid relative to the
 * pyramid's top-left. Mip 0 is at (0,0). Mips 1..N stack vertically
 * in the right column starting at x=W with cumulative y offsets:
 *   y_k = sum_{j=1..k-1} (H >> j).
 * Caller uses max(1, ...) clamping to track 1px-floor mips.
 */
export const mipOffsetInPyramid = (
  w: number,
  h: number,
  k: number,
): { x: number; y: number } => {
  if (k === 0) return { x: 0, y: 0 };
  let y = 0;
  for (let j = 1; j < k; j++) y += Math.max(1, h >> j);
  return { x: w, y };
};

/**
 * Pull `(format, width, height, mipLevelCount, host?)` out of an
 * ITexture for atlas eligibility. Mirrors `heapAdapter.describeTexture`
 * — kept local so `repack` can validate without importing the adapter.
 * Returns `null` for sources we can't measure.
 */
function describeAtlasTexture(t: ITexture): {
  format: GPUTextureFormat; width: number; height: number;
  mipLevelCount: number; host?: HostTextureSource;
} | null {
  if (t.kind === "gpu") {
    const tex = t.texture;
    return {
      format: tex.format, width: tex.width, height: tex.height,
      mipLevelCount: tex.mipLevelCount,
    };
  }
  const src = t.source;
  if (src.kind === "raw") {
    return {
      format: src.format, width: src.width, height: src.height,
      mipLevelCount: src.mipLevelCount ?? 1, host: src,
    };
  }
  const ext = src.source as unknown;
  let w = 0, h = 0;
  if (typeof HTMLVideoElement !== "undefined" && ext instanceof HTMLVideoElement) {
    w = ext.videoWidth; h = ext.videoHeight;
  } else if (typeof ImageData !== "undefined" && ext instanceof ImageData) {
    w = ext.width; h = ext.height;
  } else {
    const any = ext as { width?: number; height?: number };
    w = any.width ?? 0; h = any.height ?? 0;
  }
  if (w <= 0 || h <= 0) return null;
  return {
    format: src.format ?? "rgba8unorm",
    width: w, height: h,
    mipLevelCount: src.generateMips ? Math.floor(Math.log2(Math.max(w, h))) + 1 : 1,
    host: src,
  };
}

/**
 * Aval-identity keyed pool over the per-format page sets. Multiple
 * acquisitions of the same `aval<ITexture>` share one sub-rect; the
 * refcount drives release.
 */
export class AtlasPool {
  private readonly pagesByFormat = new Map<AtlasPageFormat, AtlasPage[]>();
  private readonly pageAddedListeners = new Map<AtlasPageFormat, Set<(pageId: number) => void>>();
  private readonly entriesByAval = new Map<aval<ITexture>, AtlasEntry>();
  private readonly entriesByRef = new Map<number, AtlasEntry>();
  private nextRef = 1;

  constructor(private readonly device: GPUDevice) {
    for (const f of ATLAS_PAGE_FORMATS) {
      this.pagesByFormat.set(f, []);
      this.pageAddedListeners.set(f, new Set());
    }
  }

  /** All currently allocated pages for a given format. */
  pagesFor(format: AtlasPageFormat): readonly AtlasPage[] {
    return this.pagesByFormat.get(format) ?? [];
  }

  /**
   * Subscribe to page-allocation events. Fires after a fresh
   * `AtlasPage` is appended to the format's page list (its `pageId`
   * is the slot the bind group should now wire up).
   */
  onPageAdded(format: AtlasPageFormat, cb: (pageId: number) => void): { dispose(): void } {
    const set = this.pageAddedListeners.get(format)!;
    set.add(cb);
    return { dispose: () => { set.delete(cb); } };
  }

  private allocatePage(format: AtlasPageFormat, pageId: number): AtlasPage {
    const texture = this.device.createTexture({
      label: `atlas/${format}/${pageId}`,
      size: { width: ATLAS_PAGE_SIZE, height: ATLAS_PAGE_SIZE, depthOrArrayLayers: 1 },
      format,
      mipLevelCount: 1,
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    return {
      format,
      texture,
      packing: TexturePacking.empty<number>(new V2i(ATLAS_PAGE_SIZE, ATLAS_PAGE_SIZE), false),
      pageId,
    };
  }

  /**
   * Tier-S eligibility: dimension threshold + supported format. If
   * this returns null, the caller should route the texture down the
   * Tier L (standalone) path.
   */
  static eligibleFormat(format: GPUTextureFormat): AtlasPageFormat | null {
    if (format === "rgba8unorm") return "rgba8unorm";
    if (format === "rgba8unorm-srgb") return "rgba8unorm-srgb";
    return null;
  }

  static eligibleSize(width: number, height: number): boolean {
    return width > 0 && height > 0 && width <= ATLAS_MAX_DIM && height <= ATLAS_MAX_DIM;
  }

  /**
   * Acquire (or share) a sub-rect. The returned `AtlasAcquisition`
   * gives the page slot index plus the mip-0 `origin`/`size` (in
   * normalized atlas coords) and `numMips`. If the texture is already
   * known (same aval), refcount bumps and we return the existing
   * placement.
   */
  acquire(
    format: AtlasPageFormat,
    sourceAval: aval<ITexture>,
    width: number,
    height: number,
    opts: AtlasAcquireOptions = {},
  ): AtlasAcquisition {
    const existing = this.entriesByAval.get(sourceAval);
    if (existing !== undefined) {
      existing.refcount++;
      return this.makeResult(existing);
    }

    const wantsMips = opts.wantsMips === true;
    const numMips = wantsMips
      ? Math.max(1, Math.min(opts.numMips ?? defaultMipCount(width, height),
                             defaultMipCount(width, height)))
      : 1;

    // For mipped textures, reserve a 1.5W × H rect; otherwise W × H.
    const reservedW = wantsMips ? Math.ceil(width * 1.5) : width;
    const reservedH = height;
    const size = new V2i(reservedW, reservedH);

    const pages = this.pagesByFormat.get(format)!;

    // Try fitting into existing pages first.
    for (let i = 0; i < pages.length; i++) {
      const page = pages[i]!;
      const next = page.packing.tryAdd(this.nextRef, size);
      if (next !== null) {
        page.packing = next;
        return this.finalize(sourceAval, format, i, page, opts.source, width, height, numMips);
      }
    }

    // Allocate a fresh page if we haven't hit the max.
    if (pages.length >= ATLAS_MAX_PAGES_PER_FORMAT) {
      throw new Error(
        `atlas: ATLAS_MAX_PAGES_PER_FORMAT (${ATLAS_MAX_PAGES_PER_FORMAT}) exceeded for format ${format}`,
      );
    }
    const pageId = pages.length;
    const page = this.allocatePage(format, pageId);
    const placed = page.packing.tryAdd(this.nextRef, size);
    if (placed === null) {
      throw new Error(
        `AtlasPool: ${reservedW}×${reservedH} doesn't fit a fresh ${ATLAS_PAGE_SIZE}² page`,
      );
    }
    page.packing = placed;
    pages.push(page);
    for (const cb of this.pageAddedListeners.get(format)!) cb(pageId);
    return this.finalize(sourceAval, format, pageId, page, opts.source, width, height, numMips);
  }

  /**
   * Look up the existing acquisition for a source aval, or `undefined`
   * if the aval is not currently held in the pool. Mirrors
   * `UniformPool.entry` — used by the heap path's reactivity loop to
   * detect "this aval has live refs and just marked".
   */
  entry(sourceAval: aval<ITexture>): AtlasAcquisition | undefined {
    const e = this.entriesByAval.get(sourceAval);
    return e === undefined ? undefined : this.makeResult(e);
  }

  /**
   * Re-pack an aval's atlas placement against a new ITexture value.
   * Mirrors `UniformPool.repack(av, newValue)` — used when an atlas-
   * routed `aval<ITexture>` source swaps. Frees the old sub-rect (and
   * the page slot in the packer) and acquires a fresh one for the new
   * texture; the returned `AtlasAcquisition` carries the (potentially
   * different) `pageId`/`origin`/`size`/`numMips`/`ref`. Refcount is
   * preserved across the swap so multi-RO sharing keeps working.
   *
   * Throws if the new texture isn't Tier-S eligible (different format
   * outside rgba8unorm/srgb, or larger than the cap). Callers should
   * rely on the heap eligibility classifier, but this is a safety net.
   */
  repack(
    sourceAval: aval<ITexture>,
    newTexture: ITexture,
    opts: AtlasAcquireOptions = {},
  ): AtlasAcquisition {
    const old = this.entriesByAval.get(sourceAval);
    if (old === undefined) {
      throw new Error("AtlasPool.repack: source aval has no live entry");
    }
    // Resolve the new texture's eligibility + dimensions.
    const desc = describeAtlasTexture(newTexture);
    if (desc === null) {
      throw new Error("AtlasPool.repack: new ITexture has no measurable dimensions");
    }
    const fmt = AtlasPool.eligibleFormat(desc.format);
    if (fmt === null) {
      throw new Error(
        `AtlasPool.repack: new texture format ${desc.format} is not Tier-S eligible`,
      );
    }
    if (!AtlasPool.eligibleSize(desc.width, desc.height)) {
      throw new Error(
        `AtlasPool.repack: new texture ${desc.width}×${desc.height} exceeds Tier-S cap`,
      );
    }

    // 1. Free the old rect from its page's packer + drop the maps.
    const savedRefcount = old.refcount;
    const oldPages = this.pagesByFormat.get(old.format)!;
    const oldPage = oldPages[old.pageId];
    if (oldPage !== undefined) {
      oldPage.packing = oldPage.packing.remove(old.ref);
    }
    this.entriesByRef.delete(old.ref);
    this.entriesByAval.delete(sourceAval);

    // 2. Acquire a fresh placement for the new texture. Reuse the
    //    `acquire` algorithm — entriesByAval no longer has us, so it
    //    picks the existing-pages-first path naturally.
    const acqOpts: AtlasAcquireOptions = {
      ...opts,
      ...(opts.source === undefined && desc.host !== undefined
        ? { source: { width: desc.width, height: desc.height, host: desc.host } }
        : {}),
    };
    const acq = this.acquire(fmt, sourceAval, desc.width, desc.height, acqOpts);

    // 3. Restore the original refcount (acquire set it to 1).
    const newEntry = this.entriesByRef.get(acq.ref);
    if (newEntry !== undefined) newEntry.refcount = savedRefcount;
    return acq;
  }

  /** Decrement the refcount; on zero, free the sub-rect. */
  release(ref: number): void {
    const e = this.entriesByRef.get(ref);
    if (e === undefined) return;
    e.refcount--;
    if (e.refcount > 0) return;
    const pages = this.pagesByFormat.get(e.format)!;
    const page = pages[e.pageId];
    if (page !== undefined) {
      page.packing = page.packing.remove(e.ref);
    }
    this.entriesByRef.delete(e.ref);
    this.entriesByAval.delete(e.key);
  }

  /** Destroy every page texture. The pool becomes unusable. */
  dispose(): void {
    for (const [, pages] of this.pagesByFormat) {
      for (const p of pages) p.texture.destroy();
    }
    this.pagesByFormat.clear();
    this.pageAddedListeners.clear();
    this.entriesByAval.clear();
    this.entriesByRef.clear();
  }

  private finalize(
    aval: aval<ITexture>,
    format: AtlasPageFormat,
    pageId: number,
    page: AtlasPage,
    source: AtlasSource | undefined,
    mip0W: number,
    mip0H: number,
    numMips: number,
  ): AtlasAcquisition {
    const ref = this.nextRef++;
    const placed = page.packing.used.get(ref);
    if (placed === undefined) {
      throw new Error("AtlasPool: invariant — placed sub-rect missing from packer");
    }
    const x = placed.min.x;
    const y = placed.min.y;
    const w = placed.max.x - placed.min.x + 1;
    const h = placed.max.y - placed.min.y + 1;
    const entry: AtlasEntry = {
      key: aval,
      format,
      pageId,
      subRect: { x, y, w, h },
      mip0: { w: mip0W, h: mip0H },
      numMips,
      refcount: 1,
      ref,
    };
    this.entriesByAval.set(aval, entry);
    this.entriesByRef.set(ref, entry);
    if (source !== undefined && source.host !== undefined) {
      this.upload(page, x, y, source.host, mip0W, mip0H, numMips);
    }
    return this.makeResult(entry);
  }

  private upload(
    page: AtlasPage,
    x: number,
    y: number,
    host: HostTextureSource,
    w: number,
    h: number,
    numMips: number,
  ): void {
    // Mip 0 always lands at (x, y) on the page with size w×h.
    this.uploadLevel(page, x, y, host, w, h);
    if (numMips <= 1) return;

    // Mip k≥1: downscale via canvas-2d, upload at the pyramid offset.
    const src = this.toDrawable(host, w, h);
    if (src === null) {
      // Headless / unsupported source — skip mip generation; level 0
      // is still in place. Callers that need mips should provide a
      // host source the pool can render-2d.
      return;
    }
    for (let k = 1; k < numMips; k++) {
      const off = mipOffsetInPyramid(w, h, k);
      const mw = Math.max(1, w >> k);
      const mh = Math.max(1, h >> k);
      const mip = this.makeMipCanvas(src, mw, mh);
      if (mip === null) continue;
      this.device.queue.copyExternalImageToTexture(
        { source: mip as GPUImageCopyExternalImageSource },
        { texture: page.texture, origin: { x: x + off.x, y: y + off.y, z: 0 } },
        { width: mw, height: mh, depthOrArrayLayers: 1 },
      );
    }
  }

  private uploadLevel(
    page: AtlasPage,
    x: number,
    y: number,
    host: HostTextureSource,
    w: number,
    h: number,
  ): void {
    if (host.kind === "external") {
      this.device.queue.copyExternalImageToTexture(
        { source: host.source as GPUImageCopyExternalImageSource },
        { texture: page.texture, origin: { x, y, z: 0 } },
        { width: w, height: h, depthOrArrayLayers: 1 },
      );
      return;
    }
    const data = host.data instanceof ArrayBuffer
      ? new Uint8Array(host.data)
      : new Uint8Array(host.data.buffer, host.data.byteOffset, host.data.byteLength);
    this.device.queue.writeTexture(
      { texture: page.texture, origin: { x, y, z: 0 } },
      data as unknown as GPUAllowSharedBufferSource,
      { bytesPerRow: w * 4, rowsPerImage: h },
      { width: w, height: h, depthOrArrayLayers: 1 },
    );
  }

  /**
   * Produce a `CanvasImageSource` we can feed to `drawImage` for
   * downscaling. Returns null when the runtime can't synthesise one
   * (no DOM / OffscreenCanvas / ImageBitmap support).
   */
  private toDrawable(
    host: HostTextureSource,
    w: number,
    h: number,
  ): CanvasImageSource | null {
    if (host.kind === "external") {
      const s = host.source as unknown;
      if (typeof ImageData !== "undefined" && s instanceof ImageData) {
        return this.imageDataToCanvas(s);
      }
      return s as CanvasImageSource;
    }
    if (typeof ImageData === "undefined") return null;
    const bytes = host.data instanceof ArrayBuffer
      ? new Uint8Array(host.data)
      : new Uint8Array(host.data.buffer, host.data.byteOffset, host.data.byteLength);
    const out = new Uint8ClampedArray(bytes.length);
    out.set(bytes);
    const img = new ImageData(out, w, h);
    return this.imageDataToCanvas(img);
  }

  private imageDataToCanvas(img: ImageData): CanvasImageSource | null {
    const c = this.makeCanvas(img.width, img.height);
    if (c === null) return null;
    const ctx = c.getContext("2d");
    if (ctx === null) return null;
    (ctx as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D).putImageData(img, 0, 0);
    return c as unknown as CanvasImageSource;
  }

  private makeMipCanvas(
    src: CanvasImageSource,
    w: number,
    h: number,
  ): CanvasImageSource | null {
    const c = this.makeCanvas(w, h);
    if (c === null) return null;
    const ctx = c.getContext("2d") as
      | CanvasRenderingContext2D
      | OffscreenCanvasRenderingContext2D
      | null;
    if (ctx === null) return null;
    ctx.imageSmoothingEnabled = true;
    (ctx as CanvasRenderingContext2D).imageSmoothingQuality = "high";
    ctx.drawImage(src, 0, 0, w, h);
    return c as unknown as CanvasImageSource;
  }

  private makeCanvas(w: number, h: number): HTMLCanvasElement | OffscreenCanvas | null {
    if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
    if (typeof document !== "undefined") {
      const c = document.createElement("canvas");
      c.width = w;
      c.height = h;
      return c;
    }
    return null;
  }

  private makeResult(e: AtlasEntry): AtlasAcquisition {
    const page = this.pagesByFormat.get(e.format)![e.pageId]!;
    const inv = 1.0 / ATLAS_PAGE_SIZE;
    return {
      pageId: e.pageId,
      origin: new V2f(e.subRect.x * inv, e.subRect.y * inv),
      size: new V2f(e.mip0.w * inv, e.mip0.h * inv),
      numMips: e.numMips,
      ref: e.ref,
      page,
    };
  }
}
