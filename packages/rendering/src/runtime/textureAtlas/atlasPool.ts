// AtlasPool — aval-keyed refcounted sub-rect allocator over shared
// 4096×4096 atlas pages.
//
// Mirrors the UniformPool / IndexPool pattern in heapScene: identity
// keyed on the source `aval<ITexture>`; multiple ROs referencing the
// same texture share one sub-rect; refcount drives free-on-release.
//
// Storage shape: per format, ONE GPUTexture with N layers
// (`texture_2d_array<f32>` view from the shader's perspective).
// "Pages" = layers in that array. When the format runs out of
// layers, the texture is reallocated with `2*N` layers and the live
// layers are GPU-side copied via `copyTextureToTexture`. Listeners
// (heapScene's bind groups) get notified via `onResize` and rebuild.
//
// `binding_array<texture_2d<f32>, N>` would be simpler but requires
// Chrome's experimental `bindless-textures` feature; WebGPU 1.0's
// `GPUBindGroupEntry.resource` doesn't accept arrays of views.
// `texture_2d_array` is native WebGPU 1.0 — works on every browser.
//
// MVP scope (per docs/heap-textures-plan.md):
//   - Tier S/M: source ≤ 1024×1024.
//   - Two formats: rgba8unorm, rgba8unorm-srgb. Each gets its own
//     format-store (separate texture_2d_array in the bind group).
//   - No eviction. Layers grow pow2 + GPU-side copy on reallocation.
//
// Mipped textures use the classic Iliffe / id Tech 1.5×1 layout: the
// packer reserves a `1.5W × H` rect, mip 0 occupies the left W×H block
// and mips 1..N stack vertically in the right W/2-wide column. CPU
// mip generation (canvas-2d) for v1; compute-shader path is future.
//
// When `tryAdd` fails on every existing page in the requested format,
// a new 4096² page is allocated.

import { V2f, V2i } from "@aardworx/wombat.base";
import type { aval } from "@aardworx/wombat.adaptive";
import type { ITexture, HostTextureSource } from "../../core/texture.js";
import { TexturePacking } from "./packer.js";

export const ATLAS_PAGE_SIZE = 4096;
/** Tier S/M source-side dimension cap. */
export const ATLAS_MAX_DIM = 1024;
/** Initial layer count per format-store. Pow2-grows on demand. */
export const ATLAS_INITIAL_LAYERS = 2;

export type AtlasPageFormat = "rgba8unorm" | "rgba8unorm-srgb";

export const ATLAS_PAGE_FORMATS: readonly AtlasPageFormat[] = ["rgba8unorm", "rgba8unorm-srgb"];

/** Format index used in the drawHeader: 0 = linear, 1 = srgb. */
export const atlasFormatIndex = (f: AtlasPageFormat): number =>
  f === "rgba8unorm-srgb" ? 1 : 0;

/**
 * Per-format storage: ONE `texture_2d_array` GPUTexture with N layers.
 * Pages are layer slots in the array. When a layer is needed past
 * `capacityLayers`, the texture is reallocated with `capacityLayers * 2`
 * layers and live layers are GPU-side-copied; listeners (heapScene
 * bind groups) get notified and rebuild.
 */
export interface AtlasFormatStore {
  readonly format: AtlasPageFormat;
  /** Backing GPUTexture (`dimension: "2d"`, `depthOrArrayLayers: capacityLayers`). Updated on realloc. */
  texture: GPUTexture;
  /** Total array layer capacity (pow2-grown). */
  capacityLayers: number;
  /** Number of layers currently in use (= pages.length). */
  liveLayerCount: number;
  /** Subscribers fire when `texture` is reallocated. */
  readonly onResize: (cb: () => void) => { dispose(): void };
}

export interface AtlasPage {
  readonly format: AtlasPageFormat;
  /**
   * Reference to the format's shared array texture. Reflects the
   * current GPUTexture (updated when the store reallocates).
   */
  texture: GPUTexture;
  /** Immutable packer state — `tryAdd`/`remove` produce new instances. */
  packing: TexturePacking<number>;
  /** Layer index in the shared `texture_2d_array` (== old `pageIndex`). */
  readonly pageIndex: number;
}

export interface AtlasAcquisition {
  /** Page index within the format's page set (0..N-1). */
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
  readonly pageIndex: number;
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
 * Aval-identity keyed pool over the per-format page sets. Multiple
 * acquisitions of the same `aval<ITexture>` share one sub-rect; the
 * refcount drives release.
 */
export class AtlasPool {
  private readonly pagesByFormat = new Map<AtlasPageFormat, AtlasPage[]>();
  private readonly storesByFormat = new Map<AtlasPageFormat, AtlasFormatStore>();
  private readonly resizeListeners = new Map<AtlasPageFormat, Set<() => void>>();
  private readonly entriesByAval = new Map<aval<ITexture>, AtlasEntry>();
  private readonly entriesByRef = new Map<number, AtlasEntry>();
  private nextRef = 1;

  constructor(private readonly device: GPUDevice) {
    for (const f of ATLAS_PAGE_FORMATS) {
      this.pagesByFormat.set(f, []);
      this.resizeListeners.set(f, new Set());
      this.storesByFormat.set(f, this.createStore(f, ATLAS_INITIAL_LAYERS));
    }
  }

  /** All currently allocated pages (= live layers) for a given format. */
  pagesFor(format: AtlasPageFormat): readonly AtlasPage[] {
    return this.pagesByFormat.get(format) ?? [];
  }

  /** The format's shared `texture_2d_array` GPUTexture. */
  formatStore(format: AtlasPageFormat): AtlasFormatStore {
    return this.storesByFormat.get(format)!;
  }

  /**
   * Subscribe to the format's `texture_2d_array` reallocation events.
   * Fires when `liveLayerCount` exceeds `capacityLayers` and the
   * texture is recreated with doubled layers (live data copied over).
   */
  onFormatResize(format: AtlasPageFormat, cb: () => void): { dispose(): void } {
    const set = this.resizeListeners.get(format)!;
    set.add(cb);
    return { dispose: () => { set.delete(cb); } };
  }

  private createStore(format: AtlasPageFormat, capacityLayers: number): AtlasFormatStore {
    const texture = this.device.createTexture({
      label: `atlas/${format}`,
      size: { width: ATLAS_PAGE_SIZE, height: ATLAS_PAGE_SIZE, depthOrArrayLayers: capacityLayers },
      format,
      mipLevelCount: 1,
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const store: AtlasFormatStore = {
      format,
      texture,
      capacityLayers,
      liveLayerCount: 0,
      onResize: (cb) => this.onFormatResize(format, cb),
    };
    return store;
  }

  /**
   * Grow the format's array texture so it can host `neededLayers`.
   * Pow2-grows. Allocates a new texture, GPU-copies the live layers,
   * destroys the old, fires resize listeners.
   */
  private growStore(format: AtlasPageFormat, neededLayers: number): void {
    const store = this.storesByFormat.get(format)!;
    if (neededLayers <= store.capacityLayers) return;
    let newCap = store.capacityLayers;
    while (newCap < neededLayers) newCap *= 2;
    const oldTex = store.texture;
    const newTex = this.device.createTexture({
      label: `atlas/${format}`,
      size: { width: ATLAS_PAGE_SIZE, height: ATLAS_PAGE_SIZE, depthOrArrayLayers: newCap },
      format,
      mipLevelCount: 1,
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    if (store.liveLayerCount > 0) {
      const enc = this.device.createCommandEncoder({ label: `atlas/${format}/grow-copy` });
      enc.copyTextureToTexture(
        { texture: oldTex, origin: { x: 0, y: 0, z: 0 } },
        { texture: newTex, origin: { x: 0, y: 0, z: 0 } },
        { width: ATLAS_PAGE_SIZE, height: ATLAS_PAGE_SIZE, depthOrArrayLayers: store.liveLayerCount },
      );
      this.device.queue.submit([enc.finish()]);
    }
    oldTex.destroy();
    store.texture = newTex;
    store.capacityLayers = newCap;
    const pages = this.pagesByFormat.get(format)!;
    for (const p of pages) p.texture = newTex;
    for (const cb of this.resizeListeners.get(format)!) cb();
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
   * gives the page index within `pagesFor(format)` plus the mip-0
   * `origin`/`size` (in normalized atlas coords) and `numMips`. If
   * the texture is already known (same aval), refcount bumps and we
   * return the existing placement.
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
    // The 1-pixel padding referenced in the design lives outside the
    // stored size — packer-level padding is not currently applied
    // (matches the previous non-mipped behaviour); add it uniformly
    // when we wire bilinear bleed handling.
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

    // Allocate a new layer in the format's array texture (grow if needed).
    const pageIndex = pages.length;
    this.growStore(format, pageIndex + 1);
    const store = this.storesByFormat.get(format)!;
    let packing = TexturePacking.empty<number>(new V2i(ATLAS_PAGE_SIZE, ATLAS_PAGE_SIZE), false);
    const placed = packing.tryAdd(this.nextRef, size);
    if (placed === null) {
      throw new Error(
        `AtlasPool: ${reservedW}×${reservedH} doesn't fit a fresh ${ATLAS_PAGE_SIZE}² page`,
      );
    }
    packing = placed;
    const page: AtlasPage = { format, texture: store.texture, packing, pageIndex };
    pages.push(page);
    store.liveLayerCount = pages.length;
    return this.finalize(sourceAval, format, pageIndex, page, opts.source, width, height, numMips);
  }

  /** Decrement the refcount; on zero, free the sub-rect. */
  release(ref: number): void {
    const e = this.entriesByRef.get(ref);
    if (e === undefined) return;
    e.refcount--;
    if (e.refcount > 0) return;
    const pages = this.pagesByFormat.get(e.format)!;
    const page = pages[e.pageIndex];
    if (page !== undefined) {
      page.packing = page.packing.remove(e.ref);
    }
    this.entriesByRef.delete(e.ref);
    this.entriesByAval.delete(e.key);
  }

  /** Destroy every format-store. The pool becomes unusable. */
  dispose(): void {
    for (const [, store] of this.storesByFormat) store.texture.destroy();
    this.storesByFormat.clear();
    this.pagesByFormat.clear();
    this.resizeListeners.clear();
    this.entriesByAval.clear();
    this.entriesByRef.clear();
  }

  private finalize(
    aval: aval<ITexture>,
    format: AtlasPageFormat,
    pageIndex: number,
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
      pageIndex,
      subRect: { x, y, w, h },
      mip0: { w: mip0W, h: mip0H },
      numMips,
      refcount: 1,
      ref,
    };
    this.entriesByAval.set(aval, entry);
    this.entriesByRef.set(ref, entry);
    if (source !== undefined && source.host !== undefined) {
      this.upload(page, pageIndex, x, y, source.host, mip0W, mip0H, numMips);
    }
    return this.makeResult(entry);
  }

  private upload(
    page: AtlasPage,
    layerIdx: number,
    x: number,
    y: number,
    host: HostTextureSource,
    w: number,
    h: number,
    numMips: number,
  ): void {
    // Mip 0 always lands at (x, y, layerIdx) with size w×h.
    this.uploadLevel(page, layerIdx, x, y, host, w, h);
    if (numMips <= 1) return;

    // Mip k≥1: downscale via canvas-2d, upload at the pyramid offset.
    // Source for downscaling: the original ImageBitmap/Canvas/etc. for
    // `external`; a freshly created ImageData for `raw`.
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
        { texture: page.texture, origin: { x: x + off.x, y: y + off.y, z: layerIdx } },
        { width: mw, height: mh, depthOrArrayLayers: 1 },
      );
    }
  }

  private uploadLevel(
    page: AtlasPage,
    layerIdx: number,
    x: number,
    y: number,
    host: HostTextureSource,
    w: number,
    h: number,
  ): void {
    if (host.kind === "external") {
      this.device.queue.copyExternalImageToTexture(
        { source: host.source as GPUImageCopyExternalImageSource },
        { texture: page.texture, origin: { x, y, z: layerIdx } },
        { width: w, height: h, depthOrArrayLayers: 1 },
      );
      return;
    }
    // raw upload — assume tightly packed rgba8 source aligned to the
    // page format. (BC-compressed sources don't fit Tier S; reject
    // upstream.)
    const data = host.data instanceof ArrayBuffer
      ? new Uint8Array(host.data)
      : new Uint8Array(host.data.buffer, host.data.byteOffset, host.data.byteLength);
    this.device.queue.writeTexture(
      { texture: page.texture, origin: { x, y, z: layerIdx } },
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
      // ImageBitmap / HTMLCanvasElement / HTMLImageElement / OffscreenCanvas /
      // HTMLVideoElement / ImageData (last needs a wrapping canvas).
      if (typeof ImageData !== "undefined" && s instanceof ImageData) {
        return this.imageDataToCanvas(s);
      }
      return s as CanvasImageSource;
    }
    // raw — synthesise an ImageData and wrap in a canvas.
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
    const page = this.pagesByFormat.get(e.format)![e.pageIndex]!;
    const inv = 1.0 / ATLAS_PAGE_SIZE;
    return {
      pageId: e.pageIndex,
      origin: new V2f(e.subRect.x * inv, e.subRect.y * inv),
      size: new V2f(e.mip0.w * inv, e.mip0.h * inv),
      numMips: e.numMips,
      ref: e.ref,
      page,
    };
  }
}
