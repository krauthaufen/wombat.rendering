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
import { type aval, HashTable } from "@aardworx/wombat.adaptive";
import type { ITexture, HostTextureSource } from "../../core/texture.js";
import { TexturePacking } from "./packer.js";
import {
  buildMipsAndGutterOnGpu, buildMipsAndGutterFromTexture, type MipSlot,
  beginMipGutterBatch, submitMipGutterBatch, type MipGutterBatch,
} from "./atlasMipGutterKernel.js";

export const ATLAS_PAGE_SIZE = 4096; // legacy default (see atlasPageSize)
/** Tier S/M source-side dimension cap. */
export const ATLAS_MAX_DIM = 1024;
/**
 * Maximum number of independent pages allowed per format. The shader's
 * `switch pageRef` runs an N-way ladder; the BGL declares N consecutive
 * texture slots per format. Allocations beyond this throw.
 */
// Pages are LAYERS of one `texture_2d_array` per format: the shader
// indexes `pageRef` at runtime, so the count is no longer pinned to a
// binding ladder. Layers grow on demand (new array texture + GPU-side
// layer copy + bind-group rebuild via onPageAdded); this cap only
// bounds the growth so a runaway streaming scene can't eat unbounded
// VRAM (16 × 4096² × rgba8 ≈ 1 GB per format family, lazily reached).
// Configurable via RuntimeOptions.atlasMaxPages.
export const ATLAS_MAX_PAGES_PER_FORMAT = 16;

/** Runtime-configurable layer cap (RuntimeOptions.atlasMaxPages).
 *  Routed through globalThis like the page size — bundlers can
 *  duplicate this module across entry points. */
export function atlasMaxPagesNow(): number {
  return ((globalThis as { __wombatAtlasMaxPages?: number }).__wombatAtlasMaxPages ?? ATLAS_MAX_PAGES_PER_FORMAT);
}
export function setAtlasMaxPages(n: number): void {
  (globalThis as { __wombatAtlasMaxPages?: number }).__wombatAtlasMaxPages = n;
}

/** Page edge in texels. Default suits small-memory devices; desktops
 *  pass RuntimeOptions.atlasPageSize = 8192 (must be set BEFORE any
 *  page allocation or shader generation). */
// NB: routed through globalThis, NOT a module `let` — bundlers can
// duplicate this module across the package's entry points, and a
// module-local value set via one copy is invisible to the other
// (observed: shader-gen kept emitting 4096 while pages were 8192).
export function atlasPageSizeNow(): number {
  return ((globalThis as { __wombatAtlasPageSize?: number }).__wombatAtlasPageSize ?? 4096);
}
export function setAtlasPageSize(n: number): void {
  (globalThis as { __wombatAtlasPageSize?: number }).__wombatAtlasPageSize = n;
}

export type AtlasPageFormat = "rgba8unorm" | "rgba8unorm-srgb";

export const ATLAS_PAGE_FORMATS: readonly AtlasPageFormat[] = ["rgba8unorm", "rgba8unorm-srgb"];

/** Format index used in the drawHeader: 0 = linear, 1 = srgb. */
export const atlasFormatIndex = (f: AtlasPageFormat): number =>
  f === "rgba8unorm-srgb" ? 1 : 0;

export interface AtlasPage {
  readonly format: AtlasPageFormat;
  /**
   * The format family's shared `texture_2d_array` — this page is layer
   * `pageId` of it. NOT stable across growth: adding pages past the
   * current layer count reallocates the array (existing layers are
   * GPU-copied across and `onPageAdded` fires so bind groups rebuild).
   * Always read through this getter, never cache the GPUTexture.
   */
  readonly texture: GPUTexture;
  /** Immutable packer state — `tryAdd`/`remove` produce new instances. */
  packing: TexturePacking<number>;
  /**
   * Layer index in the format's array texture (0..N-1). Same value
   * flows into the drawHeader `pageRef` field; the shader indexes the
   * array with it at runtime.
   */
  readonly pageId: number;
}

export interface AtlasAcquisition {
  /** Page slot index within the format's binding sequence (0..N-1). */
  readonly pageId: number;
  /** Top-left of mip 0's interior in the atlas, in atlas pixels. */
  readonly origin: V2f;
  /** Size of mip 0's interior in the atlas, in atlas pixels. */
  readonly size: V2f;
  /** Number of mip levels stored for this acquisition (1 = no pyramid). */
  readonly numMips: number;
  /** Stable refcount handle: pass back to `release`. */
  readonly ref: number;
  /** The page's GPUTexture (so the caller can pin it in the bind group). */
  readonly page: AtlasPage;
}

interface AtlasEntry {
  /**
   * Avals pointing to this entry. With §5b value-equality dedup,
   * multiple distinct constant avals can alias the same sub-rect —
   * `aliases` carries every one of them so final release can clear
   * all entriesByAval bindings. Length == 1 in the non-deduped case.
   */
  aliases: aval<ITexture>[];
  /**
   * Inner-resource reference used as the byValueKey for §5b dedup
   * (GPUTexture for kind:"gpu", HostTextureSource for kind:"host").
   * Undefined for non-constant avals (identity-only path).
   */
  readonly valueKey: GPUTexture | HostTextureSource | undefined;
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
 * Mip-k interior offset from mip-0 interior, both inside the embedded
 * Iliffe pyramid. Mip 0 is at (0,0). Mips 1..N stack vertically in
 * the column at x = W + 4 (mip-0 width + 2 px right gutter of mip-0
 * + 2 px left gutter of the mip column). Each mip k>=1 sits below
 * mip k-1 with 4 px of vertical gutter between them (2 px bottom of
 * mip k-1 + 2 px top of mip k):
 *   y_k = sum_{j=1..k-1} (max(1, H >> j) + 4).
 * Caller writes at `(uploadOrigin + mipOffsetInPyramid(w, h, k))`
 * where uploadOrigin is mip-0's interior pixel position.
 */
export const mipOffsetInPyramid = (
  w: number,
  h: number,
  k: number,
): { x: number; y: number } => {
  if (k === 0) return { x: 0, y: 0 };
  let y = 0;
  for (let j = 1; j < k; j++) y += Math.max(1, h >> j) + 4;
  return { x: w + 4, y };
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
  // URL-deferred textures are resolved at the Sg layer, never atlas-routed.
  if (t.kind === "url") return null;
  const src = t.source;
  if (src.kind === "raw") {
    return {
      format: src.format, width: src.width, height: src.height,
      mipLevelCount: src.mipLevelCount ?? 1, host: src,
    };
  }
  const ext = src.source as unknown;
  // Prefer the creation-time snapshot (survives ImageBitmap.close()).
  let w = src.width ?? 0, h = src.height ?? 0;
  if (w <= 0 || h <= 0) {
    if (typeof HTMLVideoElement !== "undefined" && ext instanceof HTMLVideoElement) {
      w = ext.videoWidth; h = ext.videoHeight;
    } else if (typeof ImageData !== "undefined" && ext instanceof ImageData) {
      w = ext.width; h = ext.height;
    } else {
      const any = ext as { width?: number; height?: number };
      w = any.width ?? 0; h = any.height ?? 0;
    }
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
  /** Per-format shared array texture (pages = layers). */
  private readonly storesByFormat = new Map<AtlasPageFormat, { texture: GPUTexture; layers: number }>();
  /**
   * Keyed by `aval<ITexture>` under the aval equality protocol: a
   * `HashTable` (not a JS `Map`) so two distinct `AVal.constant(tex)`
   * built at different call sites — including ones wrapping
   * structurally-equal `ITexture` values, e.g. `ITexture.fromUrl("X")`
   * twice (which are in fact the *same* interned object) — collapse to
   * one entry. Reactive (non-constant) avals key by reference, which is
   * correct since their value can change. This subsumes the old `§5b`
   * value-key side-table entirely.
   */
  private readonly entriesByAval = new HashTable<aval<ITexture>, AtlasEntry>();
  private readonly entriesByRef = new Map<number, AtlasEntry>();
  /**
   * §5b value-equality dedup map. Constant avals (`isConstant ===
   * true`) wrapping the same inner GPUTexture (kind:"gpu") or
   * HostTextureSource (kind:"host") collapse to one sub-rect.
   * Reactive avals fall through to identity-only since their
   * content can change.
   *
   * NOTE: now mostly vestigial — `entriesByAval` being a content-keyed
   * `HashTable` already collapses constant-aval siblings. Kept for the
   * (rare) case where two constant avals wrap *different* `ITexture`
   * wrapper objects that nevertheless reference the same inner GPU
   * resource AND don't compare equal under the ITexture protocol
   * (shouldn't happen with the factory functions, but a hand-built
   * `{kind:"gpu", texture}` literal wouldn't carry the protocol).
   */
  private readonly entriesByValueKey = new Map<GPUTexture | HostTextureSource, AtlasEntry>();
  /**
   * LRU of refcount-0 entries. Insertion order = age (oldest first).
   * Released entries land here instead of being freed immediately —
   * this lets a re-`acquire` of the same `aval<ITexture>` (or value-
   * keyed sibling) resurrect the sub-rect in O(1) with no GPU upload.
   * Entries leave the LRU when (a) `acquire` resurrects them, or
   * (b) a fresh acquire can't fit in any existing page and we need
   * to evict oldest-first to make room.
   *
   * Why we don't free on refcount → 0: a cset-driven workload (e.g.
   * the heap-demo-sg toggle) cycles the same texture's atlas slot
   * through 0-refs → fresh-acquire many times per second. The
   * eager-free path repeatedly upload-and-pack the same image; the
   * profiler showed `copyExternalImageToTexture` firing during every
   * toggle. With the LRU it fires only on first acquire.
   */
  private readonly lru = new Map<number, AtlasEntry>();
  /** Formats we've already reported as full (once per format, not per frame). */
  private readonly fullWarned = new Set<AtlasPageFormat>();
  /** One-shot closed-upload-source report (see `upload`). */
  private closedSourceWarned = false;
  /** True while every page of `format` is packed with live (referenced) entries. */
  isFull(format: AtlasPageFormat): boolean {
    const pages = this.pagesByFormat.get(format);
    if (pages === undefined || pages.length < atlasMaxPagesNow()) return false;
    for (const e of this.lru.values()) if (e.format === format) return false; // something is evictable
    return true;
  }
  private nextRef = 1;

  private innerKeyOf(t: ITexture): GPUTexture | HostTextureSource {
    if (t.kind === "gpu") return t.texture;
    if (t.kind === "host") return t.source;
    throw new Error(`AtlasPool: unsupported ITexture.kind "${t.kind}" (url textures must be resolved by the Sg layer first)`);
  }

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

  /** True for pages we can drive via the compute mip+gutter kernel.
   *  The kernel is buffer-based: it allocates a scratch storage
   *  buffer matching the sub-rect's bounding box, runs all mip
   *  downscale + gutter fill in compute passes over the buffer, then
   *  one `copyBufferToTexture` uploads the whole region to the page.
   *  No texture storage bindings are needed and no device features
   *  are required — works in core WebGPU 1.0. The only constraint is
   *  that JS-side pixel extraction succeeds, which falls back to the
   *  CPU path on headless platforms without canvas.
   *
   *  Disabled in node mock-GPU mode (which lacks queue.onSubmittedWorkDone). */
  private canUseGpuKernel(_page: AtlasPage): boolean {
    return typeof (this.device.queue as { onSubmittedWorkDone?: () => Promise<void> }).onSubmittedWorkDone === "function";
  }

  /**
   * Ensure the format's array texture has at least `layers` layers.
   * Growth allocates a fresh array (pow2 steps up to 8, then +2, capped
   * at `atlasMaxPagesNow()`), copies every existing layer across in one
   * `copyTextureToTexture`, and destroys the old texture (safe right
   * after submit — already-submitted work completes before destruction).
   * Callers fire `onPageAdded` afterwards, which rebuilds bind groups —
   * so the stale-texture window never reaches a render pass.
   */
  private ensureLayers(format: AtlasPageFormat, layers: number): { texture: GPUTexture; layers: number } {
    const cap = atlasMaxPagesNow();
    if (layers > cap) {
      throw new Error(`AtlasPool: layer request ${layers} exceeds atlasMaxPages ${cap}`);
    }
    const store = this.storesByFormat.get(format);
    if (store !== undefined && store.layers >= layers) return store;
    // +2 step (capped): a layer is huge (aps² × 4B — 268 MB at 8192²),
    // pow2 overshoot needlessly spikes VRAM (transient = old + new array
    // until the queue drains); +2 halves the number of grow-copies vs
    // exact-need. NO deferral here: a null acquisition for a NEW draw
    // has no retry path (measured: permanently untextured tiles after a
    // camera move), so growth must always succeed below the cap.
    const next = Math.min(Math.max(layers, (store?.layers ?? 0) + 2), cap);
    // The compute mip+gutter kernel works for *all* page formats: it
    // builds the pyramid + gutters in a scratch storage *buffer* and
    // finishes with `copyBufferToTexture` into the layer, so pages
    // never need STORAGE_BINDING — srgb included.
    const texture = this.device.createTexture({
      label: `atlas/${format}[${next}]`,
      size: { width: atlasPageSizeNow(), height: atlasPageSizeNow(), depthOrArrayLayers: next },
      format,
      mipLevelCount: 1,
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    if (store !== undefined && store.layers > 0) {
      // An OPEN upload batch may hold copies targeting the old texture —
      // submit it first, or those uploads land in the abandoned array
      // (and its destroy below invalidates their submit).
      this.flushUploadBatch();
      const enc = this.device.createCommandEncoder({ label: `atlas/${format}/grow` });
      enc.copyTextureToTexture(
        { texture: store.texture },
        { texture },
        { width: atlasPageSizeNow(), height: atlasPageSizeNow(), depthOrArrayLayers: store.layers },
      );
      this.device.queue.submit([enc.finish()]);
      // Immediate destroy is safe: the layer copy is already SUBMITTED
      // (submitted work completes before destruction takes effect), and
      // the batch flush above guarantees no later submit references the
      // old texture.
      store.texture.destroy();
    }
    const fresh = { texture, layers: next };
    this.storesByFormat.set(format, fresh);
    return fresh;
  }

  private allocatePage(format: AtlasPageFormat, pageId: number): AtlasPage {
    const pool = this;
    this.ensureLayers(format, pageId + 1);
    return {
      format,
      // getter: growth reallocates the array; pages must never cache it
      get texture(): GPUTexture { return pool.storesByFormat.get(format)!.texture; },
      packing: TexturePacking.empty<number>(new V2i(atlasPageSizeNow(), atlasPageSizeNow()), false),
      pageId,
    };
  }

  /** The format family's current array texture (undefined before the
   *  first page). Bind-group building reads this — pages are layers. */
  textureFor(format: AtlasPageFormat): GPUTexture | undefined {
    return this.storesByFormat.get(format)?.texture;
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
    // Returns null when the atlas is FULL (every page packed with live
    // entries). Callers must handle it — see the note at the throw site.
  ): AtlasAcquisition | null {
    const existing = this.entriesByAval.get(sourceAval);
    if (existing !== undefined) {
      const wasIdle = existing.refcount === 0;
      existing.refcount++;
      if (wasIdle) this.lru.delete(existing.ref);
      return this.makeResult(existing);
    }

    // §5b: value-equality dedup for constant avals. Two distinct
    // constant avals wrapping the same inner GPUTexture or
    // HostTextureSource share one sub-rect. We need a *value* (not
    // just dimensions) to compute the key; pull from `opts.source`
    // if provided, otherwise skip dedup (fresh allocation).
    let valueKey: GPUTexture | HostTextureSource | undefined;
    if (sourceAval.isConstant) {
      const sourceVal = opts.source;
      if (sourceVal !== undefined && sourceVal.host !== undefined) {
        valueKey = sourceVal.host;
      }
      // For kind:"gpu" sources, opts.source.host is undefined; the
      // caller (heapAdapter) provides the GPUTexture via a different
      // path and we'd need to thread it through. v1: skip dedup for
      // GPU-backed constants, dedup only host-backed constants. The
      // realistic dedup case is shared ImageBitmap / ImageData; GPU
      // textures are typically already deduped at the user layer.
      if (valueKey !== undefined) {
        const shared = this.entriesByValueKey.get(valueKey);
        if (shared !== undefined) {
          const wasIdle = shared.refcount === 0;
          shared.refcount++;
          shared.aliases.push(sourceAval);
          this.entriesByAval.set(sourceAval, shared);
          if (wasIdle) this.lru.delete(shared.ref);
          return this.makeResult(shared);
        }
      }
    }

    const wantsMips = opts.wantsMips === true;
    const numMips = wantsMips
      ? Math.max(1, Math.min(opts.numMips ?? defaultMipCount(width, height),
                             defaultMipCount(width, height)))
      : 1;

    // Every sub-rect (and every mip slot in mipped pyramids) carries a
    // 2-px gutter on each side. Inner ring = clamp-replicate edge
    // texel, outer ring = wrap (opposite-edge texel). Required so the
    // shader's hardware-bilinear at the sub-rect edge doesn't bleed
    // into a neighboring sub-rect and so the repeat-mode `shift ±1`
    // can land on the opposite-edge data.
    //
    // Non-mipped layout: (W+4) × (H+4); mip-0 interior at (+2, +2).
    // Mipped Iliffe layout: mip-0 owns (W+4) × (H+4); mips 1..N stack
    // vertically in the next column starting at x = W + 4, each with
    // its own 4-px gutter padding on each axis. Pyramid bounding rect
    // is `(W + 4 + maxMipW + 4) × max(H+4, sumMipH+gaps)` — sized
    // generously to ensure non-overlap.
    const reservedW = wantsMips
      ? (width + 4) + Math.max(1, width >> 1) + 4
      : width + 4;
    let reservedH = height + 4;
    if (wantsMips) {
      let stackedH = 0;
      for (let k = 1; k < numMips; k++) {
        stackedH += Math.max(1, height >> k) + 4;
      }
      reservedH = Math.max(reservedH, stackedH);
    }
    const size = new V2i(reservedW, reservedH);

    const pages = this.pagesByFormat.get(format)!;

    const tryFitInExisting = (): AtlasAcquisition | null => {
      for (let i = 0; i < pages.length; i++) {
        const page = pages[i]!;
        const next = page.packing.tryAdd(this.nextRef, size);
        if (next !== null) {
          page.packing = next;
          return this.finalize(sourceAval, format, i, page, opts.source, width, height, numMips, valueKey);
        }
      }
      return null;
    };

    // Try fitting into existing pages first.
    {
      const fit = tryFitInExisting();
      if (fit !== null) return fit;
    }

    // Page-fit failed. Try evicting LRU entries (oldest first) on this
    // format's pages until either the new size fits or the LRU is
    // empty. Each evicted entry frees its packer slot; we retry the
    // packer after every eviction since the freed slot may bridge two
    // existing gaps. Only entries with `format` matching the request
    // can help — wrong-format entries live on different pages.
    for (const [ref, entry] of this.lru) {
      if (entry.format !== format) continue;
      this.actuallyFree(entry);
      void ref;
      const fit = tryFitInExisting();
      if (fit !== null) return fit;
    }

    // Allocate a fresh page (array layer) if we haven't hit the cap.
    if (pages.length >= atlasMaxPagesNow()) {
      // The atlas is genuinely full: every page is packed with entries that are
      // still referenced, so there is nothing to evict. That is resource
      // pressure, NOT a program error — and throwing turned it into a fatal
      // one: the exception escapes into the render loop's frame callback, which
      // then dies (permanently, before `runFrame` learned to retry; and even
      // with retries the condition repeats every frame until the loop gives up).
      // A viewer that streams an unbounded world will hit this, and the right
      // answer is to skip the texture for now, not to stop rendering.
      //
      // The caller decides what "skip" means (see heapScene: the draw is held
      // back until space frees). We report it once per format so the condition
      // is visible without spamming a frame-rate-long log.
      if (!this.fullWarned.has(format)) {
        this.fullWarned.add(format);
        console.warn(
          `atlas: FULL for ${format} — ${pages.length} pages × ${atlasPageSizeNow()}² all in use ` +
          `(${this.entriesByRef.size} live entries, ${this.lru.size} evictable). ` +
          `New textures are deferred until space frees.`,
        );
      }
      return null;
    }
    const pageId = pages.length;
    const page = this.allocatePage(format, pageId);
    const placed = page.packing.tryAdd(this.nextRef, size);
    if (placed === null) {
      throw new Error(
        `AtlasPool: ${reservedW}×${reservedH} doesn't fit a fresh ${atlasPageSizeNow()}² page`,
      );
    }
    page.packing = placed;
    pages.push(page);
    for (const cb of this.pageAddedListeners.get(format)!) cb(pageId);
    return this.finalize(sourceAval, format, pageId, page, opts.source, width, height, numMips, valueKey);
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
    // null when the atlas is full — caller keeps its existing sub-rect.
  ): AtlasAcquisition | null {
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

    // Repack on a deduped entry (multiple aliases sharing one
    // sub-rect) is undefined: we'd be free-and-realloc'ing the
    // sub-rect underneath the OTHER aliases. The dedup path only
    // engages for `isConstant` avals, which can't fire mark
    // callbacks → repack(constant) shouldn't ever happen via the
    // reactivity loop. If a caller invokes it manually we'd rather
    // throw than silently corrupt the other aliases.
    if (old.aliases.length > 1) {
      throw new Error(
        "AtlasPool.repack: cannot repack a deduped entry shared by multiple constant avals",
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
    if (old.valueKey !== undefined) this.entriesByValueKey.delete(old.valueKey);

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
    // Atlas full: the caller keeps the OLD sub-rect (already freed above is not
    // possible here — repack only frees after a successful acquire), so the
    // texture simply doesn't update this cycle.
    if (acq === null) return null;

    // 3. Restore the original refcount (acquire set it to 1).
    const newEntry = this.entriesByRef.get(acq.ref);
    if (newEntry !== undefined) newEntry.refcount = savedRefcount;
    return acq;
  }

  /**
   * Decrement the refcount. On zero we *don't* eagerly free — the entry
   * moves to the LRU instead, so a re-acquire of the same aval (or a
   * value-keyed sibling) can resurrect it without re-uploading the
   * texture. Eviction (actual `actuallyFree`) happens lazily when a
   * fresh acquire can't fit and we need to reclaim packer space.
   */
  release(ref: number): void {
    const e = this.entriesByRef.get(ref);
    if (e === undefined) return;
    e.refcount--;
    if (e.refcount > 0) return;
    // Hold the entry idle in the LRU. Move-to-end (Map insertion-order
    // semantics): delete + set re-positions a key that was already
    // present, but the refcount-0 path shouldn't see one here — guard
    // anyway in case of double-release.
    this.lru.delete(e.ref);
    this.lru.set(e.ref, e);
  }

  /**
   * Bump refcount on an entry that's already live (used by the heap
   * path when a cached HeapDrawSpec is re-introduced: the spec already
   * carries `poolRef` from the original acquire, but every add/remove
   * cycle fires the release closure once — without a matching incRef
   * the refcount underflows and the entry gets actuallyFree'd while
   * live drawHeaders still reference it).
   *
   * Returns `true` if the bump was applied; `false` if the entry has
   * been evicted in the meantime (caller must fall back to re-acquire).
   */
  incRef(ref: number): boolean {
    const e = this.entriesByRef.get(ref);
    if (e === undefined) return false;
    if (e.refcount === 0) this.lru.delete(e.ref);
    e.refcount++;
    return true;
  }

  /**
   * Drop an LRU entry for real: remove its packer slot, drop the entry
   * from every lookup map. Caller must guarantee `refcount === 0`.
   */
  private actuallyFree(e: AtlasEntry): void {
    const pages = this.pagesByFormat.get(e.format)!;
    const page = pages[e.pageId];
    if (page !== undefined) {
      page.packing = page.packing.remove(e.ref);
    }
    this.entriesByRef.delete(e.ref);
    for (const a of e.aliases) this.entriesByAval.delete(a);
    if (e.valueKey !== undefined) this.entriesByValueKey.delete(e.valueKey);
    this.lru.delete(e.ref);
  }

  /**
   * Free every refcount-0 entry held by the LRU. Used by tests that
   * want to assert packer-empty after a release, and by callers that
   * want to release atlas memory pressure on demand. Returns the
   * number of entries freed. Outside of tests, eviction normally
   * happens lazily on packer pressure during a subsequent `acquire`.
   */
  evictIdle(): number {
    let n = 0;
    for (const [, entry] of this.lru) {
      this.actuallyFree(entry);
      n++;
    }
    return n;
  }

  /** Zero-fill an entry's full reserved rect (mip-0 + gutters + mip
   *  column). Used when an upload source is dead: black is the only
   *  acceptable content — the packer may hand back a region still
   *  holding another texture's texels. */
  private zeroFillReserved(page: AtlasPage, bx: number, by: number, w: number, h: number, numMips: number): void {
    const reservedW = numMips > 1 ? (w + 4) + Math.max(1, w >> 1) + 4 : w + 4;
    let reservedH = h + 4;
    if (numMips > 1) {
      let stackedH = 0;
      for (let k = 1; k < numMips; k++) stackedH += Math.max(1, h >> k) + 4;
      reservedH = Math.max(reservedH, stackedH);
    }
    const zeros = new Uint8Array(reservedW * reservedH * 4);
    this.writeRgba8Padded(page.texture, bx, by, zeros, reservedW, reservedH, page.pageId);
  }

  /**
   * PIN an entry by its source aval: bump the refcount so eviction can
   * never claim it while the owning BUILD is retained app-side. This is
   * how memory-lean callers (bitmaps closed after ingest) guarantee a
   * re-show never needs an upload source: the entry simply never leaves
   * the atlas until `unpinEntry`. Returns false if no live entry exists
   * (caller should keep its upload source in that case).
   */
  pinEntry(sourceAval: aval<ITexture>): boolean {
    const e = this.entriesByAval.get(sourceAval);
    if (e === undefined) return false;
    if (e.refcount === 0) this.lru.delete(e.ref);
    e.refcount++;
    return true;
  }

  /** Release a `pinEntry` ref. No-op if the entry is already gone. */
  unpinEntry(sourceAval: aval<ITexture>): void {
    const e = this.entriesByAval.get(sourceAval);
    if (e === undefined) return;
    this.release(e.ref);
  }

  /** Destroy the per-format array textures. The pool becomes unusable. */
  dispose(): void {
    for (const [, store] of this.storesByFormat) store.texture.destroy();
    this.storesByFormat.clear();
    this.pagesByFormat.clear();
    this.pageAddedListeners.clear();
    this.entriesByAval.clear();
    this.entriesByRef.clear();
    this.entriesByValueKey.clear();
    this.lru.clear();
  }

  // ── shared-encoder upload batching ─────────────────────────────────
  // While a batch is open, per-tile kernel work encodes into ONE encoder
  // and `flushUploadBatch` submits once — per-tile submits were measured
  // as multi-frame stalls when several tiles land per frame (mobile GPUs,
  // large photogrammetry textures). Callers that never open a batch keep
  // the old immediate-submit behavior.
  private uploadBatch: MipGutterBatch | null = null;

  /** Open (or keep) a shared upload batch. Pair with `flushUploadBatch`. */
  beginUploadBatch(): void {
    if (this.uploadBatch === null) this.uploadBatch = beginMipGutterBatch(this.device);
  }

  /** Submit all batched uploads in one queue submission. */
  flushUploadBatch(): void {
    if (this.uploadBatch !== null) {
      submitMipGutterBatch(this.device, this.uploadBatch);
      this.uploadBatch = null;
    }
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
    valueKey: GPUTexture | HostTextureSource | undefined,
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
      aliases: [aval],
      valueKey,
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
    if (valueKey !== undefined) this.entriesByValueKey.set(valueKey, entry);
    if (source !== undefined && source.host !== undefined) {
      // Upload places mip-0 at the interior (+2, +2) position. The
      // surrounding 2-px gutter (inner clamp ring + outer wrap ring)
      // should be filled here, but Dawn's WebGPU implementation
      // currently rejects same-texture copyTextureToTexture even
      // when the source and destination regions don't overlap
      // (validation says "overlapping layer ranges" — overly strict
      // vs §22.5.5). Until the planned compute mip+gutter kernel
      // lands, gutter cells stay uninitialized → edge bilinear
      // samples bleed at exact uv=0 / uv=1. For most atlas content
      // (low-frequency icon edges) this is visually invisible.
      this.upload(page, x + 2, y + 2, source.host, mip0W, mip0H, numMips);
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
  ): boolean {
    // CLOSED-SOURCE GUARD. Memory-lean callers close ImageBitmaps once
    // the pixels are in the atlas; if this entry was later evicted and
    // re-acquired, the upload source is gone (a closed ImageBitmap
    // reports width 0). Uploading would THROW out of the walker's
    // ingest and poison the whole delta batch. Instead: ZERO-FILL the
    // rect (the packer may hand back a region still holding another
    // texture's texels — showing those is unacceptable) and report
    // failure; `finalize` notifies onUploadSourceLost so the app can
    // repair asynchronously (re-decode from its own cache +
    // `repairUpload`).
    if (host.kind === "external") {
      const src = host.source as { width?: number; height?: number };
      if ((src.width ?? 0) === 0 || (src.height ?? 0) === 0) {
        if (!this.closedSourceWarned) {
          this.closedSourceWarned = true;
          console.warn("atlas: upload source closed/detached (width 0) — rect zero-filled, awaiting repairUpload");
        }
        this.zeroFillReserved(page, x - 2, y - 2, w, h, numMips);
        return false;
      }
    }
// GPU mip+gutter kernel path (the normal, real-device path).
    //
    // Builds the full Iliffe pyramid for the sub-rect — every mip,
    // each surrounded by the proper 2-px gutter ring (inner ring =
    // clamp-replicate of the nearest edge texel, outer ring = wrap of
    // the opposite edge) — entirely on the GPU, in a scratch storage
    // buffer, then a single `copyBufferToTexture` lands the real
    // texture (interior + gutter, all mips) in the atlas page. No
    // canvas-2d, no per-mip CPU buffer build. (`copyTextureToTexture`
    // can't be used for the gutter — WebGPU forbids same-(sub)resource
    // texture copies, §22.5.5 — which is why this goes through a
    // buffer.)
    //
    // Mip-0 source:
    //  · `external` host (ImageBitmap / <img> / <canvas> / …):
    //    `copyExternalImageToTexture` decodes it into a transient
    //    staging texture (decode happens GPU/compositor-side, off the
    //    main thread) and an `interiorFromTexture` compute pass copies
    //    that into the buffer — no `getImageData` round-trip (that
    //    canvas-2d path was ~8% of cold-boot CPU in the heap-demo-sg
    //    profile). The staging texture is freed once the work submits.
    //  · `raw` host: the pixels are already in memory; `extractPixels`
    //    is a free slice — write them straight into the buffer.
    if (this.canUseGpuKernel(page)) {
      // Bounding rect the sub-rect occupies: caller passed (x, y) as
      // the mip-0 interior position (sub-rect origin + 2/+2). Reserved
      // size matches the packer's allocation.
      const boundsX = x - 2;
      const boundsY = y - 2;
      const reservedW = numMips > 1
        ? (w + 4) + Math.max(1, w >> 1) + 4
        : w + 4;
      let reservedH = h + 4;
      if (numMips > 1) {
        let stackedH = 0;
        for (let k = 1; k < numMips; k++) stackedH += Math.max(1, h >> k) + 4;
        reservedH = Math.max(reservedH, stackedH);
      }
      // Mip slot offsets, in bounding-rect-relative pixel coords.
      // mip-0 interior is at (+2, +2) inside the bounds.
      const slots: MipSlot[] = [];
      for (let k = 0; k < numMips; k++) {
        const off = mipOffsetInPyramid(w, h, k);
        slots.push({
          origin: { x: 2 + off.x, y: 2 + off.y },
          size:   { w: Math.max(1, w >> k), h: Math.max(1, h >> k) },
        });
      }
      if (host.kind === "external") {
        try {
          const staging = this.device.createTexture({
            label: `atlas/staging(${w}x${h})`,
            size: { width: w, height: h, depthOrArrayLayers: 1 },
            format: "rgba8unorm",
            // copyExternalImageToTexture requires COPY_DST | RENDER_ATTACHMENT
            // on the destination; TEXTURE_BINDING for the kernel's read.
            usage:
              GPUTextureUsage.TEXTURE_BINDING |
              GPUTextureUsage.COPY_DST |
              GPUTextureUsage.RENDER_ATTACHMENT,
          });
          this.device.queue.copyExternalImageToTexture(
            { source: host.source as GPUImageCopyExternalImageSource },
            { texture: staging },
            { width: w, height: h, depthOrArrayLayers: 1 },
          );
          buildMipsAndGutterFromTexture(
            this.device, page.texture,
            boundsX, boundsY, reservedW, reservedH,
            staging, w, h, slots,
            this.uploadBatch ?? undefined,
            page.pageId,
          );
          if (this.uploadBatch !== null) this.uploadBatch.trashTextures.push(staging);
          else void this.device.queue.onSubmittedWorkDone().then(() => staging.destroy());
          return true;
        } catch {
          // Source not in a copyable state (e.g. a not-ready video) —
          // fall through to the CPU path below.
        }
      } else {
        const px = this.extractPixels(host, w, h);
        if (px !== null) {
          buildMipsAndGutterOnGpu(
            this.device, page.texture,
            boundsX, boundsY, reservedW, reservedH,
            px, w, h, slots,
            undefined,
            page.pageId,
          );
          return true;
        }
        // `raw` extraction shouldn't fail; if it somehow does, fall
        // through to the CPU path.
      }
    }
    // Fallback: node mock-GPU device (no compute kernel) or the kernel
    // path above threw. CPU gutter-extended upload — same visible
    // result. Mip 0 lands at (x, y) — already the interior position
    // (caller offset by +2/+2 for the gutter ring). Falls back to
    // plain uploadLevel if pixel extraction fails (headless w/o canvas).
    if (!this.uploadLevelWithGutter(page, x, y, host, w, h)) {
      this.uploadLevel(page, x, y, host, w, h);
    }
    if (numMips <= 1) return true;

    // Mip k≥1: downscale via canvas-2d, upload at the pyramid offset.
    const src = this.toDrawable(host, w, h);
    if (src === null) {
      // Headless / unsupported source — skip mip generation; level 0
      // is still in place. Callers that need mips should provide a
      // host source the pool can render-2d.
      return true;
    }
    for (let k = 1; k < numMips; k++) {
      const off = mipOffsetInPyramid(w, h, k);
      const mw = Math.max(1, w >> k);
      const mh = Math.max(1, h >> k);
      const mip = this.makeMipCanvas(src, mw, mh);
      if (mip === null) continue;
      const mx = x + off.x;
      const my = y + off.y;
      // Render the downsampled mip into a canvas, extract pixels,
      // build gutter-extended buffer, writeTexture once.
      const mipCanvas = this.makeCanvas(mw, mh);
      if (mipCanvas === null) {
        this.device.queue.copyExternalImageToTexture(
          { source: mip as GPUImageCopyExternalImageSource },
          { texture: page.texture, origin: { x: mx, y: my, z: page.pageId } },
          { width: mw, height: mh, depthOrArrayLayers: 1 },
        );
        continue;
      }
      const ctx = mipCanvas.getContext("2d") as
        | CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
      if (ctx === null) continue;
      ctx.drawImage(mip as CanvasImageSource, 0, 0, mw, mh);
      const mipPx = new Uint8Array(ctx.getImageData(0, 0, mw, mh).data.buffer);
      const ext = AtlasPool.buildExtendedWithGutter(mipPx, mw, mh);
      this.writeRgba8Padded(page.texture, mx - 2, my - 2, ext, mw + 4, mh + 4, page.pageId);
    }
    return true;
  }

  /**
   * Build a (w+4)×(h+4) extended pixel buffer from `src` (w×h, RGBA8)
   * with the 2-px gutter pre-filled:
   *   - Inner ring (1 px) = clamp-replicate (edge texel).
   *   - Outer ring (1 px) = wrap (opposite-edge texel).
   * The shader needs both: clamp-replicate absorbs FP-edge bleed for
   * any wrap mode; the outer wrap ring is sampled by the repeat
   * seam-shift ±1 path. By pre-baking these CPU-side we get a single
   * writeTexture call per slot — much faster than per-cell GPU
   * copies, and dodges Dawn's overly-strict same-texture copy
   * validation.
   *
   * Layout matches the shader expectations and `placeWithGutter` in
   * the conformance test. Corner cells are populated by independent
   * per-axis interpretation (e.g. outer-X × outer-Y = diagonal wrap;
   * inner-X × outer-Y = clamp-X × wrap-Y).
   */
  private static buildExtendedWithGutter(
    src: Uint8Array, w: number, h: number,
  ): Uint8Array {
    const ew = w + 4;
    const eh = h + 4;
    const out = new Uint8Array(ew * eh * 4);
    // pickSrcCoord: dx ∈ [-2..w+1] → (srcX, mode) per axis.
    const srcX = (dx: number): number => {
      if (dx === -2) return w - 1;            // outer wrap left  = opposite edge
      if (dx === -1) return 0;                // inner clamp left  = nearest edge
      if (dx === w) return w - 1;             // inner clamp right
      if (dx === w + 1) return 0;             // outer wrap right
      return dx;                              // interior
    };
    const srcY = (dy: number): number => {
      if (dy === -2) return h - 1;
      if (dy === -1) return 0;
      if (dy === h) return h - 1;
      if (dy === h + 1) return 0;
      return dy;
    };
    for (let dy = -2; dy < h + 2; dy++) {
      const sy = srcY(dy);
      const ey = dy + 2;
      for (let dx = -2; dx < w + 2; dx++) {
        const sx = srcX(dx);
        const ex = dx + 2;
        const si = (sy * w + sx) * 4;
        const oi = (ey * ew + ex) * 4;
        out[oi + 0] = src[si + 0]!;
        out[oi + 1] = src[si + 1]!;
        out[oi + 2] = src[si + 2]!;
        out[oi + 3] = src[si + 3]!;
      }
    }
    return out;
  }

  /**
   * Extract raw RGBA8 pixels from `host` at logical size (w, h).
   * Falls back to a canvas draw + getImageData for external sources;
   * raw sources are copied straight. Returns null when the runtime
   * can't synthesise pixels (headless without canvas, etc.).
   */
  private extractPixels(host: HostTextureSource, w: number, h: number): Uint8Array | null {
    if (host.kind !== "external") {
      const ab = host.data instanceof ArrayBuffer
        ? new Uint8Array(host.data)
        : new Uint8Array(host.data.buffer, host.data.byteOffset, host.data.byteLength);
      return ab.slice(0, w * h * 4);
    }
    const c = this.makeCanvas(w, h);
    if (c === null) return null;
    const ctx = c.getContext("2d") as
      | CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
    if (ctx === null) return null;
    ctx.drawImage(host.source as CanvasImageSource, 0, 0, w, h);
    return new Uint8Array(ctx.getImageData(0, 0, w, h).data.buffer);
  }

  /**
   * Upload a mip level INCLUDING its 2-px gutter ring. Writes a
   * single (w+4)×(h+4) block to (x-2, y-2) — caller passes the
   * interior position (x, y), this routine offsets by -2/-2 to
   * cover the gutter. Returns false if pixel extraction failed (very
   * rare; only on platforms without canvas support).
   */
  private uploadLevelWithGutter(
    page: AtlasPage, x: number, y: number,
    host: HostTextureSource, w: number, h: number,
  ): boolean {
    const px = this.extractPixels(host, w, h);
    if (px === null) return false;
    const ext = AtlasPool.buildExtendedWithGutter(px, w, h);
    this.writeRgba8Padded(page.texture, x - 2, y - 2, ext, w + 4, h + 4, page.pageId);
    return true;
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
        { texture: page.texture, origin: { x, y, z: page.pageId } },
        { width: w, height: h, depthOrArrayLayers: 1 },
      );
      return;
    }
    const data = host.data instanceof ArrayBuffer
      ? new Uint8Array(host.data)
      : new Uint8Array(host.data.buffer, host.data.byteOffset, host.data.byteLength);
    this.writeRgba8Padded(page.texture, x, y, data, w, h, page.pageId);
  }

  /**
   * Upload `src` (RGBA8, w×h) to `texture` at (x, y) via `writeTexture`.
   * `writeTexture` requires `bytesPerRow` to be a multiple of 256 when
   * the source has more than one row, so non-256-aligned widths need
   * a padded buffer.
   */
  private writeRgba8Padded(
    texture: GPUTexture, x: number, y: number,
    src: Uint8Array, w: number, h: number,
    layer = 0,
  ): void {
    const srcStride = w * 4;
    if (h === 1 || srcStride % 256 === 0) {
      // Single row or already aligned — pass straight through.
      this.device.queue.writeTexture(
        { texture, origin: { x, y, z: layer } },
        src as unknown as GPUAllowSharedBufferSource,
        { bytesPerRow: srcStride, rowsPerImage: h },
        { width: w, height: h, depthOrArrayLayers: 1 },
      );
      return;
    }
    const dstStride = Math.ceil(srcStride / 256) * 256;
    const padded = new Uint8Array(dstStride * h);
    for (let row = 0; row < h; row++) {
      padded.set(src.subarray(row * srcStride, (row + 1) * srcStride), row * dstStride);
    }
    this.device.queue.writeTexture(
      { texture, origin: { x, y, z: layer } },
      padded as unknown as GPUAllowSharedBufferSource,
      { bytesPerRow: dstStride, rowsPerImage: h },
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
    // Mip-0 interior sits at +2/+2 inside the reserved rect (skipping
    // the 2-px gutter ring on the top and left).
    return {
      pageId: e.pageId,
      origin: new V2f(e.subRect.x + 2, e.subRect.y + 2),
      size: new V2f(e.mip0.w, e.mip0.h),
      numMips: e.numMips,
      ref: e.ref,
      page,
    };
  }
}
