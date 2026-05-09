// AtlasPool — aval-keyed refcounted sub-rect allocator over shared
// 4096×4096 atlas pages.
//
// Mirrors the UniformPool / IndexPool pattern in heapScene: identity
// keyed on the source `aval<ITexture>`; multiple ROs referencing the
// same texture share one sub-rect; refcount drives free-on-release.
//
// MVP scope (per docs/heap-textures-plan.md):
//   - Tier S only: source ≤ 1024×1024, no mip chain.
//   - Two formats: rgba8unorm, rgba8unorm-srgb. Each format has its
//     own page set (separate `binding_array` in the bind group).
//   - Pages are independent GPUTextures, NOT a texture_2d_array.
//   - No eviction. No grow-copy.
//
// When `tryAdd` fails on every existing page in the requested format,
// a new 4096² page is allocated.

import { V2i } from "@aardworx/wombat.base";
import type { aval } from "@aardworx/wombat.adaptive";
import type { ITexture, HostTextureSource } from "../../core/texture.js";
import { TexturePacking } from "./packer.js";

export const ATLAS_PAGE_SIZE = 4096;
/** Tier S source-side dimension cap. */
export const ATLAS_MAX_DIM = 1024;
/** Max pages bound per format in one bind group (`binding_array<...,N>`). */
export const ATLAS_MAX_PAGES_PER_FORMAT = 8;

export type AtlasPageFormat = "rgba8unorm" | "rgba8unorm-srgb";

export const ATLAS_PAGE_FORMATS: readonly AtlasPageFormat[] = ["rgba8unorm", "rgba8unorm-srgb"];

/** Format index used in the drawHeader: 0 = linear, 1 = srgb. */
export const atlasFormatIndex = (f: AtlasPageFormat): number =>
  f === "rgba8unorm-srgb" ? 1 : 0;

export interface AtlasPage {
  readonly format: AtlasPageFormat;
  /** Owning GPU texture, allocated lazily on first acquisition. */
  texture: GPUTexture;
  /** Immutable packer state — `tryAdd`/`remove` produce new instances. */
  packing: TexturePacking<number>;
  /** Pool-internal page index (0..N within its format). */
  readonly pageIndex: number;
}

export interface AtlasAcquisition {
  /** Page index within the format's page set (0..N-1). */
  readonly pageId: number;
  /** UV scale for sub-rect — multiply incoming UV (in [0,1]) by this. */
  readonly uvScale: { readonly x: number; readonly y: number };
  /** UV bias for sub-rect — add after multiplying by uvScale. */
  readonly uvBias: { readonly x: number; readonly y: number };
  /** Stable refcount handle: pass back to `release`. */
  readonly ref: number;
  /** The page's GPUTexture (so the caller can pin it in the bind group). */
  readonly page: AtlasPage;
}

interface AtlasEntry {
  readonly key: aval<ITexture>;
  readonly format: AtlasPageFormat;
  readonly pageIndex: number;
  readonly subRect: { x: number; y: number; w: number; h: number };
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
   * sub-rect is allocated.
   */
  readonly host?: HostTextureSource;
}

/**
 * Aval-identity keyed pool over the per-format page sets. Multiple
 * acquisitions of the same `aval<ITexture>` share one sub-rect; the
 * refcount drives release.
 */
export class AtlasPool {
  private readonly pagesByFormat = new Map<AtlasPageFormat, AtlasPage[]>();
  private readonly entriesByAval = new Map<aval<ITexture>, AtlasEntry>();
  private readonly entriesByRef = new Map<number, AtlasEntry>();
  private nextRef = 1;

  constructor(private readonly device: GPUDevice) {
    for (const f of ATLAS_PAGE_FORMATS) this.pagesByFormat.set(f, []);
  }

  /** All currently allocated pages for a given format. */
  pagesFor(format: AtlasPageFormat): readonly AtlasPage[] {
    return this.pagesByFormat.get(format) ?? [];
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
   * gives the page index within `pagesFor(format)` plus the UV
   * transform. If the texture is already known (same aval), refcount
   * bumps and we return the existing placement.
   */
  acquire(
    format: AtlasPageFormat,
    sourceAval: aval<ITexture>,
    width: number,
    height: number,
    options: { source?: AtlasSource } = {},
  ): AtlasAcquisition {
    const existing = this.entriesByAval.get(sourceAval);
    if (existing !== undefined) {
      existing.refcount++;
      return this.makeResult(existing);
    }

    const pages = this.pagesByFormat.get(format)!;
    const size = new V2i(width, height);

    // Try fitting into existing pages first.
    for (let i = 0; i < pages.length; i++) {
      const page = pages[i]!;
      const next = page.packing.tryAdd(this.nextRef, size);
      if (next !== null) {
        page.packing = next;
        return this.finalize(sourceAval, format, i, page, options.source);
      }
    }

    // Allocate a new page.
    const pageIndex = pages.length;
    const tex = this.device.createTexture({
      label: `atlasPage/${format}/${pageIndex}`,
      size: { width: ATLAS_PAGE_SIZE, height: ATLAS_PAGE_SIZE, depthOrArrayLayers: 1 },
      format,
      mipLevelCount: 1,
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    let packing = TexturePacking.empty<number>(new V2i(ATLAS_PAGE_SIZE, ATLAS_PAGE_SIZE), false);
    const placed = packing.tryAdd(this.nextRef, size);
    if (placed === null) {
      throw new Error(
        `AtlasPool: ${width}×${height} doesn't fit a fresh ${ATLAS_PAGE_SIZE}² page`,
      );
    }
    packing = placed;
    const page: AtlasPage = { format, texture: tex, packing, pageIndex };
    pages.push(page);
    return this.finalize(sourceAval, format, pageIndex, page, options.source);
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

  /** Destroy every allocated page. The pool becomes unusable. */
  dispose(): void {
    for (const [, pages] of this.pagesByFormat) for (const p of pages) p.texture.destroy();
    this.pagesByFormat.clear();
    this.entriesByAval.clear();
    this.entriesByRef.clear();
  }

  private finalize(
    aval: aval<ITexture>,
    format: AtlasPageFormat,
    pageIndex: number,
    page: AtlasPage,
    source: AtlasSource | undefined,
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
      refcount: 1,
      ref,
    };
    this.entriesByAval.set(aval, entry);
    this.entriesByRef.set(ref, entry);
    if (source !== undefined && source.host !== undefined) {
      this.upload(page, x, y, source.host, w, h);
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
  ): void {
    if (host.kind === "external") {
      this.device.queue.copyExternalImageToTexture(
        { source: host.source as GPUImageCopyExternalImageSource },
        { texture: page.texture, origin: { x, y } },
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
      { texture: page.texture, origin: { x, y } },
      data as unknown as GPUAllowSharedBufferSource,
      { bytesPerRow: w * 4, rowsPerImage: h },
      { width: w, height: h, depthOrArrayLayers: 1 },
    );
  }

  private makeResult(e: AtlasEntry): AtlasAcquisition {
    const page = this.pagesByFormat.get(e.format)![e.pageIndex]!;
    const inv = 1.0 / ATLAS_PAGE_SIZE;
    return {
      pageId: e.pageIndex,
      uvScale: { x: e.subRect.w * inv, y: e.subRect.h * inv },
      uvBias: { x: e.subRect.x * inv, y: e.subRect.y * inv },
      ref: e.ref,
      page,
    };
  }
}
