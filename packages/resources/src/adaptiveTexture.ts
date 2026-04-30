// prepareAdaptiveTexture — lift `aval<ITexture>` to
// `AdaptiveResource<GPUTexture>`.
//
// Behaviour matches `prepareAdaptiveBuffer`: on `compute`, if the
// source is `gpu` return the user's handle; if `host`, ensure we
// own a GPUTexture of the right size/format and upload via the
// queue (writeTexture for raw, copyExternalImageToTexture for
// external sources).
//
// Reuse: a host source with the same dimensions + format reuses
// the existing texture; mismatch triggers a fresh allocation.
//
// Mip-map generation is not implemented in this first cut —
// `generateMips: true` on an external source allocates the mip
// levels but only mip 0 is uploaded. (Mip generation requires a
// compute pass; that lands in a follow-up.)

import {
  AdaptiveResource,
  tryAcquire,
  tryRelease,
  type ITexture,
  type ExternalTextureSource,
  type RawTextureSource,
} from "@aardworx/wombat.rendering-core";
import {
  type AdaptiveToken,
  type aval,
} from "@aardworx/wombat.adaptive";
import { TextureUsage } from "./webgpuFlags.js";

export interface PrepareAdaptiveTextureOptions {
  /** Extra usage flags. `COPY_DST` and `TEXTURE_BINDING` are added automatically. */
  readonly usage?: GPUTextureUsageFlags;
  /** Optional debug label. */
  readonly label?: string;
}

interface OwnedDesc {
  width: number;
  height: number;
  depthOrArrayLayers: number;
  format: GPUTextureFormat;
  mipLevelCount: number;
}

function descsEqual(a: OwnedDesc, b: OwnedDesc): boolean {
  return (
    a.width === b.width
    && a.height === b.height
    && a.depthOrArrayLayers === b.depthOrArrayLayers
    && a.format === b.format
    && a.mipLevelCount === b.mipLevelCount
  );
}

function descFor(src: RawTextureSource | ExternalTextureSource): OwnedDesc {
  if (src.kind === "raw") {
    return {
      width: src.width,
      height: src.height,
      depthOrArrayLayers: src.depthOrArrayLayers ?? 1,
      format: src.format,
      mipLevelCount: src.mipLevelCount ?? 1,
    };
  }
  // external
  const s = src.source;
  let width = 0;
  let height = 0;
  if (s instanceof ImageData) {
    width = s.width; height = s.height;
  } else if ("naturalWidth" in s && typeof s.naturalWidth === "number") {
    // HTMLImageElement
    width = s.naturalWidth || s.width;
    height = s.naturalHeight || s.height;
  } else if ("videoWidth" in s && typeof s.videoWidth === "number") {
    // HTMLVideoElement
    width = s.videoWidth; height = s.videoHeight;
  } else {
    // ImageBitmap, HTMLCanvasElement, OffscreenCanvas — all have width/height
    width = (s as { width: number }).width;
    height = (s as { height: number }).height;
  }
  const generateMips = src.generateMips === true;
  return {
    width, height, depthOrArrayLayers: 1,
    format: src.format ?? "rgba8unorm",
    mipLevelCount: generateMips ? mipCount(width, height) : 1,
  };
}

function mipCount(w: number, h: number): number {
  return Math.floor(Math.log2(Math.max(w, h))) + 1;
}

class AdaptiveTexture extends AdaptiveResource<GPUTexture> {
  private _owned: GPUTexture | undefined = undefined;
  private _ownedDesc: OwnedDesc | undefined = undefined;

  constructor(
    private readonly device: GPUDevice,
    private readonly source: aval<ITexture>,
    private readonly opts: PrepareAdaptiveTextureOptions,
  ) {
    super();
  }

  protected create(): void { tryAcquire(this.source); }

  protected destroy(): void {
    if (this._owned !== undefined) {
      this._owned.destroy();
      this._owned = undefined;
      this._ownedDesc = undefined;
    }
    tryRelease(this.source);
  }

  override compute(token: AdaptiveToken): GPUTexture {
    const src = this.source.getValue(token);
    if (src.kind === "gpu") {
      if (this._owned !== undefined) {
        this._owned.destroy();
        this._owned = undefined;
        this._ownedDesc = undefined;
      }
      return src.texture;
    }
    const desc = descFor(src.source);
    const usage =
      (this.opts.usage ?? 0)
      | TextureUsage.COPY_DST
      | TextureUsage.TEXTURE_BINDING
      | (src.source.kind === "external" ? TextureUsage.RENDER_ATTACHMENT : 0);
    if (this._owned === undefined || this._ownedDesc === undefined || !descsEqual(this._ownedDesc, desc)) {
      if (this._owned !== undefined) this._owned.destroy();
      const tdesc: GPUTextureDescriptor = {
        size: { width: desc.width, height: desc.height, depthOrArrayLayers: desc.depthOrArrayLayers },
        format: desc.format,
        mipLevelCount: desc.mipLevelCount,
        usage,
        ...(this.opts.label !== undefined ? { label: this.opts.label } : {}),
      };
      this._owned = this.device.createTexture(tdesc);
      this._ownedDesc = desc;
    }
    if (src.source.kind === "raw") {
      uploadRaw(this.device, this._owned, src.source);
    } else {
      uploadExternal(this.device, this._owned, src.source);
    }
    return this._owned;
  }
}

function uploadRaw(device: GPUDevice, tex: GPUTexture, src: RawTextureSource): void {
  const bytes = src.data instanceof ArrayBuffer ? new Uint8Array(src.data) : new Uint8Array(src.data.buffer, src.data.byteOffset, src.data.byteLength);
  const block = blockInfoFor(src.format);
  const blocksPerRow = Math.ceil(src.width / block.width);
  const rowsOfBlocks = Math.ceil(src.height / block.height);
  device.queue.writeTexture(
    { texture: tex },
    bytes as unknown as GPUAllowSharedBufferSource,
    { bytesPerRow: blocksPerRow * block.bytesPerBlock, rowsPerImage: rowsOfBlocks },
    { width: src.width, height: src.height, depthOrArrayLayers: src.depthOrArrayLayers ?? 1 },
  );
}

function uploadExternal(device: GPUDevice, tex: GPUTexture, src: ExternalTextureSource): void {
  device.queue.copyExternalImageToTexture(
    { source: src.source as GPUImageCopyExternalImageSource },
    { texture: tex },
    { width: (src.source as { width: number }).width, height: (src.source as { height: number }).height, depthOrArrayLayers: 1 },
  );
}

interface BlockInfo {
  readonly width: number;        // pixels per block in x
  readonly height: number;       // pixels per block in y
  readonly bytesPerBlock: number;
}

/**
 * Block dimensions + size for a WebGPU texture format. Linear
 * (uncompressed) formats report 1×1 blocks. Block-compressed
 * formats (BC*, ETC*, ASTC*, EAC) use their native block size so
 * `writeTexture`'s bytesPerRow / rowsPerImage land on the right
 * block grid.
 */
function blockInfoFor(format: GPUTextureFormat): BlockInfo {
  switch (format) {
    // 8-bit
    case "r8unorm": case "r8snorm": case "r8uint": case "r8sint":
      return { width: 1, height: 1, bytesPerBlock: 1 };
    // 16-bit
    case "rg8unorm": case "rg8snorm": case "rg8uint": case "rg8sint":
    case "r16uint": case "r16sint": case "r16float":
      return { width: 1, height: 1, bytesPerBlock: 2 };
    // 32-bit
    case "rgba8unorm": case "rgba8unorm-srgb": case "rgba8snorm":
    case "rgba8uint": case "rgba8sint":
    case "bgra8unorm": case "bgra8unorm-srgb":
    case "rgb10a2unorm": case "rgb10a2uint":
    case "rg11b10ufloat": case "rgb9e5ufloat":
    case "rg16uint": case "rg16sint": case "rg16float":
    case "r32uint": case "r32sint": case "r32float":
      return { width: 1, height: 1, bytesPerBlock: 4 };
    // 64-bit
    case "rgba16uint": case "rgba16sint": case "rgba16float":
    case "rg32uint": case "rg32sint": case "rg32float":
      return { width: 1, height: 1, bytesPerBlock: 8 };
    // 128-bit
    case "rgba32uint": case "rgba32sint": case "rgba32float":
      return { width: 1, height: 1, bytesPerBlock: 16 };

    // ---- Block-compressed formats ----
    // BC1/2/3/4/5 (DXT/RGTC) — 4×4 blocks
    case "bc1-rgba-unorm": case "bc1-rgba-unorm-srgb":
    case "bc4-r-unorm":   case "bc4-r-snorm":
      return { width: 4, height: 4, bytesPerBlock: 8 };
    case "bc2-rgba-unorm": case "bc2-rgba-unorm-srgb":
    case "bc3-rgba-unorm": case "bc3-rgba-unorm-srgb":
    case "bc5-rg-unorm":   case "bc5-rg-snorm":
    case "bc6h-rgb-ufloat": case "bc6h-rgb-float":
    case "bc7-rgba-unorm":  case "bc7-rgba-unorm-srgb":
      return { width: 4, height: 4, bytesPerBlock: 16 };

    // ETC2 — 4×4 blocks
    case "etc2-rgb8unorm": case "etc2-rgb8unorm-srgb":
    case "etc2-rgb8a1unorm": case "etc2-rgb8a1unorm-srgb":
    case "eac-r11unorm":  case "eac-r11snorm":
      return { width: 4, height: 4, bytesPerBlock: 8 };
    case "etc2-rgba8unorm": case "etc2-rgba8unorm-srgb":
    case "eac-rg11unorm":  case "eac-rg11snorm":
      return { width: 4, height: 4, bytesPerBlock: 16 };

    // ASTC — variable block size, all 16 bytes/block.
    case "astc-4x4-unorm":   case "astc-4x4-unorm-srgb":   return { width: 4,  height: 4,  bytesPerBlock: 16 };
    case "astc-5x4-unorm":   case "astc-5x4-unorm-srgb":   return { width: 5,  height: 4,  bytesPerBlock: 16 };
    case "astc-5x5-unorm":   case "astc-5x5-unorm-srgb":   return { width: 5,  height: 5,  bytesPerBlock: 16 };
    case "astc-6x5-unorm":   case "astc-6x5-unorm-srgb":   return { width: 6,  height: 5,  bytesPerBlock: 16 };
    case "astc-6x6-unorm":   case "astc-6x6-unorm-srgb":   return { width: 6,  height: 6,  bytesPerBlock: 16 };
    case "astc-8x5-unorm":   case "astc-8x5-unorm-srgb":   return { width: 8,  height: 5,  bytesPerBlock: 16 };
    case "astc-8x6-unorm":   case "astc-8x6-unorm-srgb":   return { width: 8,  height: 6,  bytesPerBlock: 16 };
    case "astc-8x8-unorm":   case "astc-8x8-unorm-srgb":   return { width: 8,  height: 8,  bytesPerBlock: 16 };
    case "astc-10x5-unorm":  case "astc-10x5-unorm-srgb":  return { width: 10, height: 5,  bytesPerBlock: 16 };
    case "astc-10x6-unorm":  case "astc-10x6-unorm-srgb":  return { width: 10, height: 6,  bytesPerBlock: 16 };
    case "astc-10x8-unorm":  case "astc-10x8-unorm-srgb":  return { width: 10, height: 8,  bytesPerBlock: 16 };
    case "astc-10x10-unorm": case "astc-10x10-unorm-srgb": return { width: 10, height: 10, bytesPerBlock: 16 };
    case "astc-12x10-unorm": case "astc-12x10-unorm-srgb": return { width: 12, height: 10, bytesPerBlock: 16 };
    case "astc-12x12-unorm": case "astc-12x12-unorm-srgb": return { width: 12, height: 12, bytesPerBlock: 16 };

    default:
      throw new Error(`blockInfoFor: unsupported format ${format}`);
  }
}

/**
 * Wrap an `aval<ITexture>` as a ref-counted, adaptively-uploaded
 * `AdaptiveResource<GPUTexture>`.
 */
export function prepareAdaptiveTexture(
  device: GPUDevice,
  source: aval<ITexture>,
  opts: PrepareAdaptiveTextureOptions = {},
): AdaptiveResource<GPUTexture> {
  return new AdaptiveTexture(device, source, opts);
}
