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

  protected create(): void {}

  protected destroy(): void {
    if (this._owned !== undefined) {
      this._owned.destroy();
      this._owned = undefined;
      this._ownedDesc = undefined;
    }
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
  const bytesPerPixel = bytesPerPixelFor(src.format);
  device.queue.writeTexture(
    { texture: tex },
    bytes as unknown as GPUAllowSharedBufferSource,
    { bytesPerRow: src.width * bytesPerPixel, rowsPerImage: src.height },
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

function bytesPerPixelFor(format: GPUTextureFormat): number {
  // Minimal table — covers the formats we expect in practice. Extend as needed.
  switch (format) {
    case "r8unorm": case "r8snorm": case "r8uint": case "r8sint": return 1;
    case "rg8unorm": case "rg8snorm": case "rg8uint": case "rg8sint":
    case "r16uint": case "r16sint": case "r16float": return 2;
    case "rgba8unorm": case "rgba8unorm-srgb": case "rgba8snorm":
    case "rgba8uint": case "rgba8sint":
    case "bgra8unorm": case "bgra8unorm-srgb":
    case "rgb10a2unorm": case "rgb10a2uint":
    case "rg11b10ufloat": case "rgb9e5ufloat":
    case "rg16uint": case "rg16sint": case "rg16float":
    case "r32uint": case "r32sint": case "r32float": return 4;
    case "rgba16uint": case "rgba16sint": case "rgba16float":
    case "rg32uint": case "rg32sint": case "rg32float": return 8;
    case "rgba32uint": case "rgba32sint": case "rgba32float": return 16;
    default:
      throw new Error(`bytesPerPixelFor: unsupported format ${format}`);
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
