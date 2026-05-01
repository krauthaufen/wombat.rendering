// allocateFramebuffer — `aval<{width,height}>` → `AdaptiveResource<IFramebuffer>`.
//
// The signature defines names + formats; the size aval drives
// allocation. Each color attachment gets its own GPUTexture sized
// to the current `width × height`; depth/stencil similarly. On
// size change, all textures are torn down and reallocated.
//
// This is the building block for `runtime.renderTo(scene, size)`:
//   - The render task acquires the framebuffer's resource;
//   - Each frame, evaluates the FBO via its own AdaptiveResource
//     pipeline, getting current views;
//   - Encodes a render pass into those views;
//   - Exposes the chosen color attachment as an
//     `aval<ITexture>` so downstream samplers see live data.

import { AdaptiveResource, type IFramebuffer } from "@aardworx/wombat.rendering-core";
import { HashMap, type AdaptiveToken, type aval } from "@aardworx/wombat.adaptive";
import { TextureUsage } from "./webgpuFlags.js";

export interface FramebufferSize {
  readonly width: number;
  readonly height: number;
}

export interface AllocateFramebufferOptions {
  /** Extra usage flags ORed into every attachment. `RENDER_ATTACHMENT` is added automatically. */
  readonly extraUsage?: GPUTextureUsageFlags;
  /** Optional debug-label prefix; attachment name is appended. */
  readonly labelPrefix?: string;
}

class AdaptiveFramebuffer extends AdaptiveResource<IFramebuffer> {
  private _ownedColors = new Map<string, GPUTexture>();
  private _ownedResolves = new Map<string, GPUTexture>();
  private _ownedDepth: GPUTexture | undefined;
  private _lastSize: FramebufferSize | undefined;

  constructor(
    private readonly device: GPUDevice,
    private readonly signature: import("@aardworx/wombat.rendering-core").FramebufferSignature,
    private readonly size: aval<FramebufferSize>,
    private readonly opts: AllocateFramebufferOptions,
  ) {
    super();
  }

  protected create(): void {}

  protected destroy(): void {
    for (const t of this._ownedColors.values()) t.destroy();
    this._ownedColors.clear();
    for (const t of this._ownedResolves.values()) t.destroy();
    this._ownedResolves.clear();
    if (this._ownedDepth !== undefined) {
      this._ownedDepth.destroy();
      this._ownedDepth = undefined;
    }
    this._lastSize = undefined;
  }

  override compute(token: AdaptiveToken): IFramebuffer {
    const size = this.size.getValue(token);
    if (
      this._lastSize === undefined
      || this._lastSize.width !== size.width
      || this._lastSize.height !== size.height
    ) {
      this.reallocate(size);
      this._lastSize = size;
    }
    const msaa = this.signature.sampleCount > 1;
    let colorViews = HashMap.empty<string, GPUTextureView>();
    let colorTextures = HashMap.empty<string, GPUTexture>();
    let resolveViews = HashMap.empty<string, GPUTextureView>();
    for (const [name, tex] of this._ownedColors.entries()) {
      colorViews = colorViews.add(name, tex.createView());
      if (msaa) {
        const resolve = this._ownedResolves.get(name);
        if (resolve === undefined) {
          throw new Error(`framebuffer: missing resolve target for "${name}"`);
        }
        resolveViews = resolveViews.add(name, resolve.createView());
        colorTextures = colorTextures.add(name, resolve);
      } else {
        colorTextures = colorTextures.add(name, tex);
      }
    }
    return {
      signature: this.signature,
      colors: colorViews,
      colorTextures,
      ...(msaa ? { resolveColors: resolveViews } : {}),
      ...(this._ownedDepth ? { depthStencil: this._ownedDepth.createView(), depthStencilTexture: this._ownedDepth } : {}),
      width: size.width,
      height: size.height,
    };
  }

  private reallocate(size: FramebufferSize): void {
    // Destroy old.
    for (const t of this._ownedColors.values()) t.destroy();
    this._ownedColors.clear();
    for (const t of this._ownedResolves.values()) t.destroy();
    this._ownedResolves.clear();
    if (this._ownedDepth !== undefined) {
      this._ownedDepth.destroy();
      this._ownedDepth = undefined;
    }
    const sampleCount = this.signature.sampleCount;
    const msaa = sampleCount > 1;
    // Multisample textures are not directly sampleable; they only
    // serve as render attachments, with a separate single-sample
    // resolve target carrying TEXTURE_BINDING.
    const colorUsage = msaa
      ? TextureUsage.RENDER_ATTACHMENT
      : TextureUsage.RENDER_ATTACHMENT
        | TextureUsage.TEXTURE_BINDING
        | (this.opts.extraUsage ?? 0);
    const resolveUsage = TextureUsage.RENDER_ATTACHMENT
      | TextureUsage.TEXTURE_BINDING
      | (this.opts.extraUsage ?? 0);
    const labelPrefix = this.opts.labelPrefix ?? "fbo";

    for (const [name, format] of this.signature.colors) {
      const tex = this.device.createTexture({
        size: { width: size.width, height: size.height, depthOrArrayLayers: 1 },
        format,
        usage: colorUsage,
        sampleCount,
        label: msaa ? `${labelPrefix}.${name}.msaa` : `${labelPrefix}.${name}`,
      });
      this._ownedColors.set(name, tex);
      if (msaa) {
        const resolve = this.device.createTexture({
          size: { width: size.width, height: size.height, depthOrArrayLayers: 1 },
          format,
          usage: resolveUsage,
          sampleCount: 1,
          label: `${labelPrefix}.${name}.resolve`,
        });
        this._ownedResolves.set(name, resolve);
      }
    }
    if (this.signature.depthStencil !== undefined) {
      this._ownedDepth = this.device.createTexture({
        size: { width: size.width, height: size.height, depthOrArrayLayers: 1 },
        format: this.signature.depthStencil.format,
        usage: TextureUsage.RENDER_ATTACHMENT,
        sampleCount,
        label: `${labelPrefix}.depthStencil`,
      });
    }
  }
}

export function allocateFramebuffer(
  device: GPUDevice,
  signature: import("@aardworx/wombat.rendering-core").FramebufferSignature,
  size: aval<FramebufferSize>,
  opts: AllocateFramebufferOptions = {},
): AdaptiveResource<IFramebuffer> {
  return new AdaptiveFramebuffer(device, signature, size, opts);
}
