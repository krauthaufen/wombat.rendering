// attachCanvas — bind a `GPUDevice` to an `HTMLCanvasElement` /
// `OffscreenCanvas` and expose the swap-chain as an
// `aval<IFramebuffer>` so render tasks just plug it into a
// `Render` command's `output`.
//
// Each call to `framebuffer.getValue(token)` returns a fresh
// `IFramebuffer` whose color view is the current swap-chain
// texture (`context.getCurrentTexture().createView()`). Inside
// the render loop we mark the framebuffer aval outdated once per
// rAF tick so the runtime samples the new texture.
//
// The size aval is fed by a `ResizeObserver`; on size change we
// re-`configure()` the canvas context and bump a cval that the
// framebuffer aval depends on.
//
// MSAA: when `sampleCount > 1`, we allocate a hidden multisample
// color (and depth) texture sized to the canvas; the swap-chain
// texture becomes the resolve target. The render-pass descriptor
// builder in `@aardworx/wombat.rendering-commands` already wires
// `resolveTarget` based on the framebuffer's `resolveColors`.

import {
  type IFramebuffer,
  type FramebufferSignature,
} from "@aardworx/wombat.rendering-core";
import {
  HashMap,
  cval,
  transact,
  type aval,
  type ChangeableValue,
} from "@aardworx/wombat.adaptive";

export interface AttachCanvasOptions {
  /** Texture format for the swap-chain. Default: device's preferred format. */
  readonly format?: GPUTextureFormat;
  /** Optional depth/stencil format. When set, an off-screen depth texture is co-allocated. */
  readonly depthFormat?: GPUTextureFormat;
  /** Pre-multiplied / opaque alpha mode. Default `"premultiplied"`. */
  readonly alphaMode?: GPUCanvasAlphaMode;
  /** Logical → physical pixel scaling. Default: `window.devicePixelRatio`. */
  readonly devicePixelRatio?: number;
  /** Color attachment name in the resulting `FramebufferSignature`. Default `"color"`. */
  readonly colorAttachmentName?: string;
  /**
   * MSAA sample count. Default 1. With `> 1`, a hidden multisample
   * color texture (and depth, if `depthFormat` is set) is allocated
   * sized to the canvas; the swap-chain texture becomes the resolve
   * target. Common values: 1 (off), 4 (fast MSAA).
   */
  readonly sampleCount?: number;
}

export interface CanvasAttachment {
  /** Reactive framebuffer; bump per frame via `markFrame()`. */
  readonly framebuffer: aval<IFramebuffer>;
  /** Reactive `{ width, height }` driven by ResizeObserver. */
  readonly size: aval<{ readonly width: number; readonly height: number }>;
  /** The signature derived from the chosen formats. */
  readonly signature: FramebufferSignature;
  /** Mark a new frame: invalidates the framebuffer aval so its next read produces a fresh swap-chain texture. */
  markFrame(): void;
  /** Disconnect the resize observer + clear the canvas context. Idempotent. */
  dispose(): void;
}

export function attachCanvas(
  device: GPUDevice,
  canvas: HTMLCanvasElement | OffscreenCanvas,
  opts: AttachCanvasOptions = {},
): CanvasAttachment {
  const ctx = canvas.getContext("webgpu") as GPUCanvasContext | null;
  if (ctx === null) throw new Error("attachCanvas: getContext('webgpu') returned null");
  const colorName = opts.colorAttachmentName ?? "color";
  const format = opts.format ?? navigator.gpu.getPreferredCanvasFormat();
  const alphaMode = opts.alphaMode ?? "premultiplied";
  const sampleCount = opts.sampleCount ?? 1;

  const dpr = opts.devicePixelRatio ?? (typeof window !== "undefined" ? window.devicePixelRatio : 1);
  // initial size from the canvas's CSS size (or its native dimensions for OffscreenCanvas).
  const initial = currentSize(canvas, dpr);
  const sizeC: ChangeableValue<{ width: number; height: number }> = cval({ width: initial.width, height: initial.height });
  const frameC: ChangeableValue<number> = cval(0);

  // Apply the initial configure. We re-configure on every size change.
  configure(ctx, device, format, alphaMode);
  applySize(canvas, initial.width, initial.height);

  // Hidden multisample color texture, present only when sampleCount > 1.
  let msaaColor: GPUTexture | undefined;
  const ensureMsaaColor = (w: number, h: number): GPUTexture | undefined => {
    if (sampleCount <= 1) return undefined;
    if (msaaColor !== undefined) msaaColor.destroy();
    msaaColor = device.createTexture({
      size: { width: w, height: h, depthOrArrayLayers: 1 },
      format,
      sampleCount,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      label: "canvas.color.msaa",
    });
    return msaaColor;
  };
  ensureMsaaColor(initial.width, initial.height);

  // Optional shared depth texture re-allocated on size change.
  let depthTexture: GPUTexture | undefined;
  const ensureDepth = (w: number, h: number): GPUTexture | undefined => {
    if (opts.depthFormat === undefined) return undefined;
    if (depthTexture !== undefined) depthTexture.destroy();
    depthTexture = device.createTexture({
      size: { width: w, height: h, depthOrArrayLayers: 1 },
      format: opts.depthFormat,
      sampleCount,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      label: "canvas.depth",
    });
    return depthTexture;
  };
  ensureDepth(initial.width, initial.height);

  // ResizeObserver — only meaningful for HTMLCanvasElement.
  let observer: ResizeObserver | undefined;
  if (typeof ResizeObserver !== "undefined" && canvas instanceof (globalThis as unknown as { HTMLCanvasElement: typeof HTMLCanvasElement }).HTMLCanvasElement) {
    observer = new ResizeObserver((entries) => {
      for (const e of entries) {
        const cssWidth = e.contentBoxSize?.[0]?.inlineSize ?? canvas.clientWidth;
        const cssHeight = e.contentBoxSize?.[0]?.blockSize ?? canvas.clientHeight;
        const w = Math.max(1, Math.floor(cssWidth * dpr));
        const h = Math.max(1, Math.floor(cssHeight * dpr));
        if (w !== sizeC.value.width || h !== sizeC.value.height) {
          applySize(canvas, w, h);
          ensureMsaaColor(w, h);
          ensureDepth(w, h);
          transact(() => {
            sizeC.value = { width: w, height: h };
          });
        }
      }
    });
    observer.observe(canvas);
  }

  const signature: FramebufferSignature = {
    colors: HashMap.empty<string, GPUTextureFormat>().add(colorName, format),
    sampleCount,
    ...(opts.depthFormat !== undefined
      ? { depthStencil: depthAttachmentSig(opts.depthFormat) }
      : {}),
  };

  // The framebuffer aval depends on (size, frame). Reading it pulls
  // a fresh swap-chain texture; the frameC bump after each markFrame()
  // is what re-fires the chain on the same size.
  const framebuffer: aval<IFramebuffer> = sizeC.bind((sz) =>
    frameC.map(() => makeFramebuffer(ctx, signature, sz, msaaColor, depthTexture, colorName)),
  );

  return {
    framebuffer,
    size: sizeC,
    signature,
    markFrame() {
      transact(() => { frameC.value = frameC.value + 1; });
    },
    dispose() {
      if (observer !== undefined) observer.disconnect();
      if (msaaColor !== undefined) { msaaColor.destroy(); msaaColor = undefined; }
      if (depthTexture !== undefined) { depthTexture.destroy(); depthTexture = undefined; }
      try { ctx.unconfigure(); } catch { /* already gone */ }
    },
  };
}

function configure(
  ctx: GPUCanvasContext,
  device: GPUDevice,
  format: GPUTextureFormat,
  alphaMode: GPUCanvasAlphaMode,
): void {
  ctx.configure({
    device, format, alphaMode,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });
}

function currentSize(canvas: HTMLCanvasElement | OffscreenCanvas, dpr: number): { width: number; height: number } {
  if ((canvas as HTMLCanvasElement).clientWidth !== undefined) {
    const c = canvas as HTMLCanvasElement;
    return {
      width: Math.max(1, Math.floor((c.clientWidth || c.width) * dpr)),
      height: Math.max(1, Math.floor((c.clientHeight || c.height) * dpr)),
    };
  }
  const c = canvas as OffscreenCanvas;
  return { width: c.width, height: c.height };
}

function applySize(canvas: HTMLCanvasElement | OffscreenCanvas, w: number, h: number): void {
  if ((canvas as HTMLCanvasElement).clientWidth !== undefined) {
    const c = canvas as HTMLCanvasElement;
    c.width = w;
    c.height = h;
  } else {
    const c = canvas as OffscreenCanvas;
    c.width = w;
    c.height = h;
  }
}

function makeFramebuffer(
  ctx: GPUCanvasContext,
  signature: FramebufferSignature,
  size: { width: number; height: number },
  msaaColor: GPUTexture | undefined,
  depthTexture: GPUTexture | undefined,
  colorName: string,
): IFramebuffer {
  const swap = ctx.getCurrentTexture();
  const swapView = swap.createView();
  const msaa = signature.sampleCount > 1;
  // Pass attachment view: the multisample texture when MSAA is on,
  // otherwise the swap-chain texture directly.
  const passView = msaa
    ? (msaaColor ?? swap).createView()
    : swapView;
  const colors = HashMap.empty<string, GPUTextureView>().add(colorName, passView);
  const resolveColors = msaa
    ? HashMap.empty<string, GPUTextureView>().add(colorName, swapView)
    : undefined;
  // `colorTextures` exposes the *sampleable* texture (always the
  // swap-chain — multisample textures are not sampleable in the
  // normal sense).
  const colorTextures = HashMap.empty<string, GPUTexture>().add(colorName, swap);
  const fb: IFramebuffer = {
    signature,
    colors,
    colorTextures,
    ...(resolveColors !== undefined ? { resolveColors } : {}),
    width: size.width,
    height: size.height,
    ...(depthTexture !== undefined
      ? { depthStencil: depthTexture.createView(), depthStencilTexture: depthTexture }
      : {}),
  };
  return fb;
}

function depthAttachmentSig(format: GPUTextureFormat) {
  const hasDepth = /^depth/.test(format) || format === "depth24plus-stencil8" || format === "depth32float-stencil8";
  const hasStencil = /stencil/.test(format);
  return { format, hasDepth, hasStencil };
}
