// renderTo — Aardvark `RenderTask.renderTo` ported.
//
// Usage:
//   const result = renderTo(ctx, scene, { size, signature, clear });
//   const colorAval: aval<ITexture> = result.color("color");
//   // bind colorAval as a texture on a downstream RenderObject
//
// `result.framebuffer` is an `AdaptiveResource<IFramebuffer>` that:
//   - Allocates the FBO on first acquire.
//   - On `compute(token)`: reads `size`, encodes a clear pass +
//     scene render into the active `RenderContext.encoder`, returns
//     the FBO. Encoding only runs when an outer encoder is active;
//     when read outside a frame the FBO is returned untouched.
//   - On last release: disposes the inner cache and frees the FBO.
//
// Each `result.color(name)` is a derived `aval<ITexture>` whose
// lifetime forwards to `framebuffer`, so consuming any attachment
// keeps the whole pipeline alive.

import {
  AdaptiveResource,
  ITexture,
  RenderContext,
  type ClearValues,
  type FramebufferSignature,
  type IFramebuffer,
  type RenderTree,
} from "@aardworx/wombat.rendering-core";
import {
  allocateFramebuffer,
  type FramebufferSize,
} from "@aardworx/wombat.rendering-resources";
import { beginPassDescriptor } from "@aardworx/wombat.rendering-commands";
import { HashMap, type AdaptiveToken, type aval } from "@aardworx/wombat.adaptive";
import { ScenePass } from "./scenePass.js";
import type { RuntimeContext } from "./renderTask.js";

export interface RenderToOptions {
  readonly size: aval<FramebufferSize>;
  readonly signature: FramebufferSignature;
  /** What to clear the FBO with each frame. Omit to skip the clear pass. */
  readonly clear?: ClearValues;
  readonly label?: string;
  /**
   * Extra GPU texture usage flags for every color attachment.
   * `RENDER_ATTACHMENT | TEXTURE_BINDING` are always set; pass
   * `COPY_SRC` here to enable readback, `STORAGE_BINDING` for
   * compute writes, etc.
   */
  readonly extraUsage?: GPUTextureUsageFlags;
}

export interface RenderToResult {
  /** The full framebuffer resource. Acquire/release for explicit lifetime control. */
  readonly framebuffer: AdaptiveResource<IFramebuffer>;
  /**
   * Returns an `aval<ITexture>` for the named color attachment.
   * The aval is itself an AdaptiveResource — acquiring it
   * activates the render-to pipeline.
   */
  color(name: string): AdaptiveResource<ITexture>;
  /**
   * Convenience for multi-target render passes: returns a
   * `HashMap<string, aval<ITexture>>` covering every color
   * attachment in the signature. Each entry shares the underlying
   * lifetime — acquiring any (or the whole map's entries) activates
   * the render-to pipeline.
   */
  colors(): HashMap<string, AdaptiveResource<ITexture>>;
  /** Returns the depth-stencil attachment as an `ITexture` (if the signature has one). */
  depthStencil(): AdaptiveResource<ITexture>;
}

class RenderToFramebuffer extends AdaptiveResource<IFramebuffer> {
  private scenePass: ScenePass | undefined;

  constructor(
    private readonly ctx: RuntimeContext,
    private readonly fbo: AdaptiveResource<IFramebuffer>,
    private readonly scene: RenderTree,
    private readonly clearValues: ClearValues | undefined,
    private readonly signature: FramebufferSignature,
  ) { super(); }

  protected override create(): void {
    this.fbo.acquire();
    this.scenePass = new ScenePass(
      this.ctx.device, this.signature, this.scene, this.ctx.compileEffect,
    );
  }

  protected override destroy(): void {
    this.scenePass?.dispose();
    this.scenePass = undefined;
    this.fbo.release();
  }

  override compute(token: AdaptiveToken): IFramebuffer {
    const fb = this.fbo.getValue(token);
    const enc = RenderContext.encoder;
    if (enc !== null && this.scenePass !== undefined) {
      const leaves = this.scenePass.resolve(token);
      if (leaves.length > 0 || this.clearValues !== undefined) {
        const pass = enc.beginRenderPass(beginPassDescriptor(fb, this.clearValues));
        for (const leaf of leaves) leaf.record(pass, token);
        pass.end();
      }
    }
    return fb;
  }
}

export function renderTo(
  ctx: RuntimeContext,
  scene: RenderTree,
  opts: RenderToOptions,
): RenderToResult {
  const fbo = allocateFramebuffer(ctx.device, opts.signature, opts.size, {
    ...(opts.label !== undefined ? { labelPrefix: opts.label } : {}),
    ...(opts.extraUsage !== undefined ? { extraUsage: opts.extraUsage } : {}),
  });
  const renderToFbo = new RenderToFramebuffer(
    ctx, fbo, scene,
    opts.clear,
    opts.signature,
  );
  return {
    framebuffer: renderToFbo,
    color(name) {
      return renderToFbo.derive<ITexture>(fb => {
        const tex = fb.colorTextures?.tryFind(name);
        if (tex === undefined) {
          throw new Error(`renderTo.color: framebuffer has no color attachment "${name}"`);
        }
        return ITexture.fromGPU(tex);
      });
    },
    colors() {
      let out = HashMap.empty<string, AdaptiveResource<ITexture>>();
      for (const [name] of opts.signature.colors) {
        out = out.add(name, this.color(name));
      }
      return out;
    },
    depthStencil() {
      return renderToFbo.derive<ITexture>(fb => {
        if (fb.depthStencilTexture === undefined) {
          throw new Error("renderTo.depthStencil: framebuffer has no depth-stencil attachment");
        }
        return ITexture.fromGPU(fb.depthStencilTexture);
      });
    },
  };
}
