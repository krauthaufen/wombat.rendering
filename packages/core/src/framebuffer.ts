// IFramebuffer — a concrete render target. The signature describes
// the *shape*; the framebuffer pairs that shape with concrete
// texture views. Both color and depth/stencil views are addressed
// by the same names declared in the signature.

import type { HashMap } from "@aardworx/wombat.adaptive";
import type { FramebufferSignature } from "./framebufferSignature.js";

export interface IFramebuffer {
  readonly signature: FramebufferSignature;
  /** name → color attachment view, matching `signature.colors`. */
  readonly colors: HashMap<string, GPUTextureView>;
  readonly depthStencil?: GPUTextureView;
  /**
   * Underlying GPU textures for each color attachment. Populated
   * by `allocateFramebuffer`; used by `renderTo` to expose
   * results as `ITexture.fromGPU(...)` to downstream samplers.
   */
  readonly colorTextures?: HashMap<string, GPUTexture>;
  readonly depthStencilTexture?: GPUTexture;
  readonly width: number;
  readonly height: number;
}
