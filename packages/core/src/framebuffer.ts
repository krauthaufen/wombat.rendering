// IFramebuffer — a concrete render target. The signature describes
// the *shape*; the framebuffer pairs that shape with concrete
// texture views. Both color and depth/stencil views are addressed
// by the same names declared in the signature.

import type { HashMap } from "@aardworx/wombat.adaptive";
import type { FramebufferSignature } from "./framebufferSignature.js";

export interface IFramebuffer {
  readonly signature: FramebufferSignature;
  /**
   * name → color attachment view used inside the render pass.
   * For `sampleCount > 1` these are views into the multisample
   * textures; the resolve target views live in `resolveColors`.
   */
  readonly colors: HashMap<string, GPUTextureView>;
  readonly depthStencil?: GPUTextureView;
  /**
   * Underlying GPU textures suitable for *sampling* downstream.
   * - `sampleCount = 1`: same textures as `colors`.
   * - `sampleCount > 1`: the resolve textures (single-sample);
   *   the multisample textures are not sampleable.
   * Populated by `allocateFramebuffer`; used by `renderTo` to
   * expose results as `ITexture.fromGPU(...)`.
   */
  readonly colorTextures?: HashMap<string, GPUTexture>;
  /**
   * name → resolve target view, only present when
   * `signature.sampleCount > 1`. Wired into the render pass
   * descriptor as `resolveTarget` so the GPU resolves
   * multisample → single-sample at end of pass.
   */
  readonly resolveColors?: HashMap<string, GPUTextureView>;
  readonly depthStencilTexture?: GPUTexture;
  readonly width: number;
  readonly height: number;
}
