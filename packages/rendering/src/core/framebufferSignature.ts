// FramebufferSignature — the typed shape of a render target.
// Attachments are addressed by name (never by index). The runtime
// maps names → WebGPU attachment slots using a shader's
// ProgramInterface, so users only ever talk about names like
// "albedo" / "normal" / "objectId", not location indices.

import type { HashMap } from "@aardworx/wombat.adaptive";

export interface DepthStencilAttachmentSignature {
  readonly format: GPUTextureFormat;
  readonly hasDepth: boolean;
  readonly hasStencil: boolean;
}

export interface FramebufferSignature {
  readonly colors: HashMap<string, GPUTextureFormat>;
  /**
   * Canonical name → location ordering for the color attachments.
   * `colorNames[i]` names the attachment at WebGPU color-attachment
   * slot `i` (and the fragment-shader `@location(i)`). This is the
   * authoritative ordering used by the render-pass descriptor, the
   * shader's fragment-output linker, and pipeline color-target
   * indexing — `colors` (a HashMap) provides only name → format
   * lookup and has no inherent order.
   */
  readonly colorNames: readonly string[];
  readonly depthStencil?: DepthStencilAttachmentSignature;
  readonly sampleCount: number;
}
