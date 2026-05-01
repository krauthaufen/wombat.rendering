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
  readonly depthStencil?: DepthStencilAttachmentSignature;
  readonly sampleCount: number;
}
