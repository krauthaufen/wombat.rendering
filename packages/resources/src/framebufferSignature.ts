// Helpers for constructing a `FramebufferSignature` from plain
// records. Pure value construction — no GPU calls — but lives in
// the resources package because it pairs with `allocateFramebuffer`.

import { HashMap } from "@aardworx/wombat.adaptive";
import type {
  DepthStencilAttachmentSignature,
  FramebufferSignature,
} from "@aardworx/wombat.rendering-core";

export interface FramebufferSignatureSpec {
  readonly colors: Record<string, GPUTextureFormat>;
  readonly depthStencil?: { readonly format: GPUTextureFormat };
  readonly sampleCount?: number;
}

export function createFramebufferSignature(
  spec: FramebufferSignatureSpec,
): FramebufferSignature {
  let colors = HashMap.empty<string, GPUTextureFormat>();
  for (const [name, format] of Object.entries(spec.colors)) {
    colors = colors.add(name, format);
  }
  const sig: FramebufferSignature = {
    colors,
    sampleCount: spec.sampleCount ?? 1,
    ...(spec.depthStencil ? { depthStencil: depthStencilFor(spec.depthStencil.format) } : {}),
  };
  return sig;
}

function depthStencilFor(format: GPUTextureFormat): DepthStencilAttachmentSignature {
  const hasDepth = /^depth/.test(format) || format === "depth24plus-stencil8" || format === "depth32float-stencil8";
  const hasStencil = /stencil/.test(format);
  return { format, hasDepth, hasStencil };
}
