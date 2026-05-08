// Command — the per-frame primitive consumed by `runtime.compile`.
// Four kinds:
//   Render: encode a RenderTree into an FBO.
//   Clear:  clear individual named attachments of an FBO.
//   Copy:   buffer↔buffer / texture↔texture transfers.
//   Custom: escape hatch — receives the raw GPUCommandEncoder.
//
// Compute and Upload are NOT commands; they live inside
// AdaptiveResource update logic and run implicitly when a Render
// command needs a dirty resource.

import type { ClearValues } from "./clear.js";
import type { RenderTree } from "./renderTree.js";

export interface BufferCopyRange {
  readonly srcOffset: number;
  readonly dstOffset: number;
  readonly size: number;
}

export interface BufferCopy {
  readonly kind: "buffer";
  readonly src: GPUBuffer;
  readonly dst: GPUBuffer;
  readonly range?: BufferCopyRange;
}

export interface TextureCopy {
  readonly kind: "texture";
  readonly src: GPUImageCopyTexture;
  readonly dst: GPUImageCopyTexture;
  readonly size: GPUExtent3DStrict;
}

export type CopySpec = BufferCopy | TextureCopy;

/**
 * `Render` and `Clear` no longer carry their own framebuffer — the
 * task is created against a fixed `FramebufferSignature`, and the
 * concrete `IFramebuffer` is supplied at `run`/`encode` time. All
 * Render/Clear commands in a task share that framebuffer; multi-
 * framebuffer scenes compose multiple tasks at the call site.
 *
 * `Copy` and `Custom` are framebuffer-agnostic and unchanged.
 */
export type Command =
  | { readonly kind: "Render"; readonly tree: RenderTree }
  | { readonly kind: "Clear"; readonly values: ClearValues }
  | { readonly kind: "Copy"; readonly copy: CopySpec }
  | { readonly kind: "Custom"; readonly encode: (cmd: GPUCommandEncoder) => void };
