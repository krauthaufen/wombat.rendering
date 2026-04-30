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
import type { IFramebuffer } from "./framebuffer.js";
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

export type Command =
  | { readonly kind: "Render"; readonly output: IFramebuffer; readonly tree: RenderTree }
  | { readonly kind: "Clear"; readonly output: IFramebuffer; readonly values: ClearValues }
  | { readonly kind: "Copy"; readonly copy: CopySpec }
  | { readonly kind: "Custom"; readonly encode: (cmd: GPUCommandEncoder) => void };
