// ClearValues — per-attachment, name-keyed clear specification.
// Color clears are typed as V4f / V4i / V4ui to match the
// attachment's format; depth and stencil are plain numbers since
// WebGPU's GPURenderPassDepthStencilAttachment takes scalars.
// Attachments not named in `colors` keep their previous contents.

import type { HashMap } from "@aardworx/wombat.adaptive";
import type { V4f, V4i, V4ui } from "@aardworx/wombat.base";

export type ClearColor = V4f | V4i | V4ui;

export interface ClearValues {
  readonly colors?: HashMap<string, ClearColor>;
  readonly depth?: number;
  readonly stencil?: number;
}
