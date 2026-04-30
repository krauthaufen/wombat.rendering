// copy(cmd, spec) — encode a buffer↔buffer or texture↔texture
// transfer described by a `CopySpec`.

import type { CopySpec } from "@aardworx/wombat.rendering-core";

export function copy(cmd: GPUCommandEncoder, spec: CopySpec): void {
  if (spec.kind === "buffer") {
    const range = spec.range ?? { srcOffset: 0, dstOffset: 0, size: spec.src.size };
    cmd.copyBufferToBuffer(spec.src, range.srcOffset, spec.dst, range.dstOffset, range.size);
  } else {
    cmd.copyTextureToTexture(spec.src, spec.dst, spec.size);
  }
}
