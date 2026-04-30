// BufferView — a typed slice of an `IBuffer` used as a vertex
// attribute, instance attribute, or index source. The buffer
// itself can be a real GPU buffer or host-side memory; the
// runtime handles upload transparently.

import type { IBuffer } from "./buffer.js";

export interface BufferView {
  readonly buffer: IBuffer;
  /** Byte offset into `buffer` where the view starts. */
  readonly offset: number;
  /** Number of elements (not bytes) in the view. */
  readonly count: number;
  /** Stride between consecutive elements, in bytes. */
  readonly stride: number;
  /** Per-element format (vertex format for attributes, index format for indices). */
  readonly format: GPUVertexFormat | GPUIndexFormat;
}
