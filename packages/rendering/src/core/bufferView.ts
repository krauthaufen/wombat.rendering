// BufferView — a typed slice of an `IBuffer` used as a vertex
// attribute, instance attribute, or index source. The buffer
// itself can be a real GPU buffer or host-side memory; the
// runtime handles upload transparently.
//
// `format` carries the WebGPU vertex format when this view is
// bound as a vertex / instance attribute. `indexFormat` is set
// when the view is bound as an index buffer; in that case
// `format` is ignored. Both are optional so callers don't have
// to fill in the irrelevant one.

import type { IBuffer } from "./buffer.js";

export interface BufferView {
  readonly buffer: IBuffer;
  /** Byte offset into `buffer` where the view starts. */
  readonly offset: number;
  /** Number of elements (not bytes) in the view. */
  readonly count: number;
  /** Stride between consecutive elements, in bytes. */
  readonly stride: number;
  /** Vertex format — used when bound as a vertex / instance attribute. */
  readonly format: GPUVertexFormat | GPUIndexFormat;
  /** Index format — used when bound as an index buffer. Overrides `format`. */
  readonly indexFormat?: GPUIndexFormat;
}
