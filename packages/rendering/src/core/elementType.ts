// ElementType — the data type of a single element in a BufferView.
//
// Mirrors Aardvark.Rendering's `BufferView.ElementType : System.Type`
// pattern: structural information about the element kept separate
// from the per-frame buffer content. The runtime maps `ElementType`
// to a `GPUVertexFormat`/`GPUIndexFormat` and a byte size at
// pipeline-layout time.

export type ElementType =
  // scalar
  | "f32" | "u32" | "i32"
  | "u16" | "i16"
  | "u8"  | "i8"
  // vec — float
  | "v2f" | "v3f" | "v4f"
  // vec — int
  | "v2i" | "v3i" | "v4i"
  // vec — uint
  | "v2u" | "v3u" | "v4u"
  // matrix (float, square)
  | "m22f" | "m33f" | "m44f"
  // packed colour
  | "c4b" | "c3b";

export const ElementType = {
  /** Size of one element in bytes. */
  byteSize(t: ElementType): number {
    switch (t) {
      case "f32": case "u32": case "i32": return 4;
      case "u16": case "i16": return 2;
      case "u8":  case "i8":  return 1;
      case "v2f": case "v2i": case "v2u": return 8;
      case "v3f": case "v3i": case "v3u": return 12;
      case "v4f": case "v4i": case "v4u": return 16;
      case "m22f": return 16;
      case "m33f": return 36;
      case "m44f": return 64;
      case "c4b": return 4;
      case "c3b": return 3;
    }
  },

  /**
   * GPUVertexFormat for use as a vertex / instance attribute.
   * `normalized=true` only affects integer types — picks the `unorm`
   * / `snorm` form. Matrix types are not directly supported as
   * vertex formats; the auto-instancing pass splits them into
   * column attributes upstream.
   */
  toVertexFormat(t: ElementType, normalized = false): GPUVertexFormat {
    switch (t) {
      case "f32": return "float32";
      case "u32": return "uint32";
      case "i32": return "sint32";
      case "v2f": return "float32x2";
      case "v3f": return "float32x3";
      case "v4f": return "float32x4";
      case "v2i": return "sint32x2";
      case "v3i": return "sint32x3";
      case "v4i": return "sint32x4";
      case "v2u": return "uint32x2";
      case "v3u": return "uint32x3";
      case "v4u": return "uint32x4";
      case "u16": return normalized ? "unorm16x2" as GPUVertexFormat : "uint16";
      case "i16": return normalized ? "snorm16x2" as GPUVertexFormat : "sint16";
      case "u8":  return normalized ? "unorm8x4"  as GPUVertexFormat : "uint8";
      case "i8":  return normalized ? "snorm8x4"  as GPUVertexFormat : "sint8";
      case "c4b": return normalized ? "unorm8x4" : "uint8x4";
      case "c3b":
        // No 3-component byte format in WebGPU; promoted to x4 with
        // a dummy fourth byte by the upload path.
        throw new Error(`ElementType.toVertexFormat: c3b cannot be a vertex format directly; pad to c4b`);
      case "m22f": case "m33f": case "m44f":
        throw new Error(`ElementType.toVertexFormat: ${t} is a matrix; split into column attributes first`);
    }
  },

  /** Index format. Throws unless the type is `u16` or `u32`. */
  toIndexFormat(t: ElementType): GPUIndexFormat {
    if (t === "u16") return "uint16";
    if (t === "u32") return "uint32";
    throw new Error(`ElementType.toIndexFormat: ${t} is not a valid index type (use u16 or u32)`);
  },
} as const;
