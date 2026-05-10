// Slot-ID encoding for the §7 uber kernel.
//
// One u32 namespace covers both the Constituents heap (df32 trafos
// uploaded from CPU) and the Intermediates heap (df32 outputs of
// layer-1 recipes). The high bit selects the heap; the low 31 bits are
// the linear index within that heap. Each slot holds one mat4 in df32
// = 16 vec2<f32> = 128 bytes.
//
// Uber-kernel side: a single helper reads from either heap based on
// the flag — see `read_mat4_df` in the WGSL source.

/** Branded u32 — top bit selects heap, low 31 bits are slot index. */
export type SlotId = number & { readonly __slotId: unique symbol };

const FLAG_INTERMEDIATE = 0x80000000;

export const SlotId = {
  /** Encode a Constituents-heap slot. */
  constituent(idx: number): SlotId {
    return (idx & 0x7FFFFFFF) as SlotId;
  },
  /** Encode an Intermediates-heap slot. */
  intermediate(idx: number): SlotId {
    return ((idx & 0x7FFFFFFF) | FLAG_INTERMEDIATE) >>> 0 as SlotId;
  },
  isIntermediate(s: SlotId): boolean {
    return (s & FLAG_INTERMEDIATE) !== 0;
  },
  index(s: SlotId): number {
    return s & 0x7FFFFFFF;
  },
};

/** Bytes per df32 mat4 slot: 16 entries × 2 floats × 4 bytes. */
export const DF32_MAT4_BYTES = 128;
