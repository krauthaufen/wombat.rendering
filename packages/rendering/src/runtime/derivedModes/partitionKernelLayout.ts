// Partition kernel record layout — shared by the codegen + dispatcher.
//
// Each record in the master buffer is six u32s:
//   [firstEmit, drawIdx, indexStart, indexCount, instanceCount, modelRef]
//
// `firstEmit` is reserved (GPU-written by the downstream scan); CPU
// writes 0. `modelRef` is the per-RO arena byte offset of the rule's
// input uniform (today: ModelTrafo — generalizes when rules read
// multiple uniforms).

export const PARTITION_RECORD_U32   = 6;
export const PARTITION_RECORD_BYTES = PARTITION_RECORD_U32 * 4;
