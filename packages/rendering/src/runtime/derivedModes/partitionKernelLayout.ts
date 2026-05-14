// Partition kernel record layout — shared by the codegen + dispatcher.
//
// Each record in the master buffer has a FIXED 6-u32 prefix and a
// per-bucket variable tail of K uniform refs:
//
//   [firstEmit, drawIdx, indexStart, indexCount, instanceCount, comboId,
//    ref0, ref1, …, ref{K-1}]
//
// `firstEmit` is reserved (GPU-written by the downstream scan); CPU
// writes 0. `comboId` selects which combo of axis-rules the partition
// kernel evaluates — a combo is a per-RO choice of (at most one) rule
// per mode axis. Each `refI` is the per-RO arena byte offset of the
// I-th uniform appearing across all rules in the bucket
// (`bucket.uniformOrder`). The kernel reads `r.refI` for whichever
// uniform a given rule body declares via `ReadInput("Uniform", name)`
// — the codegen translates name → refI lookup.

export const PARTITION_RECORD_PREFIX_U32 = 6;

export function partitionRecordU32(numUniforms: number): number {
  return PARTITION_RECORD_PREFIX_U32 + numUniforms;
}
export function partitionRecordBytes(numUniforms: number): number {
  return partitionRecordU32(numUniforms) * 4;
}
