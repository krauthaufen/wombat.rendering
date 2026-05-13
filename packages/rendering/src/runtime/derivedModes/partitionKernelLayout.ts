// Partition kernel record layout — shared by the codegen + dispatcher.
//
// Each record in the master buffer is seven u32s:
//   [firstEmit, drawIdx, indexStart, indexCount, instanceCount, modelRef, ruleId]
//
// `firstEmit` is reserved (GPU-written by the downstream scan); CPU
// writes 0. `modelRef` is the per-RO arena byte offset of the rule's
// input uniform (today: ModelTrafo). `ruleId` selects which rule
// function the partition kernel dispatches to — allowing multiple
// ROs in the same bucket to carry different rules (the kernel emits
// one `rule_<axis>_<ruleId>` fn per distinct rule and switches on
// `r.ruleId`). For buckets with a single rule, every record carries
// ruleId=0.

export const PARTITION_RECORD_U32   = 7;
export const PARTITION_RECORD_BYTES = PARTITION_RECORD_U32 * 4;
