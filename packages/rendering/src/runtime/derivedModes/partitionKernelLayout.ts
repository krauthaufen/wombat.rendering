// Partition kernel record layout — shared by the codegen + dispatcher.
//
// Each record in the master buffer is seven u32s:
//   [firstEmit, drawIdx, indexStart, indexCount, instanceCount, modelRef, comboId]
//
// `firstEmit` is reserved (GPU-written by the downstream scan); CPU
// writes 0. `modelRef` is the per-RO arena byte offset of the rule's
// input uniform (today: ModelTrafo). `comboId` selects which combo
// of axis-rules the partition kernel evaluates — a combo is a
// per-RO choice of (at most one) rule per mode axis. The kernel
// emits one `rule_<axis>_<axisRuleId>` fn per distinct rule and one
// `combo_<comboId>` fn that composes per-axis indices into a global
// slot index via mixed-radix encoding. For ROs without any rule on
// any axis, `comboId` selects the trivial combo (all axes fixed at
// baseDescriptor values).

export const PARTITION_RECORD_U32   = 7;
export const PARTITION_RECORD_BYTES = PARTITION_RECORD_U32 * 4;
