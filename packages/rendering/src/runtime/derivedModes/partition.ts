// Per-frame partition kernel — turns per-RO slot assignments into
// per-slot draw metadata (record lists + cumulative emit prefix sums
// + total indexCount × instanceCount per slot).
//
// Each bucket runs this kernel once per frame (dirty-gated). Output
// feeds one `drawIndirect` per slot in the encode loop. Records routed
// to the same slot share an indirect call; records in different slots
// land in separate `setPipeline + drawIndirect` calls.
//
// The CPU function `partitionCPU` is the algorithmic reference and
// the basis for unit tests. The WGSL kernel below mirrors the same
// algorithm using two compute passes (histogram + scatter), with an
// exclusive prefix scan on the histogram in between.
//
// Layout choices:
//
//   - Records are NOT physically re-sorted. Instead, the kernel writes
//     a permuted index list `slotRecords[]` (one u32 per record), plus
//     per-slot offsets `slotOffsets[]` and counts `slotCounts[]`. The
//     existing drawTable stays put — the VS prelude does one extra
//     read of `slotRecords[…]` to map from "drawIndex within slot"
//     back to "drawIndex in the full drawTable."
//   - `slotEmitPrefix[]` is the slot-local prefix-sum of
//     `record.indexCount * record.instanceCount`. The VS prelude
//     binary-searches THIS (per-slot) instead of the global table.
//   - `slotTotalEmit[]` is the dispatched vertex count for that
//     slot's drawIndirect.
//
// Buffers grow lazily; the partition wrapper resizes as slot counts
// or record counts change. v1 caps slots per bucket at MAX_SLOTS
// (256), exceeded slots throw with a friendly diagnostic.

/** v1 cap on distinct pipeline slots per bucket. */
export const MAX_SLOTS_PER_BUCKET = 256;

export interface PartitionInput {
  /** Per-record slot index (length = numRecords). */
  readonly slot:          ReadonlyArray<number>;
  /** Per-record (indexCount * instanceCount) — emit count contributed
   * to its slot's drawIndirect vertex total. */
  readonly emitCount:     ReadonlyArray<number>;
  /** Number of distinct slots used. Slots outside this range produce
   * an error. */
  readonly slotCount:     number;
}

export interface PartitionResult {
  /** Per-slot record count. Length = slotCount. */
  readonly slotCounts:     Uint32Array;
  /** Per-slot cumulative record-count prefix sum (length = slotCount + 1). */
  readonly slotOffsets:    Uint32Array;
  /** Per-slot total emit count (vertex count for the drawIndirect). */
  readonly slotTotalEmit:  Uint32Array;
  /** Permuted record indices, grouped by slot. Length = numRecords. */
  readonly slotRecords:    Uint32Array;
  /** Slot-local prefix sum of emitCounts, length = numRecords + slotCount.
   * For slot s, the prefix array is
   *   slotEmitPrefix[slotOffsets[s] + s ..= slotOffsets[s+1] + s]
   * (one extra entry per slot for the upper bound). */
  readonly slotEmitPrefix: Uint32Array;
}

/**
 * Pure-CPU partition. Reference implementation for the WGSL kernel
 * below. Order within a slot is stable wrt. input record order so
 * tests can predict the permuted layout deterministically.
 */
export function partitionCPU(input: PartitionInput): PartitionResult {
  const { slot, emitCount, slotCount } = input;
  const numRecords = slot.length;
  if (emitCount.length !== numRecords) {
    throw new Error("partition: slot.length != emitCount.length");
  }
  if (slotCount > MAX_SLOTS_PER_BUCKET) {
    throw new Error(`partition: slotCount ${slotCount} exceeds cap ${MAX_SLOTS_PER_BUCKET}`);
  }

  const slotCounts    = new Uint32Array(slotCount);
  const slotOffsets   = new Uint32Array(slotCount + 1);
  const slotTotalEmit = new Uint32Array(slotCount);
  const slotRecords   = new Uint32Array(numRecords);
  const slotEmitPrefix = new Uint32Array(numRecords + slotCount);

  // Pass 1: histogram + per-slot emit sum.
  for (let i = 0; i < numRecords; i++) {
    const s = slot[i]!;
    if (s < 0 || s >= slotCount) {
      throw new Error(`partition: record ${i} has out-of-range slot ${s} (slotCount=${slotCount})`);
    }
    slotCounts[s] = slotCounts[s]! + 1;
    slotTotalEmit[s] = slotTotalEmit[s]! + emitCount[i]!;
  }

  // Pass 2: exclusive prefix sum -> per-slot start offsets.
  let acc = 0;
  for (let s = 0; s < slotCount; s++) {
    slotOffsets[s] = acc;
    acc += slotCounts[s]!;
  }
  slotOffsets[slotCount] = acc;

  // Pass 3: scatter — second pass over records to place each at its
  // slot's current cursor, emitting the within-slot prefix sums.
  const cursors = new Uint32Array(slotCount);
  for (let i = 0; i < numRecords; i++) {
    const s = slot[i]!;
    const base    = slotOffsets[s]!;     // first slotRecords[] index for this slot
    const prefBase = base + s;           // first slotEmitPrefix[] index for this slot
    const k       = cursors[s]!;         // within-slot position
    slotRecords[base + k] = i;
    // Slot prefix sums are length (slotCounts[s] + 1): first entry
    // is 0, then cumulative; the last entry equals slotTotalEmit[s].
    if (k === 0) {
      slotEmitPrefix[prefBase] = 0;
    }
    slotEmitPrefix[prefBase + k + 1] = slotEmitPrefix[prefBase + k]! + emitCount[i]!;
    cursors[s] = k + 1;
  }

  return { slotCounts, slotOffsets, slotTotalEmit, slotRecords, slotEmitPrefix };
}

// ─── WGSL kernel ───────────────────────────────────────────────────────
//
// Mirrors `partitionCPU` in two compute passes:
//
//   Pass A (histogram):
//     - one thread per record
//     - read slot[i], emitCount[i]
//     - atomicAdd(slotCounts[s], 1)
//     - atomicAdd(slotTotalEmit[s], emitCount[i])
//
//   Pass B (scatter):
//     - one thread per record
//     - look up slot[i]
//     - atomicAdd(cursors[s], 1) -> within-slot position k
//     - slotRecords[slotOffsets[s] + k] = i
//     - slotEmitPrefix[slotOffsets[s] + s + k + 1] is computed by a
//       second scan pass (the within-slot prefix sum). v1 emits a
//       serial CPU-side scan for simplicity; v2 should fold this into
//       a per-slot warp-scan once buckets get large.
//
// Between the two passes, the CPU runs an exclusive scan on
// `slotCounts` to produce `slotOffsets` (small array, ≤ MAX_SLOTS).
//
// The within-slot prefix scan in pass B uses a non-atomic read of
// `slotEmitPrefix[prefBase + k]` — this is safe because the only
// thread that writes index `prefBase + k` is the one whose `cursors`
// fetchAdd returned `k`, and that thread completed before any thread
// whose fetchAdd returned `k+1` could read `prefBase + k`.
//
// (Wait — that's only true if we have a memory barrier across
// invocations, which WGSL doesn't guarantee inside one dispatch. In
// practice v1 punts: the kernel emits the histogram + slotRecords,
// and the host or a follow-up pass computes slotEmitPrefix. The
// kernel below reflects this: it produces slotCounts, slotOffsets
// (after host scan), slotTotalEmit, and slotRecords. The
// slotEmitPrefix scan is performed in a third compute pass on the
// already-scattered records.)

export const PARTITION_HISTOGRAM_WGSL = /* wgsl */ `
struct PartitionInputs {
  numRecords:   u32,
  slotCount:    u32,
  _pad0:        u32,
  _pad1:        u32,
};
@group(0) @binding(0) var<uniform>            uIn:           PartitionInputs;
@group(0) @binding(1) var<storage, read>      slot:          array<u32>;
@group(0) @binding(2) var<storage, read>      emitCount:     array<u32>;
@group(0) @binding(3) var<storage, read_write> slotCounts:    array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> slotTotalEmit: array<atomic<u32>>;

@compute @workgroup_size(64)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= uIn.numRecords) { return; }
  let s = slot[i];
  atomicAdd(&slotCounts[s], 1u);
  atomicAdd(&slotTotalEmit[s], emitCount[i]);
}
`;

export const PARTITION_SCATTER_WGSL = /* wgsl */ `
struct PartitionInputs {
  numRecords:   u32,
  slotCount:    u32,
  _pad0:        u32,
  _pad1:        u32,
};
@group(0) @binding(0) var<uniform>            uIn:         PartitionInputs;
@group(0) @binding(1) var<storage, read>      slot:        array<u32>;
@group(0) @binding(2) var<storage, read>      slotOffsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> cursors:     array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> slotRecords: array<u32>;

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= uIn.numRecords) { return; }
  let s    = slot[i];
  let base = slotOffsets[s];
  let k    = atomicAdd(&cursors[s], 1u);
  slotRecords[base + k] = i;
}
`;
