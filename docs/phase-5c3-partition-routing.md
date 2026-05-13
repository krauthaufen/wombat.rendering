# Phase 5c.3 — GPU-driven routing via a partition kernel

The remaining piece to make the derived-modes path "everything on
GPU". Today (0.9.33) the routing decision — "which slot does an RO
belong to" — is computed CPU-side by `ModeKeyTracker.recompute()`
running the rule's closure on each dirty RO. For ROs with reactive
inputs that fire often (e.g. per-frame trafo animation), this is the
remaining source of CPU work scaling with N.

Phase 5c.3 moves the routing decision GPU-side: a compute kernel
reads per-RO inputs from the arena, evaluates the rule, and writes
records directly into the right per-slot drawTable via atomic
scatter. CPU never reads back routing info; CPU encode iterates the
fixed slot table as today and the partition kernel's output drives
which records each slot draws.

## Architecture

**Bucket-level master record pool.** Today each slot owns its own
drawTable; records are written there at addRO time (CPU) and moved
between slots on reslot (CPU). With GPU routing, the bucket gains:

- `masterRecords: array<Record>` — bucket-level GPU buffer holding
  ALL ROs in the bucket in addRO order. Each entry has the same
  shape as today's drawTable record (firstEmit, drawIdx,
  indexStart, indexCount, instanceCount) plus a u32 `localSlot` and
  a u32 `modelRef` (arena offset of the RO's ModelTrafo).
- `numRecords: u32` — count of live records.

When CPU `addRO`s a new RO with a GPU rule, the record goes into
`masterRecords[numRecords++]`. Records DON'T go into any slot's
drawTable at addRO time.

**Per-frame partition kernel.** Each frame for any GPU-routed
bucket:

1. **Clear** per-slot record counters (`slot.recordCount` becomes a
   GPU atomic, reset to 0 by a small zero-clear pass).
2. **Partition** kernel: one thread per record. Reads the record's
   modelRef, computes the rule's modeKey (e.g. `flipCullByDet`),
   looks up the slot index via a `modeKeyToSlotIdx` GPU buffer (CPU
   uploads when slots are created), atomic-adds to
   `slot[slotIdx].recordCount`, and writes the record to
   `slot[slotIdx].drawTable[atomicValue]`.
3. **Existing scan** runs per slot (as today).

After the partition, each slot's drawTable contains exactly the
records the rule routed to it, in scatter order. CPU never moves
records.

**CPU encode** stays identical to 0.9.33: iterate `bucket.slots`,
`setPipeline + drawIndirect` per non-empty slot. `slot.recordCount`
is read from the GPU's atomic via the indirect buffer's first u32 —
which already happens today (the scan writes `indirect[0] =
totalEmit`).

## What CPU still does

- `addRO`: write the record into `masterRecords` (one writeBuffer
  per addRO, same as today's slot.drawTable write).
- `removeDraw`: swap-pop the record OUT of masterRecords.
- `modeKeyToSlotIdx` table: maintain CPU-side, upload on slot
  creation. For the determinant-flip rule, this table has at most
  3 entries (one per CullMode value).
- Slot creation: when a never-before-seen modeKey appears, create a
  new slot + add to the lookup table.
- `tracker.recompute()`: REMOVED for kernel-driven buckets. The
  bucket-level "modeKey changed" notion goes away — every record
  routes per-frame on GPU based on current input values.

## What CPU stops doing

- The dirty-mode-keys flush + reslot loop: not needed. Even when
  cullModeC flips, the kernel re-routes records GPU-side the next
  frame. No `dirtyModeKeyDrawIds`, no per-RO swap-pop+push.
- `ModeKeyTracker.modeKey` cache: useless for kernel-driven
  buckets. Tracker only needs to track input dependencies (for
  dirty-gating the partition kernel — skip the dispatch when no
  rule input marked).

## Implementation steps

### Step 1 — bucket-level master record pool

- Add `bucket.masterRecords?: GrowBuffer` and a CPU shadow.
- Add `bucket.masterRecordCount: number` and per-RO localSlot map.
- For ROs in GPU-routed buckets, `addRO` writes to the master, not
  to a specific slot's drawTable.
- `removeDraw` swap-pops master.

### Step 2 — modeKey → slotIdx GPU lookup

- Per bucket: a GPU buffer with at most MAX_SLOTS u64-modeKey + u32
  slotIdx triples. CPU populates when slots are created, uploads
  via writeBuffer.
- Linear-scan lookup in the partition kernel is fine for ≤8 slots.

### Step 3 — partition kernel

Three small compute passes per frame per GPU-routed bucket:

```wgsl
// 1. Clear per-slot record counters in indirect args.
@compute @workgroup_size(8) fn clearCounts(...)

// 2. Partition: per record, compute modeKey, scatter into slot.
@compute @workgroup_size(64) fn partition(...) {
  let r = masterRecords[gid.x];
  let key = computeRule(r.modelRef);   // rule-specific helper
  let slotIdx = modeKeyToSlot(key);    // linear scan
  let off = atomicAdd(&slot[slotIdx].count, 1);
  slot[slotIdx].drawTable[off] = r;
}
```

### Step 4 — wire into encodeComputePrep

Run partition BEFORE the existing scan. Mark all slots scanDirty
after partition. Scan reads from the freshly-scattered drawTable.

### Step 5 — drop CPU reslot for GPU-routed buckets

- `tracker.recompute()` not called for these buckets.
- `dirtyModeKeyDrawIds` doesn't include their RO drawIds.
- The mark-callback path stays only for non-GPU-routed buckets.

### Step 6 — gating + opt-in

- A bucket goes GPU-routed when its first RO has a GPU rule.
- Or: a per-scene flag that enables the path globally.
- Mixed-mode buckets (some ROs with rules, others without) are
  rejected for simplicity. Practically: a scene either has GPU
  rules or doesn't.

## Estimated effort

- Step 1: 2 hours (master pool + addRO/removeDraw routing).
- Step 2: 1 hour (lookup table + upload).
- Step 3: 3-4 hours (WGSL kernels + dispatcher).
- Step 4: 1 hour (wire dispatch).
- Step 5: 1 hour (skip CPU paths for GPU buckets).
- Step 6: 1 hour (opt-in flag, mixed-mode error).
- Tests: 2-3 hours (mock-GPU coverage of the partition kernel API,
  live-GPU verification of the demo).

Total: ~12-13 hours. Should fit in a focused 2-day session.

## What this delivers

- Per-frame cost of routing 20k ROs through a determinant-flip
  rule: one compute pass + one atomic-scatter + per-slot scans.
  CPU work: ~0.
- Today's cost: 20k tracker.recompute + 20k record swap-pop+push on
  any cullModeC flip. CPU does it; bounded but linear in N.
- The "everything on GPU" claim becomes literal: kernel decides
  routing, kernel computes drawIndirect args, encode just iterates
  fixed slots.

The current 0.9.33 state is the chassis (multi-slot buckets, per-
slot encode, pipeline cache). 5c.3 mounts the engine that drives
the routing on GPU.
