# Multi-pipeline buckets — Phase 5 integration notes

Captured at the boundary between the standalone Phase 1–4 modules and
the heapScene wiring. Foundation findings from reading
`packages/rendering/src/runtime/heapScene.ts` at commit `7203e8d`.

## What's already shipped (Phases 1–4)

Standalone modules with passing tests. Each one is a pure data
structure or algorithm; no heapScene dependencies.

- `runtime/pipelineCache/{descriptor,bitfield,cache}.ts` — canonical
  descriptor + bitfield-encoded modeKey (u64 bigint) + async/sync
  pipeline cache. 12 tests.
- `runtime/derivedModes/modeKeyCpu.ts` — `ModeKeyTracker` subscribes
  to a `PipelineState`'s mode-axis avals, emits a stable modeKey from
  current aval *values* (not identities). 6 tests.
- `runtime/derivedModes/slotTable.ts` — per-bucket `SlotTable`
  mapping modeKey → slot. Refcounts ROs per slot; `ready()` awaits
  pre-warm. 7 tests.
- `runtime/derivedModes/partition.ts` — `partitionCPU` reference for
  the histogram + scatter partition algorithm + WGSL kernel
  skeletons. 6 tests.

31 new passing tests; total runtime suite still green.

## The hard part: heapScene's flat Bucket struct

`interface Bucket` (`heapScene.ts:1025`) holds, **per bucket**, what
the new design needs to make **per slot**:

```
pipeline:            GPURenderPipeline       ← per slot
drawHeap:            DrawHeap                ← per bucket (shared)
drawTableBuf:        GrowBuffer              ← per slot (records sort by slot)
firstDrawInTileBuf:  GrowBuffer              ← per slot (binary-search-fold operates on a contiguous range)
indirectBuf:         GPUBuffer               ← per slot (one drawIndirect per slot)
recordCount:         number                  ← per slot
slotToRecord/recordToSlot: number[]          ← per slot
totalEmitEstimate:   number                  ← per slot
scanDirty:           boolean                 ← per slot
```

And, per bucket (unchanged):

```
label, textures, layout, bindGroup, drawHeap, drawHeaderStaging,
localPosRefs / localNorRefs / localEntries / localToDrawId /
localPerDrawAvals / localPerDrawRefs / localLayoutIds, drawSlots, dirty,
isAtlasBucket, localAtlasReleases, localAtlasTextures, localAtlasArrIdx
```

The per-slot state is non-trivial — drawTable + scan + indirect +
binary-search-fold machinery is the core of the megacall path.

## Recommended Phase 5 staging

The integration shouldn't try to flip everything at once. Three
sub-phases inside Phase 5, each independently mergeable:

### 5a. Bucket lifecycle: hoist per-slot state into a struct

Add `interface BucketSlot` holding the per-slot fields listed above.
`Bucket` gets a new `slots: BucketSlot[]` field. **No behavior
change** — start with `slots = [<one BucketSlot wrapping today's
flat fields>]`. Every code site that reads `bucket.pipeline`,
`bucket.drawTableBuf`, etc., is rewritten to read
`bucket.slots[0].pipeline` etc.

This is a mechanical rename / restructure. Boring; large diff;
zero risk to render output.

### 5b. SlotTable lives on each Bucket; psKey still in bucket key

Each Bucket gets a `slotTable: SlotTable` (Phase 3's). Each
`BucketSlot` corresponds to one `SlotTableEntry`. addRO calls
`bucket.slotTable.addRO(descriptor)` — still produces exactly one
slot per bucket because we haven't dropped psKey from the bucket
key yet. Bucket lookup remains `findOrCreateBucket(effect, textures,
ps)` via the existing psKey.

Why this in-between step: it proves the SlotTable + ModeKeyTracker
integration end-to-end without touching the bucket-key change.
Reactive cvals still produce wrong-bucket results (the original bug),
but the new machinery is plumbed and stress-testable.

### 5c. Drop psKey from bucket key; allow multi-slot buckets

The actual unlock. Bucket key becomes `(family#{schemaId}, textures)`.
addRO assigns the RO to the correct slot via the SlotTable; records
with new descriptors append new slots. Encode loops over
`bucket.slots`, emitting `setPipeline + drawIndirect` per non-empty
slot.

Records routed to different slots **must be in their own
drawTable** (the binary-search-fold expects a contiguous range per
drawIndirect). Two options:

1. **Per-slot drawTable** — each `BucketSlot` keeps its own
   drawTable + scan buffers + indirect (the shape 5a hoisted into
   the struct). addRO appends to the right slot's table.
   removeDraw + addDraw on rebind. Memory cost: small fixed
   overhead per slot.
2. **Unified drawTable, partition-sorted** — one drawTable per
   bucket, the Phase 4 partition kernel scatters records into
   slot ranges each frame; each slot's drawIndirect reads its sub-
   range. Memory cost: one extra permuted-index buffer; runtime
   cost: one extra compute pass.

**Recommend option 1** — it's mechanically simpler (each slot is a
mini-bucket sharing the parent's arena + bindGroup), no extra
per-frame compute, no new WGSL kernels needed in heapScene itself.
The Phase 4 partition kernel becomes useful for Task 2 (when rules
drive modeKey changes at GPU rate) but is over-engineered for
Task 1's CPU-driven modeKey flow.

### Decision points to lock down before starting 5c

1. **Per-slot vs unified drawTable** — recommend per-slot (above).
2. **When does an RO's modeKey change drive a bucket rebind?**
   Each frame? Just on mark? `ModeKeyTracker.onDirty` fires
   synchronously on mark; the heap scene's frame-prep step
   (currently `encodeComputePrep` or a sibling) is the right
   place to flush rebinds.
3. **scene.ready() shape.** Probably wraps `await all
   bucket.slotTable.ready()` across the live bucket set. Each
   `buildHeapScene` returns a scene with `ready: () => Promise<void>`.
4. **Family-merge code (§6, dead).** Currently still threaded
   through `findOrCreateBucket` via `familyFor(effect)`. With
   multi-pipeline buckets, the family-merge collapse becomes
   irrelevant for the cull/blend/depth axis (modes vary per slot,
   not per family). But the family machinery also handles
   shader-module unification. Keep family-merge code intact;
   bucket key still uses `family#schemaId` (the effect→family
   indirection) but drops the psKey suffix.
5. **Pipeline builder wiring.** The current
   `findOrCreateBucket` builds the pipeline inline; with SlotTable,
   the builder is called by `SlotTable.addRO`'s precompile path.
   Need to factor out the pipeline-build closure (`effect, layout,
   bindGroupLayout, framebufferSignature, descriptor → GPURenderPipeline`)
   so it's stored on the Bucket and reused per slot.

## Testing strategy for Phase 5

- **5a regression**: existing 100+ heap tests must remain green.
  Pure restructure; if any test fails, the hoist is wrong.
- **5b**: add a test that mutates a `cval<CullMode>` and asserts
  the rendered cullmode actually flips (today's broken case). This
  test should FAIL pre-5b and PASS post-5b — even though psKey is
  still in the bucket key, the SlotTable picks the new descriptor
  on rebind. (Actually wait — if psKey is still in the bucket key,
  the cval flip moves the RO to a *different* bucket. That's the
  same broken behaviour as today, just routed differently. So 5b
  doesn't actually fix the reactive case; only 5c does. Keep this
  test in mind; it lands with 5c.)
- **5c regression**: same heap tests green. New test: 20k ROs with
  20k distinct cvals for cullMode all valued "back" → ONE bucket,
  one slot.
- **5c correctness**: mixed-cull demo + reactive flip both render
  correctly.

## What I'd start with on Day 1 of the next session

1. Open `heapScene.ts` to line 1025. Define `BucketSlot` interface
   holding the per-slot fields.
2. Migrate the Bucket interface to hold `slots: BucketSlot[]` (start
   with length 1 always).
3. Grep every read of the per-slot fields and rewrite. Most are in
   `addDraw`, `removeDraw`, `encodeComputePrep`, `encodeIntoPass`,
   `flushHeaders`, `rebuildBindGroups`.
4. Run the full test suite; fix any breakage. Commit 5a.
5. Repeat the discovery pass to ensure nothing reads the flat
   fields directly anymore. (TypeScript helps: if the flat fields
   are deleted from `Bucket`, the compiler enumerates every read.)

The first commit (5a) is mechanical and the largest in line count.
After it lands, 5b and 5c each move forward in smaller surgical
edits.
