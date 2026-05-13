# Multi-pipeline buckets — implementation plan

**Task 1 of 2** for the derived-modes design (`derived-modes.md`).
This task is the architectural change: bucket key drops
pipelineState; ROs with different declared pipeline state values
coexist in one bucket; encode iterates a fixed slot table; pipelines
are pre-warmed from the realized value set.

**Ships independently** of `derivedMode(...)` rules (Task 2). Delivers:

- Correct reactive cullmode (today silently broken — `psIdOf` in
  `heapScene.ts:2088` hashes by aval *identity*, so
  `cval.value = "front"` keeps the RO in the wrong bucket).
- The "20k cvals → 20k buckets" pathological case collapses to one
  bucket. Each `<Sg PipelineState={{ cull: cval(...) }}>` no longer
  fragments encoding.
- Runtime machinery (slot table, partition kernel, encode loop,
  pipeline cache) that Task 2 layers rules onto without modification.

User-facing API is unchanged: `<Sg PipelineState={{ cull: cval(...) }}>`
keeps working, but now correctly.

## Phase 1 — Pipeline cache + bitfield encoding

CPU-only foundation. Testable in isolation.

**New module:** `packages/rendering/src/runtime/pipelineCache/`

- `descriptor.ts` — `PipelineStateDescriptor` (canonicalized struct;
  field order, default elision, attachment list sort). Per-effect
  static fields (depthBias triplet, attachment formats, sample
  count) are recorded but not part of the derivable modeKey.
- `bitfield.ts` — `encodeModeKey(descriptor) → bigint` /
  `decodeModeKey(bigint) → descriptor`. Bitfield layout
  documented; ~24 bits for typical single-attachment scenes. Encoded
  key IS the cache key — no hash function, no collision check.
- `cache.ts` — `PipelineCache`:
  - `Map<bigint, GPURenderPipeline>` keyed on modeKey.
  - `precompile(descriptors[]): Promise<void>` →
    `createRenderPipelineAsync` per descriptor, awaits all.
  - `getOrCreateSync(modeKey, descriptor): GPURenderPipeline` for the
    runtime-mutation path (sync createRenderPipeline; driver compile
    in background).
- `manifest.ts` — IndexedDB persistence. Stores realized
  descriptors keyed on `(appVersion, wgslHash)`. Replayed on scene
  boot to warm the browser's internal WGSL compile cache before
  pipeline-link.

**Tests:** `tests/pipelineCache/` — bitfield round-trip, descriptor
canonicalization equality, concurrent precompile dedup, manifest
round-trip via `fake-indexeddb`.

## Phase 2 — Per-RO modeKey production (CPU-side)

The simple version: read aval values, pack to modeKey, dirty-track
via aval subscriptions. No GPU kernel yet.

**New file:** `packages/rendering/src/runtime/derivedModes/modeKeyCpu.ts`

- Per RO: subscribe to each `PipelineState` mode-axis aval (cullMode,
  blend, depthCompare, …). On mark, recompute that RO's modeKey,
  push the change into a per-bucket dirty list.
- Per frame: flush dirty list → upload changed `(roIndex, modeKey)`
  pairs to the GPU-side `modeKeys` buffer via writeBuffer. Bounded
  by changed-aval count.
- Initial population at addRO time: compute modeKey from current
  values, write into both the GPU buffer and a CPU-side mirror
  (the mirror feeds enumeration in Phase 3).

**Tests:** unit-level — reactive cval mutation marks one RO dirty,
flush uploads exactly one entry, modeKey reflects the new value.

## Phase 3 — Scene-side enumeration + pre-warm

Where the realized-value set becomes live pipelines.

**Edits to:** `heapScene.ts` scene construction + addRO path.

- Maintain `scene.slotTable: { modeKey: bigint, descriptor,
  indirectOffset, pipeline: GPURenderPipeline }[]` and a GPU-resident
  `modeKeyToSlot` lookup buffer.
- On addRO: compute the RO's modeKey from current aval values.
  - If `modeKey` is in `slotTable`: assign it; done.
  - If not: append a new slot, append a new descriptor; mark scene
    "needs pre-warm" for that descriptor.
- `scene.ready(): Promise<void>` → resolves when all current
  slotTable descriptors have linked pipelines. Replays the manifest
  first to warm the browser WGSL cache, then `precompile`s the
  enumerated descriptors.
- On reactive aval mutation introducing a never-before-seen value:
  same path as addRO's new-slot case, but at runtime: sync
  `createRenderPipeline` + append slot. The frame this happens on
  takes a GPU-queue stall.

**Tests:** scene-build integration — populate ROs with N distinct
cullMode values, assert slotTable has N entries, all pipelines ready
after `scene.ready()`. Mutate a cval to a fresh value, assert slot
table grows by 1.

## Phase 4 — Partition kernel

GPU compute pass that turns per-RO modeKeys into per-slot draw
metadata.

**New file:**
`packages/rendering/src/runtime/derivedModes/partition.ts`

- Inputs: `modeKeys[numROs]`, `modeKeyToSlot[]` lookup buffer.
- Per thread (one per record):
  - Read `slot = modeKeyToSlot[modeKey]` (binary search the lookup
    table, or hash-map style; table is small, ≤ ~100 slots).
  - Atomic-add `indexCount × instanceCount` to
    `indirectArgs[slot].indexCount`; atomic-add 1 to
    `indirectArgs[slot].instanceCount` (or accumulate
    `instanceCount` properly per the indirect-args layout).
  - Atomic-fetch-add into the slot's segment cursor; write the
    record's drawTable index at that position in the sorted
    drawTable.
- Output: per-bucket `indirectArgs[P_total]` and a permuted
  drawTable sorted by slot.

**Tests:** plant 1000 records with 3 distinct modeKeys, assert
exactly 3 non-empty indirectArgs entries and correctly permuted
drawTable.

## Phase 5 — Encode rewrite + bucket-key change

The integration moment.

**Edits to:** the per-bucket encode loop (`bucketEncoder.ts` or
inline in `heapRenderer.ts`).

```ts
for (const slot of scene.slotTable) {
  encoder.setPipeline(slot.pipeline);
  encoder.setBindGroup(0, bucket.bindGroup);
  if (slot.dynamic.blendConstant)  encoder.setBlendConstant(slot.dynamic.blendConstant);
  if (slot.dynamic.stencilRef !== undefined) encoder.setStencilReference(slot.dynamic.stencilRef);
  encoder.drawIndirect(bucket.indirectArgs, slot.indirectOffset);
}
```

**Edits to:** `heapEligibility.ts`, the bucket-keying code in
`heapScene.ts`:

- `psIdOf` removed. Bucket key becomes `(effect, textureSet)`.
- ROs with the same effect + textureSet but different declared
  PipelineState values now share a bucket and arena.

**Regression tests:**
- Every existing heap-renderer test re-runs green.
- New regression test: 20k ROs each with a fresh `cval("back")` for
  cullMode → one bucket, one slot.
- New correctness test: `cullCval.value = "front"` then
  `scene.frame()` → that RO renders with cull "front" (today: fails
  silently).

## Phase 6 — Runtime mutation polish

Hardening the runtime-mutation path from Phase 3 + 5.

- Slot table grow: reallocate `modeKeyToSlot` and `indirectArgs`
  buffers when slot count crosses a threshold (start small, pow2
  grow).
- Per-frame dirty-gate: if no aval marked AND no addRO/removeRO
  this frame, skip the modeKey upload + partition kernel entirely.
  Reuse last frame's slot contents. Static scenes pay zero.
- Surface diagnostics: scene exposes `scene.stats.slotCount` and
  warns on > 50 (state-change overhead becomes measurable).

**Tests:** dirty-gate skips kernel when no mark; grows correctly;
warn fires.

## Phase 7 — Demo + e2e tests

**Demo:** extend `examples/heap-demo-sg/` with:
- A reactive cullmode toggle (button flips `cullCval.value` between
  "back" / "front" / "none"). Visually verify the RO actually
  changes cullmode — this is the previously-broken case.
- A "20k cvals" stress test demonstrating the historical pathology
  collapses to one bucket.

**Real-GPU tests** (headed-Chromium): 10k / 30k records across
2 / 4 / 8 distinct mode combos. Assert:
- One bucket per effect (regardless of PipelineState variation).
- P_total drawIndirect calls per bucket; all visible pixels
  correct.
- Frame time within 5% of single-mode baseline.

## Order of merges

1. Phase 1 — pipeline cache + manifest. Exported subpath, no
   integration.
2. Phase 2 — CPU modeKey production. Tested in isolation against
   mocked drawHeader.
3. Phase 3 — scene-side enumeration + pre-warm. Pipelines exist;
   encode still uses old path.
4. Phase 4 — partition kernel. Output buffers built; encode still
   uses old path.
5. Phase 5 — encode + bucket-key change. **Integration moment.**
   Gated behind `enableMultiPipelineBuckets: false` for one
   release; flip to default-on after regression validation.
6. Phase 6 — runtime polish.
7. Phase 7 — demos + e2e tests.

## Estimate

| Phase | LOC | Days |
|---|---:|---:|
| 1 — pipeline cache + manifest | ~500 | 1.5 |
| 2 — CPU modeKey production | ~250 | 0.5 |
| 3 — scene enumeration + pre-warm | ~350 | 1 |
| 4 — partition kernel | ~350 | 1 |
| 5 — encode + bucket-key change | ~300 | 1 |
| 6 — runtime polish | ~200 | 0.5 |
| 7 — demo + e2e tests | ~800 | 1 |

Total: ~2750 LOC, ~6.5 days end-to-end. Phase 5 is the only
cross-cutting moment; everything else is additive.

## What this does NOT include

- `derivedMode(...)` API — that's Task 2 (`derived-mode-rules-plan.md`).
- IR analyzer for conservative output domain — Task 2.
- GPU-side mode-eval kernel computing modeKeys from uniform-driven
  rules — Task 2. Until then, modeKey production is "read current
  aval values" (CPU-side).
- The determinant-flip-cull motivating case — needs Task 2's rules
  to express `sign(det(u.ModelTrafo.upperLeft3x3))`.

Task 1 is the chassis; Task 2 mounts the engine.
