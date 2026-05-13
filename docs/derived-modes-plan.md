# Derived modes — implementation plan

Companion to `derived-modes.md` (the design sketch). This is the
build sequence: phases, files touched, success criteria. Roughly
modelled on how §7 v2 was staged.

## Scope summary

- Pipeline state (cull, blend, depth compare, depthBias triplet, …)
  becomes a per-RO **derived value**, evaluated each frame by a
  §7-shaped compute pass, partitioning the draw stream at encode.
- WebGPU 1.0 has no pipeline blob; we persist **descriptor manifests**
  and rely on the browser's WGSL compile cache + async link.
- Bucket key drops pipelineState entirely → `(effect, textureSet)`.

## Phase 0 — Pipeline descriptor cache (CPU only, no GPU work yet)

Foundation that everything else stands on. Ship-able on its own and
unit-testable without any compute / encode changes.

**New module:** `packages/rendering/src/runtime/pipelineCache/`

- `descriptor.ts` — `PipelineStateDescriptor` (normalized struct: cull,
  frontFace, topology, stripIndexFormat, depthCompare, depthWriteEnabled,
  blend factors+ops per attachment, color write mask, stencil ops,
  sample count, alphaToCoverage, depthBias triplet). Canonicalization:
  field order, default-elision, attachment list sort.
- `hash.ts` — FNV-64 over the canonicalized descriptor → `bigint` key.
  Plus a `descriptorEquals(a, b)` for collision-safe lookup.
- `cache.ts` — `PipelineCache`:
  - `Map<bigint, { descriptor, status: "pending" | "ready" | "error", pipeline?: GPURenderPipeline }>`
  - `getOrCreate(device, descriptor, layoutBundle): Promise<GPURenderPipeline>`
  - Internal `createRenderPipelineAsync` with bigint-key dedup so two
    concurrent requests for the same descriptor share one compile.
  - Synchronous `lookup(hash)`: returns ready pipeline or `null`. Used
    by encode hot path.
- `manifest.ts` — IndexedDB persistence. Stores
  `{ appVersion, wgslHashes[], descriptors[] }`. On boot, replays
  descriptors into the cache via async createPipeline. Records new
  descriptors as they're observed.

**Tests:** `tests/pipelineCache/` — hash determinism, descriptor
canonicalization, dedup on concurrent getOrCreate, manifest round-trip
through fake-indexeddb.

**No integration yet** — this module sits parallel to the heap renderer.
Done when: hashing is stable, async dedup works, manifest persists.

## Phase 1 — Rule authoring API & build-time marker

Mirrors §7's `derivedUniform` ergonomics so users don't learn a new shape.

**Runtime (wombat.rendering):**

- `packages/rendering/src/runtime/derivedModes/marker.ts`:
  - `derivedMode<Axis extends ModeAxis>(axis, build, leafTypes?)` →
    `DerivedModeRule { axis, ir, hash, __derivedModeRule: true }`.
  - `ModeAxis` enum: `"cull" | "frontFace" | "topology" | "depthCompare"
    | "depthWriteEnabled" | "blend" | "colorWriteMask" | "stencilOps"
    | "depthBias"`.
  - `build` is a closure `(u, declared) => ModeValue<Axis>`. The proxy
    minting `u.<Name>` reuses §7's tracing mechanism wholesale —
    `DerivedExpr` and the existing leaf builders.
  - Output type per axis is a small enum (or struct for `depthBias`).
    Validation in the marker rejects out-of-range values.
- `isDerivedModeRule(x)` brand check (mirrors `isDerivedRule`).

**Build-time (wombat.shader-vite):**

- Extend `inline.ts`'s `derivedUniform` handler to also recognize
  `derivedMode(...)` calls and apply the same `UniformScope`-derived
  `leafTypes` hint injection. Same `declare module
  "@aardworx/wombat.shader/uniforms"` augmentations are read.
- The `declared` parameter (second arg of the closure) is typed by the
  marker — no augmentation needed.
- Tests in `wombat.shader/tests/inline-derived-mode.test.ts`.

Done when: a user can write
`derivedMode("cull", (u, d) => u.Side === 0 ? "back" : "front")` in a
heap-demo file and get a properly typed `DerivedModeRule` value at
runtime.

## Phase 2 — Mode evaluation compute pass

Reuses §7's kernel scaffolding but emits to a different output buffer.

**New module:** `packages/rendering/src/runtime/derivedModes/dispatch.ts`.

- `DerivedModesRegistry`: per-axis maps `DerivedModeRule.hash →
  { rule, ruleId }`. Composed with the §7 uniform registry — they share
  the same flatten / inputsOf machinery.
- Kernel layout: per RO, write to `modeHashOut[ro_ix]: u32` (lo) +
  `u32` (hi) of the bigint hash, plus a side `modeDescriptorScratch`
  buffer holding the resolved descriptor fields (only for axes that
  need pre-image inspection at partition time — typically just the
  depthBias triplet, since other axes are small-enum and the hash is
  enough).
- Codegen: extend `codegen.ts`'s expression printer to support the
  enum outputs (small u32 literal). For `depthBias`, the rule's output
  is a 3-tuple — encoded into the scratch buffer; hash incorporates
  all three.
- Dispatch: one extra compute call per frame, after §7's pre-pass.
  Same threading (one thread per (RO × mode axis)).

Done when: a unit test installs a `derivedMode` rule and inspects the
output buffer for correct per-RO hashes.

## Phase 3 — Counting-sort partition pass

Pure GPU pass — takes the per-RO `modeHashOut`, produces a segmented
drawTable + segment metadata.

**New module:** `packages/rendering/src/runtime/derivedModes/partition.ts`.

Two-pass counting-sort over the per-bucket records:

1. **Compact distinct hashes:** workgroup-local hash table + a serial
   merge into a per-bucket "distinct hashes this frame" array.
   K = #distinct ≤ small cap (start with 64 per bucket; resize on
   overflow, log a warning).
2. **Histogram:** one thread per record, atomic-add into a per-hash
   counter. Exclusive scan over counters → segment offsets.
3. **Scatter:** one thread per record, atomic-fetch-add for its
   segment offset, writes its original record index into the sorted
   table at that position.

Output:
- `sortedDrawTable: { firstEmit, drawIdx, indexStart, indexCount, instanceCount }[]`
  (same shape as today's drawTable, just permuted).
- `segments: { hashLo, hashHi, firstRecord, count }[]`.

Done when: a test with N records and K manual hashes produces the
expected segmented table.

## Phase 4 — Segmented encode

Encode loop rewrite. The hot path.

**Edits to** `packages/rendering/src/runtime/heapRenderer.ts` (or
wherever the bucket-encode loop lives — likely
`bucketEncoder.ts`):

```ts
for (const bucket of buckets) {
  for (let i = 0; i < bucket.segmentCount; i++) {
    const seg = bucket.segments[i];
    const pipeline = pipelineCache.lookup(seg.hashLo, seg.hashHi);
    if (!pipeline) {
      pipelineCache.kickAsync(seg.descriptor);
      // Two policies; pick at scene construction:
      // (a) skip this segment this frame (rule output dropped a frame),
      // (b) fall back to bucket.staticPipeline (the SG-declared default)
      //     for these records — they render with the default mode, not
      //     the rule's mode, for one frame.
      continue; // (a)
    }
    encoder.setPipeline(pipeline);
    encoder.setBindGroup(0, bucket.bindGroup);
    // dynamic state (extracted from descriptor at compile time):
    if (seg.hasBlendConstant) encoder.setBlendConstant(seg.blendConstant);
    if (seg.hasStencilRef)    encoder.setStencilReference(seg.stencilRef);
    encoder.drawIndirect(bucket.indirectBuf, seg.indirectOffset);
  }
}
```

The `drawIndirect` args buffer needs K entries per bucket (built by
the partition pass). Reallocates if K grows.

Done when: a heap-demo bucket with two RO-driven cull modes emits two
drawIndirects, both rendering correctly.

## Phase 5 — Drop pipelineState from bucket key

The unlock that makes mode switches free.

**Edits to** `heapEligibility.ts`, `bucketKeying.ts`:

- Bucket key becomes `(effect, textureSet)`. PipelineState comparison
  removed.
- An RO with no `derivedMode` rules gets a trivial rule per axis that
  evaluates to the SG-declared value — so the partition pass still
  produces one segment per bucket, with K=1.
- Migration: existing tests / demos with mixed pipeline state across
  the same effect collapse into one bucket. Check that no shader
  assumes pipelineState in the bucket key.

Done when: heap-demo with mixed-cull ROs (some `cull: "back"`, some
`cull: "none"`) renders correctly with one bucket per effect.

## Phase 6 — Persistence + pre-warm API

The "expensive only on first ever visit" piece.

**Edits to** `pipelineCache/manifest.ts` and the public scene API.

- `pipelineCache.recordDescriptor(d)` — fires on any cache miss
  (including async-pending hits). Throttled writes to IndexedDB
  (debounce 1 s).
- On scene construction: `pipelineCache.warmFromManifest(appKey)` →
  replays persisted descriptors via `getOrCreate`. Returns a promise;
  scene can choose to wait (loading screen) or proceed (first frames
  may use fallback).
- Public:
  ```ts
  preWarmPipelineCache(scene, descriptors: PipelineStateDescriptor[]): Promise<void>
  ```
  Lets app code declare its hot combos explicitly, independent of
  persistence.
- Manifest keyed on `(appVersion, wgslHash)` — bumping the shader
  invalidates the entry naturally, the next session re-discovers and
  re-persists.

Done when: a demo with a fresh IndexedDB sees first-frame pipeline
misses; a second load with warm manifest renders frame 0 with K
pipelines ready.

## Phase 7 — Demo + integration tests

**Demo:** extend `examples/heap-demo-sg/` with a "mode zoo" — a few
ROs whose cull mode is uniform-driven, a layer slider that changes
depthBias per group. Visual confirmation that mixed modes share a
bucket and arena.

**Real-GPU tests:** `tests/heap-derivedMode-e2e.test.ts` driven
through the headed-Chromium harness. Sweep at 10K / 30K records,
2 / 4 / 8 distinct mode combos. Assert: one bucket per effect, K
drawIndirects, no per-mode VRAM duplication, frame time within 5%
of single-mode baseline.

**Mock-GPU tests:** unit-level coverage of registry, partition,
encode segment construction, manifest persistence.

## Out of scope (deferred)

- **Auto-promoting a static pipeline binding to a rule.** Users
  authoring `<Sg Pipeline={{ cull: "back" }}>` go through the trivial-
  rule path (Phase 5). No optimization yet for "this binding never
  varies, skip the segment table" — measurement first.
- **Cross-bucket pipeline sharing.** Two buckets with different
  effects but the same `depthCompare` don't share a pipeline (they
  can't — pipeline is `(layout, shader, state)`). No optimization
  warranted.
- **GPU-side pipeline indirection.** Some future WebGPU extension may
  let the GPU pick a pipeline. Today: pipeline switching is CPU-side
  in the encode loop. K small drawIndirects is the cost; acceptable.
- **WebGL2 backport.** Heap renderer is WebGPU-only; derived modes
  inherits.

## Order of merges

1. Phase 0 alone — pipeline cache + manifest, no integration. Ships
   to wombat.rendering as a new exported subpath.
2. Phases 1+2 — rule API + compute pass, evaluated but unused by
   encode. Tests verify hashes.
3. Phase 3 — partition pass. Tests verify segmentation.
4. Phase 4 — encode segmented draws, gated behind
   `enableDerivedModes: false` default. Existing scenes unaffected.
5. Phase 5 — flip default, drop pipelineState from bucket key. Big
   regression-test moment; keep the gate flag for one release in case
   of fallback.
6. Phase 6 — persistence + pre-warm. Independent of correctness;
   pure perf.
7. Phase 7 — demo + e2e tests harden the path.

Each phase lands as one or two PRs to wombat.rendering with paired
tests. Phase 5 is the only one with cross-cutting risk; everything
else is additive.

## Estimate

Phase 0:   ~400 LOC, 1 day.
Phase 1:   ~250 LOC, 0.5 day. Mostly trivial given §7 marker as template.
Phase 2:   ~500 LOC, 1.5 days. Codegen extension is the bulk.
Phase 3:   ~400 LOC, 1.5 days. Counting-sort in WGSL is fiddly.
Phase 4:   ~200 LOC, 1 day.
Phase 5:   ~150 LOC + regression risk, 1 day.
Phase 6:   ~250 LOC, 0.5 day.
Phase 7:   ~400 LOC (demo) + ~600 LOC (tests), 1.5 days.

Total: ~3150 LOC, ~8.5 days end-to-end. Phases 0–4 are the
load-bearing work; 5 is the integration moment; 6–7 are polish.
