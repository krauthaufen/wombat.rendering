# Derived modes — implementation plan

Companion to `derived-modes.md` (final design). This is the build
sequence: phases, files touched, success criteria. Modelled on §7 v2's
staging.

## Scope summary

- Pipeline state becomes a per-RO **derived value** with rules
  authored alongside derived uniforms (`derivedMode(axis, rule)`).
- Continuous baked fields (depthBias triplet, blendConstant,
  stencilReference) are **carved out** — they stay static per-effect
  / per-SG-scope or as dynamic per-pass state.
- The full pipeline domain is **enumerated at scene build** via
  cooperating SG-side traversal + rule-IR analysis, and pre-warmed
  with `createRenderPipelineAsync`. `scene.ready()` resolves after
  all pipelines link.
- Bucket key drops pipelineState → `(effect, textureSet)`.
- Runtime is **readback-free**: pipeline domain has fixed slot
  indices; partition kernel writes per-slot indirect args; encode
  iterates slots and emits one `drawIndirect` per slot. Empty slots
  draw zero.

## Phase 0 — Pipeline cache + canonicalization

Foundation. CPU-only; testable without GPU integration.

**New module:** `packages/rendering/src/runtime/pipelineCache/`

- `descriptor.ts` — `PipelineStateDescriptor` (normalized struct).
  Canonicalization: field order, default-elision, attachment list
  sort. Carve-out enforced: depthBias triplet / blendConstant /
  stencilRef present only as the static / dynamic-state values they
  resolve to, not as per-RO-varying fields.
- `bitfield.ts` — `encodeModeKey(descriptor) → bigint` and
  `decodeModeKey(bigint) → descriptor`. Bitfield layout documented
  here (~24 bits typical, ≤ 64 bits worst case). The encoded key is
  the cache key directly.
- `cache.ts` — `PipelineCache`:
  - `Map<bigint, GPURenderPipeline>` keyed on encoded modeKey.
  - `precompile(descriptors[]): Promise<void>` → fires
    `createRenderPipelineAsync` for each, awaits all.
  - `getSync(modeKey): GPURenderPipeline | null` — sync lookup, hot
    path.
- `manifest.ts` — IndexedDB persistence. Stores realized descriptors
  keyed on `(appVersion, wgslHash)`. On boot, replays into the cache
  during pre-warm so the browser's internal WGSL cache is hot.

**Tests:** `tests/pipelineCache/`:
- Bitfield round-trip determinism.
- Descriptor canonicalization equality.
- `precompile` deduplicates concurrent requests for the same modeKey.
- Manifest round-trip through `fake-indexeddb`.

Done when: encode/decode is provably symmetric, cache hits/misses
behave as documented, manifest persists across simulated reloads.

## Phase 1 — Rule authoring API + build-time marker

Mirrors `derivedUniform` ergonomics so users don't learn a new shape.

**Runtime:** `packages/rendering/src/runtime/derivedModes/`

- `marker.ts`:
  - `derivedMode<A extends ModeAxis>(axis, build, leafTypes?, domain?)`
    → `DerivedModeRule { axis, ir, __derivedModeRule: true, domain? }`.
  - `ModeAxis = "topology" | "stripIndexFormat" | "frontFace" | "cullMode"
              | "unclippedDepth" | "depthWriteEnabled" | "depthCompare"
              | "blend" | "colorWriteMask" | "stencil"`.
  - `build: (u, declared) => ModeValue<A>`. The proxy minting
    `u.<Name>` reuses §7's `DerivedExpr` / leaf-builder machinery
    wholesale.
  - Validation: rule output values must be members of axis's valid
    enum (or the user-supplied `domain`).
- `isDerivedModeRule(x)` — brand check.

**Build-time:** `wombat.shader/packages/vite/src/inline.ts`

- Extend the `derivedUniform` recognizer to also match
  `derivedMode(...)` calls. Same `UniformScope`-derived `leafTypes`
  hint injection; same validation that rule leaves aren't resources.
- Warn on `derivedMode(...)` calls that aren't at module scope
  (anti-pattern: rules constructed inside render closures defeat
  enumeration).

**Tests:** `wombat.shader/tests/inline-derived-mode.test.ts` — typed
leaves from UniformScope, two-arg signature handling, anti-pattern
warning, validation errors for bad outputs.

Done when: a user can write
`derivedMode("cull", (u, d) => u.Side === 0 ? "back" : d)` and get a
properly typed `DerivedModeRule` with a typed IR.

## Phase 2 — Static analyzer

The compile-time piece that makes "no readback" sound.

**New module:** `packages/rendering/src/runtime/derivedModes/analyzer.ts`

- `analyzeOutputDomain(rule: DerivedModeRule, declaredInputs: Set<ModeValue>):
    { kind: "finite", values: Set<ModeValue> } | { kind: "open", reason: string }`
- Conservative IR interpretation:
  - Literal return → singleton set.
  - Identity (`return declared`) → declared.
  - Conditional with literal arms → union of arm sets.
  - Switch over a uniform with known finite domain → union of arm
    sets. Uniform domains come from the marker's `leafTypes` or an
    optional `valueDomain` augmentation.
  - Anything else (arithmetic, unbounded inputs, dynamically-keyed
    table lookups) → `open` with a clear "I can't prove this is
    finite — supply a `domain` hint" diagnostic.
- `domain?` hint on the marker short-circuits the analyzer.

**Tests:** `tests/derivedModes/analyzer.test.ts` — every IR shape with
both finite and open cases.

Done when: every example rule in the design doc analyzes correctly;
authored "open" rules produce actionable diagnostics.

## Phase 3 — SG-side declared-mode collection + enumeration

**Edits to:** wombat.dom's SG compile (`src/scene/compile.ts`) and
wombat.rendering's `heapScene.ts`.

- During SG traversal, gather `declaredModes : Map<EffectId, Map<Axis,
  Set<ModeValue>>>`. Today's `derivePipelineState` resolves a single
  value per axis per leaf; new code accumulates them as a set per
  effect.
- After traversal, for each effect:
  - For each axis: run `analyzeOutputDomain(rule, declaredModes[E][axis])`.
  - Compute the cartesian product across axes → set of effective
    `PipelineStateDescriptor`s for this effect.
- Union across effects → scene's full pipeline domain. Bail with a
  clear error if cardinality exceeds a configurable cap (default 256).
- Assign each descriptor a stable **slot index** (0 .. P_total - 1).
  Slot indices are scene-lifetime stable; runtime additions append.

**Tests:** scene-build integration tests asserting:
- Static-everywhere scenes produce one slot.
- Cull-flip-via-determinant scenes produce two.
- Scenes with multiple effects produce the right union.
- Open rule + no `domain` → build error.

Done when: a heap-demo with mixed cull values reports 2 slots; with
multiple blend modes reports more; cap-violation reports a useful
error.

## Phase 4 — Pre-warm + `scene.ready()`

Where the pipeline domain becomes live `GPURenderPipeline` objects.

**Edits to:** `heapScene.ts` scene construction.

- After enumeration, build the `slotTable: { modeKey: bigint,
  descriptor, indirectOffset, pipeline?: GPURenderPipeline }[]`.
- `scene.ready(): Promise<void>` → fires `pipelineCache.precompile(...)`
  for the slot table's descriptors, awaits all. On resolution, every
  slot's `pipeline` field is populated.
- `manifest.warmFromManifest(appKey)` runs before `precompile` to
  replay persisted descriptors → browser's WGSL cache is hot before
  the actual pipeline objects are created.
- Public API:
  ```ts
  preWarmPipelineCache(scene, descriptors?): Promise<void>
  ```
  — descriptors omitted = pre-warm the enumerated set; supplied =
  also pre-warm an explicit extra set (for app-level hot combos that
  the analyzer doesn't see, e.g. dynamically-loaded effects).

Done when: a demo scene awaits `scene.ready()` and first frame
encodes with zero pipeline misses.

## Phase 5 — Mode-eval kernel

GPU compute pass. Reuses §7's kernel skeleton.

**New file:** `packages/rendering/src/runtime/derivedModes/kernel.ts`

- Codegen: extend §7's `printExpr` to emit the discrete-enum-bit
  packing for each axis. Output is one u32 (or u64 for stencil-rich
  scenes) per RO into a `modeKeysOut` buffer.
- Inputs: the same constituents heap / host uniform offsets §7 uses.
  Mode rules read the same leaves as derived uniforms.
- Per-axis emit: for axes with NO rule for the given effect, the
  kernel emits the constant bits from the SG-declared value (also
  stored per-RO in the drawHeader, or as a per-effect constant in
  the kernel).
- Dispatch: one thread per RO per affected bucket. Same threading
  shape as §7.

**Tests:** `tests/derivedModes/kernel.test.ts` — manual kernel
invocations producing expected modeKeys for known input sets.

Done when: a unit test plants `derivedMode` rules + inputs, runs the
kernel via MockGPU, and reads the expected modeKey bits back from the
buffer.

## Phase 6 — Partition kernel

GPU compute pass that converts per-RO modeKeys into per-slot draw
metadata.

**New file:** `packages/rendering/src/runtime/derivedModes/partition.ts`

- Inputs: `modeKeysOut`, `modeKeyToSlot` lookup table (GPU-resident,
  written CPU-side at enumeration time and on runtime mutations).
- Per thread (one per record):
  - Look up `slot = modeKeyToSlot[modeKey]`.
  - Atomic-add `(indexCount += record.indexCount × record.instanceCount,
    instanceCount += 1)` into `indirectArgs[slot]`.
  - Append the record's drawTable index to `slot`'s segment of the
    sorted output drawTable (via atomic-fetch-add into a per-slot
    cursor).
- Output: per-bucket `indirectArgs[P_total]` and a permuted drawTable
  sorted by slot.

**Tests:** `tests/derivedModes/partition.test.ts` — N records with K
distinct modeKeys produces the right per-slot counts and a correctly
permuted drawTable.

Done when: 1000-record bucket with 3 distinct modes produces 3
non-empty + (P_total - 3) zero-count indirectArgs entries.

## Phase 7 — Encode rewrite + bucket-key change

The integration moment.

**Edits to:** the per-bucket encode loop (likely
`bucketEncoder.ts` or `heapRenderer.ts`).

```ts
for (const slot of scene.slotTable) {
  encoder.setPipeline(slot.pipeline);
  encoder.setBindGroup(0, bucket.bindGroup);
  // dynamic state from the descriptor (always set, cheap):
  if (slot.dynamic.blendConstant) encoder.setBlendConstant(slot.dynamic.blendConstant);
  if (slot.dynamic.stencilRef !== undefined) encoder.setStencilReference(slot.dynamic.stencilRef);
  encoder.drawIndirect(bucket.indirectArgs, slot.indirectOffset);
}
```

**Edits to:** `heapEligibility.ts` + `bucketKeying.ts`:

- Bucket key drops pipelineState entirely → `(effect, textureSet)`.
- ROs with no `derivedMode` rules on any axis still go through the
  same machinery — their identity rules produce a singleton modeKey,
  occupying one slot.
- `psIdOf` and identity-based PipelineState hashing in `heapScene.ts`
  are removed.

**Migration tests:** every existing heap-renderer test gets verified.
The historical "20k cvals → 20k buckets" pathological case is added
as a regression test and asserted to produce ONE bucket.

Done when: existing heap-demo / heap-demo-sg render correctly; mixed-
cull demo renders correctly; the 20k-cvals stress test produces one
bucket.

## Phase 8 — Runtime mutation path

Handles RO additions whose declared values or rules introduce
unseen pipeline configurations.

**Edits to:** `heapScene.ts` add-RO path.

- On addRO: collect the RO's declared modes + rule output domains.
- Diff against existing `scene.slotTable` modeKeys.
- New keys: `device.createRenderPipeline(descriptor)` **synchronously**
  for each. JS returns immediately; driver compile happens in
  background. Append to slotTable; resize `indirectArgs` buffer and
  `modeKeyToSlot` GPU lookup.
- Encode the RO into its bucket as normal. First submit using a
  newly-created pipeline stalls the GPU queue while compile finishes
  — that's the accepted runtime cost.

**Note:** the persistence manifest (Phase 0) helps here — the
browser's WGSL compile cache is warm if the WGSL has been seen
before, so even the runtime-add stall is sub-ms on repeat sessions.

Done when: a demo adds an RO at runtime with a never-before-seen
modeKey, the frame renders correctly (with a measurable but bounded
frame-time spike), and subsequent frames are normal.

## Phase 9 — Demo + e2e tests

**Demo:** extend `examples/heap-demo-sg/`:
- A "mode zoo" — ROs with uniform-driven cull (determinant flip),
  pulse-driven blend toggle, layer-driven static depthBias (showing
  the carve-out — depthBias is per-effect, not per-RO derived).
- A "20k cvals" stress test demonstrating the historical
  pathological case now collapses to one bucket.

**Real-GPU tests** (headed-Chromium): sweep 10K / 30K records,
2 / 4 / 8 distinct mode combos. Assert:
- One bucket per effect.
- P_total drawIndirect calls per bucket (most empty).
- Frame time within 5% of single-mode baseline.
- No GPU→CPU readback (verifiable by intercepting `mapAsync` calls).

**Mock-GPU tests:** end-to-end coverage of enumeration, pre-warm,
kernel, partition, encode, runtime mutation.

## Out of scope (deferred)

- **GPU-side rule chaining** — a mode rule consuming a §7-derived
  uniform's value. Doable in v2 by scheduling §7 before mode-eval;
  not in v1.
- **Mode rules reading textures / SSBOs** — not allowed (same
  constraint as derived uniforms).
- **`unclippedDepth` extension** — supported if available; gated on
  feature detection.
- **WebGL2 backport** — heap renderer is WebGPU-only.

## Order of merges

1. Phase 0 — pipeline cache + manifest, standalone, exported subpath.
2. Phase 1 — rule API + vite marker. No runtime integration yet.
3. Phase 2 — analyzer. Pure CPU, testable in isolation.
4. Phase 3 — SG-side enumeration. Wires Phases 1+2 into scene-build.
5. Phase 4 — pre-warm + `scene.ready()`. Pipelines exist but unused.
6. Phases 5+6 — kernels. Wired up but encode still uses old path
   (gated behind `enableDerivedModes: false` default).
7. Phase 7 — encode rewrite + bucket-key change. The integration
   moment; cross-cutting regression risk. Gate flag flips here.
8. Phase 8 — runtime mutation path. Independent of correctness for
   static scenes.
9. Phase 9 — demo + e2e tests harden the path.

Each phase lands as one or two PRs with paired tests. Phase 7 is the
only cross-cutting moment; everything else is additive.

## Estimate

| Phase | LOC | Days |
|---|---:|---:|
| 0 — pipeline cache + manifest | ~500 | 1.5 |
| 1 — rule API + vite marker | ~300 | 0.5 |
| 2 — analyzer | ~400 | 1 |
| 3 — SG enumeration | ~300 | 1 |
| 4 — pre-warm + scene.ready | ~150 | 0.5 |
| 5 — mode-eval kernel | ~450 | 1.5 |
| 6 — partition kernel | ~350 | 1 |
| 7 — encode + bucket-key change | ~250 | 1 |
| 8 — runtime mutation | ~200 | 0.5 |
| 9 — demo + e2e tests | ~1000 | 1.5 |

Total: ~3900 LOC, ~10 days end-to-end. Phases 2 + 3 (analyzer + SG
enumeration) are the load-bearing intellectual work; Phases 5 + 6 are
the load-bearing kernel work; Phase 7 is the integration moment.
