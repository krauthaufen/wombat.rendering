# Derived mode rules — implementation plan

**Status (wombat.rendering 0.16.1) — SHIPPED.** See the status banner
at the top of `derived-modes.md` for the implementation map and what
diverges from the plan below. Headlines:

- The "per-bucket pipeline domain enumerated at scene build"
  framing was replaced by a **per-RO combo registry**. ROs sharing
  an effect can each carry an arbitrary subset of axis rules; the
  partition kernel emits one composer fn per combo, dispatched on
  `r.comboId`.
- Multi-axis combos in one bucket via mixed-radix-encoded cartesian.
- Rules read any number of arena uniforms (every packer type).
- `static-enumeration + scene.ready()` pre-warm is NOT shipped —
  pipelines are created synchronously as combos register. The
  runtime-mutation stall behaviour described below still applies
  to first-use compilation.

The body below is preserved as the original Task 2 plan.

---

**Task 2 of 2** for the derived-modes design (`derived-modes.md`).
Layered on **Task 1** (`multi-pipeline-buckets-plan.md`) which must
ship first.

This task adds the `derivedMode(axis, (u, declared) => modeValue)`
authoring shape on top of Task 1's multi-pipeline-bucket machinery.
Everything in Task 1 — slot table, partition kernel, encode loop,
pipeline cache, runtime-mutation path — is reused unchanged. Task 2
adds rule authoring, IR analysis, and a GPU-side mode-eval kernel
that replaces Task 1's "read current aval values" modeKey-production
step.

**Delivers:**

- Per-RO pipeline state as a function of uniform values (motivating
  case: `cull = flipCull(declared) if sign(det(ModelTrafo)) < 0`).
- Static enumeration of the full possible pipeline domain at scene
  build via cooperating SG-traversal + rule-IR analyzer — frame 0
  has every needed pipeline pre-warmed.
- GPU-side rule evaluation, so trafos and other GPU-resident
  uniforms feed mode rules without CPU readback.

## Prerequisites

Task 1 must be live:
- Pipeline cache + slot table + `modeKeyToSlot` GPU lookup.
- Partition kernel reading per-RO modeKey and writing per-slot
  indirect args.
- Encode loop iterating slot table.
- Runtime sync-createPipeline path for unseen values.

## Phase 1 — Rule authoring API + vite marker

**Runtime:** `packages/rendering/src/runtime/derivedModes/marker.ts`

- `derivedMode<A extends ModeAxis>(axis, build, leafTypes?, domain?)
    → DerivedModeRule { axis, ir, __derivedModeRule: true, domain? }`.
- `ModeAxis` enum covering every derivable axis (see
  `derived-modes.md`'s "What can and cannot be derived").
- `build: (u, declared) => ModeValue<A>` — the proxy minting
  `u.<Name>` reuses §7's `DerivedExpr` / leaf-builder machinery
  wholesale. The `declared` parameter is the second proxy, typed by
  the marker.
- Output validation: rule returns must be valid axis enum values
  (or members of the user-supplied `domain`).
- `isDerivedModeRule(x)` brand check.
- Public re-export from `@aardworx/wombat.rendering/runtime`.

**Build-time:** `wombat.shader/packages/vite/src/inline.ts`

- Extend the existing `derivedUniform` recognizer to also match
  `derivedMode(...)` calls. Same `UniformScope`-derived `leafTypes`
  hint injection mechanism; same rejection of resource-typed leaves
  (samplers / textures / SSBOs).
- Warn if the call is not at module scope (anti-pattern: rules
  constructed inside render closures defeat enumeration).

**Tests:** `wombat.shader/tests/inline-derived-mode.test.ts` — typed
leaves, two-arg signature handling, anti-pattern warning, leaf-type
rejection of resources.

## Phase 2 — Static analyzer

Pure CPU. Testable in isolation.

**New file:**
`packages/rendering/src/runtime/derivedModes/analyzer.ts`

```ts
type AnalyzerResult =
  | { kind: "finite", values: ReadonlySet<ModeValue> }
  | { kind: "open", reason: string };

analyzeOutputDomain(
  rule: DerivedModeRule,
  declaredInputs: ReadonlySet<ModeValue>,
): AnalyzerResult
```

Conservative IR interpretation:

- Literal return → singleton set.
- Identity (`return declared`) → declared.
- Conditional with literal arms → union of arm sets.
- Switch over a uniform with a known finite domain → union of arm
  sets. Uniform value-domains come from the marker's `leafTypes`
  or an optional augmentation (TBD: extend `UniformScope` shape).
- Tuple/object destructuring on `declared` is followed.
- Anything not provably finite → `open` with an actionable
  diagnostic.
- Marker-level `domain?` hint short-circuits the analyzer.

**Tests:** `tests/derivedModes/analyzer.test.ts` — exhaustive IR
shapes; every example rule in the design doc analyzes correctly;
authored "open" rules produce useful diagnostics.

## Phase 3 — SG-side declared-mode set collection

Today's SG traversal computes a single value per axis per leaf.
This phase extends it to gather the **set** of declared values per
axis per effect, fed to the analyzer.

**Edits to:** wombat.dom's `src/scene/compile.ts`
(`derivePipelineState` and friends).

- Walk the SG; per effect, accumulate the set of declared values
  contributed by every branch:
  ```ts
  declaredModes: Map<EffectId, Map<Axis, Set<ModeValue>>>
  ```
- Identity rules (no `derivedMode` for an axis) → axis's contribution
  to the pipeline domain is just the declared set.
- For each `derivedMode` rule:
  `axisOutputs[axis] = analyzeOutputDomain(rule, declaredModes[E][axis])`.
- Cross-product across axes → set of effective
  `PipelineStateDescriptor`s for the effect.
- Union across effects → `scene.pipelineDomain: Set<descriptor>`.
- Configurable cap (default 256). Build fails with a friendly
  "your scene's pipeline cross-product is too large" message when
  exceeded.

**Tests:** scene-build integration:
- Single declared value + identity rule → 1 descriptor.
- Determinant-flip-cull rule + `declared ∈ {"back"}` → 2 descriptors
  (`{ cull: "back" }`, `{ cull: "front" }`).
- Multiple branching SG declarations → correct union.
- Open rule + missing `domain` → friendly error.
- Cap exceeded → friendly error with the offending axis.

## Phase 4 — Replace Task 1's pre-warm with the enumerated domain

**Edits to:** `heapScene.ts` scene construction.

- Where Task 1 populated `scene.slotTable` from the realized values
  of current ROs at addRO time, Task 2 populates it from
  `scene.pipelineDomain` at scene-build time.
- `scene.ready()` precompiles every descriptor in the enumerated
  domain (instead of only the currently-realized ones).
- ROs added at runtime whose declared values fall in the enumerated
  domain → no slot extension, no stall.
- ROs added at runtime whose declared/rule outputs introduce
  truly-new values → incremental analyzer pass on the new subtree,
  extend slot table, sync createRenderPipeline. (Same path as
  Task 1's runtime mutation, but rare in practice — the analyzer
  catches most domains at scene build.)

**Tests:** scene with declared cull `{"back"}` + flip rule
pre-warms `{back, front}`. Adding an RO with `declared = "back"`
takes zero slot extensions. Adding an RO with `declared = "none"`
(new declared value) extends correctly.

## Phase 5 — GPU-side mode-eval kernel

This is the kernel piece. Task 1's CPU-side modeKey production is
replaced (per-bucket toggleable) by a GPU kernel that evaluates
rules against the constituents heap + host uniforms.

**New file:**
`packages/rendering/src/runtime/derivedModes/kernel.ts`

- Codegen: extend §7's `printExpr` to:
  - Emit small integer literal outputs for axis enum values.
  - Pack outputs across all axes into the per-RO u32/u64 modeKey
    using the bitfield layout from Task 1.
- Inputs: same constituents heap + host uniform offsets §7 uses.
  Rule leaves resolve to those.
- For axes WITHOUT a rule on a given effect: kernel emits the
  constant bits from the SG-declared value (stored per-RO in the
  drawHeader, or as a per-effect constant baked into the kernel).
- Dispatch: one thread per RO in the bucket. Same threading shape
  as §7. Could share the §7 dispatch if rules don't chain across
  passes (v1: separate dispatches; v2 fuse if profile says it
  matters).
- Output goes to the same `modeKeys[numROs]` buffer Task 1's CPU
  path was writing to.

**Per-bucket selector:** if a bucket has any `derivedMode` rules,
its modeKey production switches from Task 1's CPU upload path to
the GPU kernel. Otherwise stays on the CPU path. Saves the kernel
dispatch for rule-free buckets.

**Tests:** `tests/derivedModes/kernel.test.ts` — install
`derivedMode("cull", (u, d) => sign(u.Det) < 0 ? "front" : "back")`,
provide constituents + host data, run kernel via MockGPU, read back
expected modeKeys.

## Phase 6 — Integration + dirty-gating refinements

- Wire the kernel into the per-frame compute submission sequence
  (after §7 derived-uniforms pre-pass; before Task 1's partition
  kernel which reads `modeKeys`).
- Dirty-gating: rules are dirty when ANY input aval marks. If no
  rule input marked AND no scene mutation, skip the mode-eval
  kernel; partition kernel reuses last frame's `modeKeys` buffer.
- Diagnostics: surface per-bucket counts of (CPU modeKey path, GPU
  rule kernel path) so users can see which buckets have rules
  active.

**Tests:** dirty-gate skips kernel correctly; mixed CPU/GPU
modeKey buckets produce correct slot assignments.

## Phase 7 — Demo + e2e tests

**Demo:** add to `examples/heap-demo-sg/`:
- A determinant-flip-cull effect: model trafo with a per-RO scale
  whose sign toggles via a checkbox. Visually verify rendered
  cullmode flips with sign, no flicker, no stutter.
- A blend-preset rule example (`u.Premultiplied`-driven preset
  switch).

**Real-GPU tests** (headed-Chromium): rule-driven scenes at 10k /
30k records. Assert:
- All pipelines enumerated at scene-build land in
  `scene.slotTable` after `await scene.ready()`.
- No runtime `createRenderPipeline` calls during steady state.
- Frame time within 5% of equivalent Task-1-only scene.

## Order of merges

1. Phase 1 — rule API + vite marker. No runtime integration.
2. Phase 2 — analyzer. Pure CPU, testable in isolation.
3. Phase 3 — SG enumeration. Wires Phases 1+2 into scene-build,
   but Task 1 still drives pre-warm.
4. Phase 4 — switch pre-warm source from "realized values" to
   "enumerated domain." Integration point with Task 1.
5. Phase 5 — GPU mode-eval kernel. Gated per-bucket; CPU path
   remains the default for rule-free buckets.
6. Phase 6 — dirty-gating + diagnostics.
7. Phase 7 — demo + e2e tests.

## Estimate

| Phase | LOC | Days |
|---|---:|---:|
| 1 — rule API + vite marker | ~300 | 0.5 |
| 2 — analyzer | ~400 | 1 |
| 3 — SG enumeration | ~300 | 1 |
| 4 — pre-warm switch | ~100 | 0.25 |
| 5 — GPU mode-eval kernel | ~500 | 1.5 |
| 6 — dirty-gating + diagnostics | ~150 | 0.5 |
| 7 — demo + e2e tests | ~700 | 1 |

Total: ~2450 LOC, ~5.75 days end-to-end. Phases 2 + 3 carry the
intellectual load (analyzer correctness + SG-traversal extension);
Phase 5 is the kernel work. Everything sits atop Task 1's machinery.

## Out of scope (deferred to v2)

- **GPU-side rule chaining** — a mode rule consuming a §7-derived
  uniform's value rather than a raw uniform. Doable by scheduling
  §7 before mode-eval; v1 disallows.
- **Mode rules reading textures / SSBOs / samplers** — same
  constraint as derived uniforms; build-time marker rejects.
- **Cross-effect rule sharing** — each effect's rules are
  independent in v1; if scenes commonly share rule shapes across
  effects, factor later.
