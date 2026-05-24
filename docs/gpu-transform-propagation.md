# GPU Transform Propagation — Implementation Plan

Status: **IMPLEMENTED** (rendering 0.19.12 + dom 0.14.2). The SG emits a per-RO
Model ancestor chain; the heap GPU-composes a per-RO Model constituent (fwd+inv,
df32) in a chain pass before §7; §7 derives ModelView / inverses / NormalMatrix /
custom rules from it unchanged. A root `cval` shared by N descendants now marks
exactly its 2 constituent slots (not N composites) — the fan-out is gone. Real-GPU
tested (chain math, chain→constituent→§7 ModelView, fan-out) + structural (CPU)
tests + end-to-end validated on heap-demo-sg (correct render incl. trafo-determinant
cull). Constant-run folding shipped in dom 0.14.3 (consecutive constant scopes
pre-multiply into one chain link). **Remaining optimization (deferred, not blocking):**
the Phase-2 colored prefix-scan (shared ancestor-prefix sharing across siblings).
Companion to the TODO entry of the same name.

## Problem

Some scene-graph attributes compose non-trivially down the hierarchy — `Trafo`
(parent·child mat-mul) and `active` (AND) being the ones that matter. Today the
SG composes them **eagerly on the CPU** during traversal:

- `wombat.dom traversalState.ts:119` — `composeModel` = `AVal.zip(child, parent).map((c,p) => c.mul(p))`, built per scope in `pushTrafo` (`:404`).
- `traversalState.ts:450-453` — `pushActive` = `AVal.zip(parent, child).map((p,c) => p && c)`.

So each leaf gets its **own** composed `ModelTrafo`/`active` aval that depends on
every ancestor. A single `cval<Trafo3d>` above N leaves ⇒ **N composed avals
marked** for one logical change ⇒ N dirty heap slots ⇒ N CPU re-evaluations and
uploads. This is the worst adaptive fan-out in the stack, and aardvark eats it
too.

**Goal:** the CPU does O(changed constituents) work, never O(ROs). Composition
moves to the GPU, reusing the df32 machinery the heap already has.

## Key realization — §7 already does 80% of this

The §7 derived-uniforms compute pass is *already* a per-RO df32 matrix-chain
product with exactly the indirection we need:

- **Constituent buffer** (`derivedUniforms/slots.ts`): one df32 mat4 slot per
  *distinct* `aval<Trafo3d>`, stored as fwd+inv pair (128 B each, hi/lo split).
  Dirty tracking already drains to O(changed) slot uploads (`pullDirty`).
- **Records** (`derivedUniforms/records.ts`): `[rule_id, out_slot, in_slot…]`
  where each `in_slot` is an **index** (handle) into the constituent buffer —
  *not* an inline matrix. This is precisely the "indices, not matrices"
  indirection that keeps a shared root at O(1): N records reference the root's
  one slot by index.
- **`matmulChain` kernel arm** (`derivedUniforms/codegen.ts:186-214`): composes
  `L₀·L₁·…·Lₙ` in df32 (`df_mul`/`df_add`, codegen.ts:79-115), one GPU thread per
  record, writing the result to the RO's drawHeader slot.
- **Dispatch + reactivity** (`heapScene.ts:3966-3983`): `pullDirty` →
  `uploadDirty` (O(changed)) → `encode` (dispatch gated on "any dirty").

The only gaps between this and SG flattening:

1. **Chain length is per-*rule* (compile-time)**, fixed by the rule IR's leaves
   (e.g. `MVP = Proj·View·Model`, 3 leaves). SG chains are **per-RO and
   variable-length** (the ancestor path).
2. **The chain's constituents come from a fixed recipe** (Model/View/Proj), not
   from the SG ancestor path.
3. **The SG composes eagerly** — it must instead *emit the chain*.

So Phase 1 is "make §7's chain variable-length and SG-sourced," not "write a
scan." The colored prefix-scan (shared-prefix sharing) is Phase 2, and only if
the GPU redundancy ever profiles as a cost.

---

## Phase 1 — per-RO variable-length chain product (kills the fan-out)

One GPU thread per RO, each looping its own ancestor chain. No prefix sharing
yet; siblings redundantly re-multiply shared ancestors (cheap after constant
folding). **This alone removes the CPU fan-out** — the win is the index
indirection + O(changed) uploads, not GPU sharing.

### 1a. SG side — emit chains, not composed avals (`wombat.dom`)

In `traversalState.ts`, stop building `composeModel`/`pushActive` composites.
Instead accumulate an **immutable chain** down the traversal:

- `TraversalState.modelChain: readonly TrafoLink[]` where a `TrafoLink` is either
  a distinct `aval<Trafo3d>` (dynamic) or a folded constant matrix.
- `pushTrafo(t)` appends to the chain (structural, O(1)), **no `AVal.zip`**.
- **Constant folding**: consecutive `AVal.constant` links collapse into one
  constant `M44d` at append time (identity check on the aval). A typical chain
  `[rootDyn, const, const, const]` compacts to `[rootDyn, foldedConst]`.
- `active`: a parallel `activeChain: readonly aval<boolean>[]` (constants folded
  to a single bool; if any folds to `false`, the leaf is dropped exactly as
  today at `compile.ts:682`).

The leaf→RO boundary (`compile.ts buildRenderObject`) attaches the chain to the
RO instead of a composed `ModelTrafo` aval. New `RenderObject` field, e.g.
`modelChain?: readonly TrafoLink[]` (and `activeChain?`), replacing the lazy
`ModelTrafo` resolution in `traversalState.tryGet`. Keep a *lazy*
`ModelTrafo` aval available for CPU consumers (see 1d) but never force it during
traversal/render.

> Migration note: `tryGetAutoUniform` (`traversalState.ts:740-755`) currently
> lazily composes `ModelView`/`MVP`. Those move to the GPU chain (1b); the
> provider stops handing composed trafos to the heap and instead exposes the
> chain.

### 1b. Rendering side — chain buffer + variable-length matmul arm

Reuse the constituent buffer and records; add a **chain buffer** and a new arm:

- **Constituent registration** (existing `slots.acquire`): each distinct chain
  link aval → a constituent slot. Shared root aval → one slot, N references.
- **Chain buffer**: a GPU buffer of constituent slot indices (the flattened
  per-RO chains). Per-RO record becomes `[rule_id = CHAIN, out_slot,
  chainOffset, chainLen]` (fixed stride); the variable part lives in the chain
  buffer at `[chainOffset, chainOffset+chainLen)`. (This is the same buffer
  Phase 2's scan will re-traverse.)
- **New kernel arm `CHAIN`** (`codegen.ts`): read `chainOffset`/`chainLen`, loop
  the chain buffer gathering constituent slots, accumulate the df32 product with
  the existing `df_mul`/`df_add`, write to `out_slot` (the RO's `ModelTrafo`
  drawHeader slot). A runtime loop instead of the per-rule unrolled chain.
- **Spec plumbing**: `HeapDrawSpec` carries the per-RO `modelChain` (ordered
  link avals). `registerRoDerivations` (`sceneIntegration.ts:185-246`) registers
  each link as a constituent and emits the chain-buffer entries + the `CHAIN`
  record.

### 1c. Bypass the CPU upload for GPU-composed uniforms

Today a `ModelTrafo` uniform goes through `pool.acquire` + the per-frame
drawHeader `writeBuffer` (`heapScene.ts:3057-3131`, `:3887-3903`). For
GPU-composed trafos:

- Allocate the `ModelTrafo` arena slot once, **don't** `pool.acquire` its value,
  and **don't** subscribe/CPU-upload it. The `CHAIN` arm writes that slot
  directly. (§7 already does exactly this for its derived outputs — same path.)
- The heap must treat these slots as **GPU-managed** (skip dirty/upload). This
  is the one place to be careful the two dirty systems don't both claim the slot.

### 1d. CPU value access (picking, gizmos)

The value still exists — only eager materialization is gone. Provide
`composedTrafoOf(ro): Trafo3d` that walks the RO's chain in **f64** on demand
(the link avals are CPU values), forced **outside** an adaptive computation
(allow-force) so one query doesn't re-subscribe the fan-out. Picking uses the
GPU compaction pass (separate TODO) which gathers `ModelForward` from the same
constituent/output buffer, so it doesn't even need this; gizmos/non-pick queries
do. **Hard invariant**: nothing may adaptively depend on *all* composed trafos
on the CPU.

---

## Phase 2 — colored/segmented prefix-scan (optional, shared-prefix sharing)

Only if Phase 1's redundant sibling re-multiplication profiles as a cost (deep,
wide hierarchies). Re-traverse the same chain buffer with a **segmented df32
mat-mul prefix-scan**, reusing `scanKernel.ts`'s tile→block→propagate skeleton:

- Swap the `+` operator (`scanKernel.ts:64` blelloch step, `:92` input, `:101`
  output) for df32 mat-mul; add per-segment head-flags ("colors").
- Layout the chain buffer as an **SG Euler-tour / topological order** so shared
  ancestor prefixes are computed once and inherited (segment tail = world
  trafo). mat-mul is associative (scan-valid) + non-commutative (order-preserving
  root→leaf).
- Same constituent buffer, same outputs; only the GPU traversal of the chain
  buffer changes. Keep "recompute all colors when any constituent dirty" — don't
  build a GPU dirty-subtree frontier (the indirection already makes CPU
  O(changed); GPU brute-force beats the bookkeeping).

---

## §7 fusion, inverses, NormalMatrix

- **Fusion**: a leaf's `MVP` is just `[Proj, View, …modelChain]` — prepend the
  View/Proj constituents to the model chain and the `CHAIN` arm yields `MVP`
  directly; `ModelView` = `[View, …modelChain]`. §7's fixed `matmulChain`
  collapses into the degenerate case of the variable `CHAIN` arm. End state: one
  constituent list + one chain pass produces `ModelTrafo` / `ModelView` / `MVP`.
- **Inverses / NormalMatrix**: each constituent slot already stores fwd **and**
  inv. `(A·B·C)⁻¹ = C⁻¹·B⁻¹·A⁻¹`, so an inverse chain = the reversed chain over
  the inv slots — representable as another `CHAIN` record. `NormalMatrix =
  upperLeft3(transpose(inverse(model)))` = reversed-inv chain → upper3 →
  transpose (reuse §7's normal-matrix arm tail).

---

## Hard parts / risks

1. **SG *structure* changes vs value changes.** Value change → mark one
   constituent, GPU re-composes: clean. *Structure* change (add/remove/reparent)
   → incrementally rebuild the chain buffer + records (O(structural delta)).
   This is new bookkeeping in the SG→heap bridge and the riskiest piece.
2. **GPU-managed slot bypass (1c).** Ensuring the heap's CPU dirty/upload path
   and the chain pass don't both write the same slot.
3. **Variable-length kernel + record sizing.** Fixed-stride record + separate
   chain buffer; chain-buffer growth/compaction under churn.
4. **Constant-fold correctness.** Only `AVal.constant` (immutable identity) is
   foldable; a `cval` that happens to be unchanged is **not**.
5. **Precision parity.** GPU df32 vs CPU f64 (`composedTrafoOf`) differ by df32
   error — negligible for render + pick, but document it.

---

## Refinements from implementation (Slice 1 landed)

Slice 1 shipped: the variable-length `CHAIN` arm + records + dispatch + tests
(commit ab691e7). Reading the heap's §7 wiring (`heapScene.ts:3381-3428`) refined
the integration plan:

- **The chain must produce a per-RO `Model` *constituent*, not final uniforms
  directly.** The real effects need more than the forward `ModelView`/`MVP`:
  heap-demo's VS uses `ModelViewTrafoInv` (normals) and a **custom** rule
  `WorldUpInModel = u.ModelTrafo.inverse()…`. A "chain emits the final forward
  uniform" shortcut can't serve inverses or arbitrary custom rules that read
  `Model`. So the chain writes the RO's `Model` (fwd **and** inv halves) into a
  **per-RO constituent slot**, and §7 consumes it through its existing recipes
  (`ModelView`, `ModelViewInv`, `NormalMatrix`, custom rules) **unchanged**.
  This is the true §7 fusion and it's drop-in.
- **Inverse needs no new arm.** `(L0·…·Ln)⁻¹ = Ln⁻¹·…·L0⁻¹`. The fwd arm already
  products whatever handles/order it's given, so the inv half = the **same arm
  over the links' inv slots in reverse order**. (NormalMatrix = `upper3(transpose
  (inv))` — one small extra arm, or compute it in §7 from the GPU-written inv.)
- **Two dispatches, ordered.** A compute dispatch has no intra-dispatch ordering,
  so a record that *writes* a constituent and one that *reads* it can't share a
  dispatch. The chain pass (writes per-RO `Model` constituents) must run as a
  **separate dispatch before** the §7 pass (reads them). That means: (a) the
  constituents buffer becomes `read_write` for the chain pass, (b) the chain
  records live in their own dispatch phase, (c) `encode()` sequences chain-pass
  → §7-pass. This is the main structural change for the heap slice.
- **Per-RO output constituent slots.** Need a slot allocator for GPU-written
  outputs (not aval-keyed like `ConstituentSlots.acquire`); the chain RO owns a
  fwd+inv pair that §7's `Model` leaf points at.

Revised slice boundary: **Slice 2 = the two-phase dispatch + chain→Model-
constituent write + §7 `Model`-leaf redirect** (rendering-only, hand-built specs,
real-GPU tested incl. an inverse/normal round-trip). **Slice 3 = heap spec
plumbing (`HeapDrawSpec.modelChain`) + SG chain emission + structural add/
remove/reparent**, A/B pixel-validated against the eager path on heap-demo-sg.

## Sequencing

1. **Rendering: variable `CHAIN` record + arm + chain buffer**, driven by a
   hand-built `HeapDrawSpec.modelChain` in a unit/real-GPU test (no SG yet).
   Validate against the existing `matmulChain` for fixed chains (identical
   output).
2. **1c GPU-managed slot bypass** + reactivity (one constituent change → one
   upload → re-dispatch).
3. **SG side (1a)**: chains in `traversalState`, constant folding, RO field,
   `compile.ts` wiring; keep the eager path behind a flag for A/B.
4. **`composedTrafoOf` (1d)** + wire picking/gizmos.
5. **§7 fusion** (model chain absorbs View/Proj; NormalMatrix inverse-chain).
6. **`active` AND-chain** (parallel arm/pass).
7. **Phase 2 scan** — only if profiling demands.

## Validation

- **Correctness**: pixel-identical to the eager-CPU path on heap-demo-sg
  (A/B via the Phase-1 flag).
- **The whole point**: a scene of N ROs under one root `cval<Trafo3d>` — assert
  changing it produces **exactly one** constituent upload (not N) and one
  dispatch, and the render updates. This is the regression test that proves the
  fan-out is gone.
- **Perf**: frame time on root-cval change at N = 20K (CPU should be flat in N).
- **Precision**: large-world-coordinate scene renders without f32 wobble (df32
  chain).

## Files (anticipated)

- `wombat.dom`: `src/scene/traversalState.ts` (chains, folding), `compile.ts`
  (RO wiring), `src/scene/sg.ts`/`constructors.ts` (field plumbing),
  `src/core/renderObject.ts` (`modelChain`/`activeChain` fields — in rendering).
- `wombat.rendering`: `src/runtime/derivedUniforms/{records,codegen,slots,
  sceneIntegration,dispatch}.ts` (chain buffer + `CHAIN` arm + registration),
  `src/runtime/heapScene.ts` (GPU-managed slot bypass, spec plumbing),
  `src/core/renderObject.ts` + `heapAdapter.ts` (`modelChain` on spec),
  `src/runtime/derivedUniforms/cpuEval.ts` or a new module (`composedTrafoOf`).
