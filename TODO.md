# wombat.rendering — TODO

Status: ✅ production-ready (0.19.3). WebGPU render layer + heap renderer
fast-path. Shipped: megacall encode (O(buckets) not O(draws); 50K objects ~44fps
on iPhone), chunked GPU arenas + best-fit freelist + pooled allocation, value-keyed
multi-pipeline buckets, GPU derived uniforms (registry/rule IR) + derived modes
(pipeline-state-as-fn-of-uniforms via GPU partition kernel), texture atlasing
(tiers S/M/L, reactive residency), per-RO instancing.

**The big architectural threads (IR heap rewriter, decoder composition, periodic
compaction, stencil modes, cubemap atlasing, scene prewarm, device.lost,
family-merge, WebGL2) live in `~/claude/wombat-todo.md`.** Design docs kept under
`docs/` (heap-design-arc, heap-renderer-design, derived-modes, derived-uniforms,
derived-uniforms-extensible, heap-decoder-composition, heap-ir-refactor,
heap-textures-plan, multi-pipeline-buckets-integration, heap-debug-tools).

## Open (repo-level)

- **MSAA-on-canvas real-GPU test** — the offscreen MSAA path is validated via
  `renderTo`; the canvas swap-chain MSAA path inherits the same resolveTarget
  plumbing but has no direct browser test reading back the swap-chain texture.
- **alphaToCoverage real-GPU test** — wired in `prepareRenderObject`, no
  end-to-end pixel validation yet.
- **Per-instance FS varying × derived-modes test** — per-instance attributes as
  FS varyings are real-GPU tested (`tests-browser/instancing-real`), but NOT in
  combination with `derivedMode`; add that combined case.
- **Example ports** — examples that were waiting on the scene-graph layer can now
  use wombat.dom; revisit (hello-cube + UBO/camera, instanced grid, post chain).

## Derived-modes close-out (none urgent)

- **Lift the `totalSlots ≤ 16` kernel cap** — widen the partition's bind-group
  layout; the cartesian over active axes exceeds 16 fast (cull(3) × depthCompare(8)
  = 24). Pairs with the stencil-axis rules (central TODO #6).
- **Build-time diagnostics** — warn on open derived-mode output domains and on
  per-frame `derivedMode(...)` construction. (No build plugin ships from this
  package today; would live in wombat.shader-vite or a new one.)
- **GPU-side rule chaining** — a derived-mode rule reading a §7-derived uniform's
  output rather than a raw arena uniform.

## GPU transform propagation (IMPLEMENTED, 0.19.12 / dom 0.14.2)

**Full plan + status: `docs/gpu-transform-propagation.md`.** Shipped: SG emits a
per-RO Model ancestor chain → heap GPU-composes a per-RO Model constituent
(fwd+inv df32, chain pass before §7) → §7 derives ModelView / inverses /
NormalMatrix / custom rules unchanged. A root cval over N descendants marks 2
slots, not N. Constant-run folding (dom 0.14.3) + Phase-2 prefix-sharing suffix
trie (0.19.13, shares ancestor sub-chains across siblings, topological level
dispatch) both shipped — feature complete. Solves the worst adaptive fan-out: a `cval<Trafo3d>` above 20K objects marks 20K
composed `ModelTrafo` avals → 20K dirty heap slots, for ONE logical change. Same
shape for `active` (AND-composition) and any other non-trivially-composed SG
attribute. The heap already composes trafos in a compute shader (§7) — extend
that to the SG chain so the CPU never composes chains during traversal.

- **Constituent buffer** (df32 matrices), one slot per *distinct* trafo aval.
  Subscriptions scale with distinct trafos, NOT ROs — a root cval has ONE
  subscriber. On mark: write one slot + flag dirty. CPU work = O(changed), no
  fan-out.
- **Chain buffer** holds `(constituentIndex, color)` — INDICES, not matrices.
  Critical: inlining matrices would re-duplicate a shared root into N segments
  and bring the CPU fan-out back as N buffer writes. Indices keep a root change
  at O(1) CPU.
- **Colored / segmented df32 mul-scan** (mat4×mat4 is associative + non-
  commutative → order-preserving root→leaf). Reuse `scanKernel.ts`'s
  tile→block→propagate skeleton: swap `+`→df32 `mul`, add segment head-flags.
  Segment tail = the RO's world trafo → write its heap slot. `active` = same
  kernel, boolean AND.
- **Constant-fold** runs at SG build (via `AVal.constant` identity): a chain
  `[rootDyn, const, const]` compacts to `[rootDyn, foldedConst]`. Most chains →
  1 dynamic + 1 const, so per-leaf chains stay tiny.
- **Recompute policy**: gate the pass on "any constituent dirty"; when it runs,
  re-scan all colors (20K short df32 chain-products ≈ µs). Don't build a GPU
  dirty-subtree frontier — the index indirection already makes CPU O(changed);
  GPU brute-force beats the bookkeeping.
- **§7 unification**: a leaf's MVP chain is `[Proj, View, …modelAncestors]` —
  View/Proj composition is the same scan with a couple constituents prepended.
  §7 collapses into the degenerate 2-element case: one constituent list, one
  scan produces ModelTrafo / ModelView / MVP / NormalMatrix.

Layouts: (1) per-RO contiguous compacted chains, color = RO id — simplest first
cut; siblings redundantly re-multiply shared ancestors (cheap after folding).
(2) SG Euler-tour / topological layout sharing ancestor prefixes — the deep/
wide-tree optimization, later.

Hard parts: (a) **chain buffer under SG *structure* changes** (add/remove/
reparent) — incremental rebuild of the index/color layout, distinct from value
changes; (b) the heap must treat scan-output slots as **GPU-managed** and skip
its CPU dirty/upload path (§7 already does this for derived uniforms).

**CPU value access** — the value still exists, we just don't eagerly materialize
all of them. Walk one RO's chain on demand in f64 (`composedTrafoOf(ro)`),
forced OUTSIDE an adaptive computation so it doesn't re-subscribe the fan-out
(gizmos / non-pick queries). Forbidden: anything that adaptively needs ALL
composed trafos on the CPU (that IS the fan-out — push it to the GPU, which now
holds them).

## Pixel picking → GPU compaction pass (design captured)

Picking is complex today. The **BVH/CPU path stays** for shapes whose pick
geometry ≠ render geometry (wireframe cube, fat lines, billboards). For the ~90%
(pixel-accurate against real geometry), a GPU pick pass digests the readback so
the CPU gets per-object data directly — and the trafo rides along (the GPU has
it from the transform-propagation scan), so no CPU chain-walk needed for picks:

- One compute pass over the **33×33** cursor neighborhood. Per pixel: read `id`,
  reconstruct position (depth + inv-VP, or a position attachment).
- **Compact per id**: dedup ~1089 pixels → the handful of distinct ids present,
  each carrying its position cluster. Fits one workgroup (~17 KB shared);
  distinct-id set via shared-mem atomics or a tiny sort.
- **Gather `ModelForward` once per distinct id** from the scan's trafo buffer
  (`trafo[id]`; keep pick id == heap draw id == trafo slot so it's a direct
  gather). Download `{ id, ModelForward, [p0,p1,…] }` — tiny, CPU-ready:
  object-space coords via `inv(Model)·p`, nearest-hit / centroid / brush from
  the cluster. Download `ModelForward` as df32 if far-from-origin precision
  matters.
- **Refine later**: with the rough hit known, run a real ray/triangle
  intersection for an exact point.
- **See-through / pick-through** (the exciting one): walk the OIT A-buffer
  linked-list per pixel instead of a single id buffer — same compaction, but
  emit the **ordered front-to-back hit sequence** (transparent layers + the
  opaque frontmost as the terminator). Then SG events get a second propagation
  axis: **depth/ray order, orthogonal to the SG hierarchy bubble.**
  - Per hit (front→back) run the normal hierarchy dispatch; if rejected,
    advance to the next ray hit. The whole walk is CPU-side over the one
    downloaded ordered list — no per-layer re-pick / GPU round-trip.
  - Verbs: `event.pickThrough()` ("not mine" → next ray hit) is **orthogonal**
    to `stopPropagation()` (stops the hierarchy bubble for the current hit).
    Default: a handler **consumes**; no handler = implicit pass-through.
  - Compaction granularity for pick-through: dedup *consecutive* same-id
    fragments (one object's front face) but **preserve cross-object depth
    order** — do NOT fully collapse per-id (that loses the ordering pick-through
    rides on).
  - Bounded by the A-buffer depth cap (how many layers the linked list
    captured).
  - Unlocks: translucent overlays deferring to content behind, non-interactive
    "ghost"/preview geometry, and **click-to-cycle stacked objects** (each click
    `pickThrough`s one more layer) — the DCC "select the thing behind" gesture,
    for free.

Makes pick another consumer of the GPU trafo buffer (render + pick read one
GPU-resident source). Depends on the transform-propagation scan above (for the
per-id `ModelForward` gather) and on the existing OIT A-buffer (for the ordered
layers).

## Conditional (only if profiling demands)

- **Per-vertex (drawIdx, vid) lookup table** to replace the megacall binary
  search (≈8 B/emit, ≈5.76 MB at 720K verts). Worth it only if mobile profiling
  shows per-vertex reads are the bottleneck.

## Out of scope (recorded so they don't resurface)

- **Multiple queues / async submission** — WebGPU exposes one queue per device.
- **Per-token source-map granularity** beyond per-statement (diminishing returns).
- **Scene graph** — lives in wombat.dom, not here.
