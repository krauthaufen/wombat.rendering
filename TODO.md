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
- **`runFrame` dirty-skip** — skip rAF work when nothing is marked dirty (idle
  canvases should do nothing per frame). In `loop.ts`.
- **Per-instance FS varying regression test** — works, but not regression-tested
  against complex derived-modes setups.
- **Example ports** — examples that were waiting on the scene-graph layer can now
  use wombat.dom; revisit (hello-cube + UBO/camera, instanced grid, post chain).

## Derived-modes close-out (none urgent)

- **Lift the `totalSlots ≤ 16` kernel cap** — widen the partition's bind-group
  layout; the cartesian over active axes exceeds 16 fast (cull(3) × depthCompare(8)
  = 24). Pairs with the stencil-axis rules (central TODO #6).
- **Build-time vite-plugin diagnostics** — warn on open output domains and on
  per-frame `derivedMode(...)` construction.
- **GPU-side rule chaining** — a derived-mode rule reading a §7-derived uniform's
  output rather than a raw arena uniform.

## Conditional (only if profiling demands)

- **Per-vertex (drawIdx, vid) lookup table** to replace the megacall binary
  search (≈8 B/emit, ≈5.76 MB at 720K verts). Worth it only if mobile profiling
  shows per-vertex reads are the bottleneck.

## Out of scope (recorded so they don't resurface)

- **Multiple queues / async submission** — WebGPU exposes one queue per device.
- **Per-token source-map granularity** beyond per-statement (diminishing returns).
- **Scene graph** — lives in wombat.dom, not here.
