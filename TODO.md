# wombat.rendering — deferred work

What's NOT yet implemented. DONE markers pruned periodically.

## Examples

- Only `examples/hello-triangle`. More examples (`hello-cube` with
  a UBO + camera, instanced grid, post-processing chain) wait for
  the scenegraph layer that produces RenderObjects from a
  declarative tree — that layer lives outside this repo by design.

## Tests we'd take if free

- **MSAA-on-canvas real-GPU test.** `attachCanvas({ sampleCount })`
  is wired and the offscreen MSAA path is validated through
  `renderTo`; the canvas-swap-chain MSAA path inherits the same
  `beginPassDescriptor` resolveTarget plumbing, so it's likely
  fine. A direct browser test rendering MSAA into the canvas and
  reading back the swap-chain texture would close the gap.
- **alphaToCoverage real-GPU test.** Wired in `prepareRenderObject`
  via `pipelineState.alphaToCoverage`, no end-to-end pixel
  validation yet.

## Heap scaling (post-0.17.0)

See `docs/heap-future-work.md` §2-§5. Shipped this round: large-object
eject, best-fit Freelist, hardware-aware chunk cap, cursor-shrink.
Still open:

- **§3 full multi-chunk + multi-draw-call.** Bucket gains per-chunk
  `chunkParts`; allocator opens a new chunk when the current one
  hits its hardware-cap; encode iterates `bucket × chunkPart × slot`.
  Pool entries key by `(aval, chunkIdx)` so shared avals across
  chunks duplicate (acceptable cost). Trigger: when a real
  workload's arena steady-state crosses a single chunk's adapter
  cap (post-§2 eject, this is photogrammetry / dense pointcloud
  territory only).
- **§5 periodic compaction.** Walks live allocs, relocates them to
  fill the freelist, patches refs in drawHeaders + master records
  + derived-uniform input slots. Real defrag, not just
  cursor-shrink. Land when a high-churn workload shows visible
  fragmentation that cursor-shrink alone doesn't address.
- **§5 drop-empty-chunks.** Pairs with §3 — trivially "do nothing"
  in the single-chunk world.

## Derived-modes close-out (heap path)

The derived-modes architecture is shipped (see `docs/derived-modes.md`
status banner — wombat.rendering 0.16.1). Open items, none urgent:

- **`totalSlots ≤ 16` kernel cap.** Lift mechanically: widen the
  partition's bind-group layout (one BGL entry per slot count +
  draw table). At present the cartesian over active axes can
  exceed 16 quickly (cull(3) × depthCompare(8) = 24 already).
- **Static `scene.ready()` pre-warm** of the cartesian pipeline
  domain. Today pipelines are created sync on first `registerCombo`;
  first use of a freshly-introduced pipeline still stalls the GPU
  queue while the driver compiles. The design's enumeration story
  would let `await scene.ready()` cover everything.
- **Build-time vite-plugin diagnostics**: warn on open output
  domains, per-frame `derivedMode(...)` construction.
- **GPU-side rule chaining**: a derived-mode rule reading a
  §7-derived uniform's output rather than a raw arena uniform.
- **Stencil-axis rules** when stencil is enabled.

## Out-of-scope (recorded so they don't surprise us)

- **Multiple queues / async submission.** WebGPU's spec exposes one
  queue per `GPUDevice`. No portable concurrent submission today;
  revisit if/when WebGPU gains it.
- **Mock vs real-GPU coverage policy.** Mock tests assert
  implementation invariants (call-pattern, cache hits, ordering);
  real-GPU tests assert pixel correctness. Both kept.
- **`device.lost` rebuild-recovery.** On lost, the Runtime
  auto-disposes; user-level avals (`cval` / `clist` / `cset`)
  survive; caller constructs a fresh Runtime from a new device and
  re-compiles the same `alist<Command>`. Per-device caches use
  `WeakMap<GPUDevice, …>` so they re-key naturally.
  `MockGPU.simulateLost(...)` exposes a manually-resolvable
  `device.lost` for the auto-dispose test.
- **Per-token source-map granularity beyond what's there.** WGSL
  emitter ships per-segment maps for assignments / expression
  statements / output writes — sufficient for "go to source"
  navigation. Threading spans through every IR Expr node in the
  walk would be possible but adds plumbing for diminishing
  returns.
- **Scenegraph.** Lives outside this repo.
