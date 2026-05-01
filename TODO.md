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
