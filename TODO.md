# wombat.rendering — deferred work

What's NOT yet implemented. DONE markers are pruned periodically
to keep this list scannable.

## Spec coverage

- **Multi-attachment renderTo.** Today `renderTo` returns a single
  `RenderToResult`; multi-target renders work but the user has to
  consume each color attachment via `result.color(name)` one at a
  time. Sufficient for v0.1.
- **MSAA on the canvas.** `allocateFramebuffer` handles MSAA color +
  resolve targets for offscreen / `renderTo` framebuffers, but the
  window/canvas swap chain stays single-sample. MSAA-to-canvas would
  need a hidden multisample target + resolve-into-swap-chain in
  `runFrame`. Add when a use case calls for it.

## Source-map decoder for shader-compile errors

`installShaderDiagnostics` now decodes the v3 source-map's `mappings`
and reports each WGSL-line message with its originating
`(file, line, col)` (falling back to the most recent mapped line
when the exact one is unmapped). Per-token granularity would require
the IR emitters to thread per-Expr spans into the writer; not worth
the cost yet.

## Compute primitive

`prepareComputeShader` lifts a `ComputeShader` (single-stage peer of
`Effect`) into an imperative dispatch surface: `createInputBinding()`
yields a mutable, name-keyed binding (`setUniform` / `setBuffer` /
`setTexture` / `setSampler`); `dispatch(binding, groups)` opens an
encoder, encodes the compute pass, submits, and awaits. Storage
buffers are validated end-to-end on real GPU.

What's still open:

- **Storage textures in compute bindings.** `setTexture` only handles
  sampled textures today; storage-texture writes need a separate
  binding entry path with the right `storageTexture` layout.
- **End-to-end uniform-block test for compute.** `setUniform` shares
  the byte-poke / layout machinery the render path's UBO test
  already covers, but a compute-side test that exercises the full
  uniform-binding path (parseShader + uniform ValueDef → iface
  uniform-block → write-buffer) is still missing.

## Examples

- Only `examples/hello-triangle`. Per the user's plan, more
  examples (`hello-cube` with a UBO + camera, instanced grid,
  post-processing chain) wait for the scenegraph layer that will
  produce RenderObjects from a declarative tree.

## Out-of-scope (recorded so they don't surprise us)

- **Multiple queues / async submission.** WebGPU's spec exposes
  one queue per `GPUDevice`. There is no portable way to do
  concurrent submission today; revisit if/when WebGPU gains it.
- **Mock-tested deltas vs real-GPU.** Mock tests assert
  implementation invariants (call-pattern, cache hits, ordering);
  real-GPU tests assert pixel correctness. Both kept.
- **`device.lost` rebuild-recovery.** Documented: on lost, the
  Runtime auto-disposes; user-level avals (`cval` / `clist` /
  `cset`) survive; caller constructs a fresh Runtime from a new
  device and re-compiles the same `alist<Command>`. Per-device
  caches use `WeakMap<GPUDevice, …>` so they re-key naturally.
  The mock now exposes a manually-resolvable `device.lost` via
  `MockGPU.simulateLost(...)` for the auto-dispose test.
