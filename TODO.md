# wombat.rendering — deferred work

What's NOT yet implemented, in rough priority order. Items
removed once they're truly done — Phase B–H DONE markers have
been pruned; the live list below is the work that remains.

## Resource layer

- ~~**Mip-map generation**.~~ DONE (manual): `generateMips(device,
  encoder, texture)` records 2×2 box-filter compute dispatches
  for every mip pair below the base level. Pipelines are cached
  per (device, format). Real-GPU validation in
  `tests-browser/mipgen-real.test.ts`. *Auto-wiring* into
  `prepareAdaptiveTexture` (so a fresh upload regenerates mips
  without a user-issued `Custom` command) needs a two-phase
  resolve refactor (texture upload before render-pass open) —
  call `generateMips(...)` directly from a `Custom` command for
  now. Pre-existing TODO note updated.
- ~~**Storage-buffer fast path**.~~ Audited: the "double wrap"
  case (`prepareAdaptiveBuffer` over a user-provided
  `AdaptiveResource<GPUBuffer>`) costs one extra evaluation hop
  per frame, which is noise. `tryAcquire`/`tryRelease` already
  propagate through correctly. Closed without code change.
- ~~**Buffer staging path**.~~ Audited: `queue.writeBuffer` is
  already backed by an internal staging pool in Chrome / Dawn;
  beating it requires our own pool with explicit staging-buffer
  reuse, which is an optimisation we should drive from real
  profiling data not speculation. Closed without code change;
  reopen if profiling shows `writeBuffer` overhead.

## Runtime

- ~~**`device.lost` recovery**.~~ Documented: on lost, the
  Runtime auto-disposes; user-level avals (`cval` / `clist` /
  `cset`) survive; caller constructs a fresh Runtime from a new
  device and re-compiles the same `alist<Command>`. The
  per-device caches use `WeakMap<GPUDevice, …>` so they re-key
  naturally. A single-call `replaceDevice` isn't useful since
  every prepared object bakes the device in — discard + rebuild
  is the right path. JSDoc on `Runtime.deviceLost` explains it.

## Shader integration

- **Source-map surfacing.** `CompiledStage.sourceMap` is
  re-exported but not surfaced when WebGPU rejects the WGSL —
  errors point at emitted source, not the original TS.
- **Vite-plugin example.** No example uses the `vertex(...) /
  fragment(...)` marker workflow with `@aardworx/wombat.shader-vite`;
  `examples/hello-triangle` calls `parseShader → stage` at runtime.

## Upstream wombat.shader (all fixed in this round)

- ~~**`liftReturns` no-op for bare V4f returns**.~~ FIXED:
  `liftReturns` now handles bare-value returns when the entry has
  exactly one declared output. Vertex bare V4f → WriteOutput on
  the @builtin(position) target; fragment bare V4f → WriteOutput
  on the colour @location(0) target. Spans preserved through
  the rewrite so source maps continue to work.
- ~~**Inline plugin emitted `__wombat_*` entry-point names**.~~
  FIXED: WGSL forbids leading double underscores; renamed to
  `wombat_<marker>_<offset>`.
- ~~**Inline plugin only read parameter annotations**.~~ FIXED:
  uses the TS type-checker's `getContextualType` →
  `getCallSignatures` chain to recover the lambda's parameter +
  return types. Generic args (`vertex<I, O>(...)`), explicit
  lambda annotations, and body-driven inference all work, in any
  combination. `unknown` / `any` / `never` skipped; body
  return-expression type used as the final fallback.
  `vertex<I, O = unknown>` and `fragment` got default type-arg
  values so partial generic specification works.
- ~~**Bare-V4f returns produced no entry outputs in the
  inline-plugin path**.~~ FIXED: `withDerivedOutputs` now
  synthesises a default output — vertex → `gl_Position`
  (@builtin position), fragment → `outColor` (@location 0).

## Out-of-scope (recorded so they don't surprise us)

- **Multiple queues / async submission.** WebGPU's spec exposes
  one queue per `GPUDevice`. There is no portable way to do
  concurrent submission today; revisit if/when WebGPU gains it.
- **Mock-tested deltas vs real-GPU.** Mock tests assert
  implementation invariants (call-pattern, cache hits, ordering);
  real-GPU tests assert pixel correctness. Both kept.
