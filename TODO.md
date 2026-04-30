# wombat.rendering — deferred work

What's NOT yet implemented in v0.1, in rough priority order.

## Resource layer (packages/resources)

- **Storage buffers** lowering uses `prepareAdaptiveBuffer` per source
  `aval<IBuffer>`. A fast path that lets compute-produced
  `AdaptiveResource<GPUBuffer>` flow directly in (no extra wrapping)
  would avoid a redundant indirection.
- **Instance attributes.** `RenderObject.instanceAttributes` is
  defined in core but ignored in `prepareRenderObject`. Needs per-
  attribute `stepMode: "instance"` in the vertex layout.
- **Blend / depth-stencil / multisample state.** `PipelineState.blends`
  and `PipelineState.stencil` are read into the pipeline desc only
  for `depth.write` / `depth.compare`. Blend per-attachment, stencil
  ops, and `multisample.alphaToCoverageEnabled` need to flow through.
- **`BufferView.indexFormat`.** `BufferView.format` is `GPUVertexFormat | GPUIndexFormat`,
  but `prepareRenderObject` currently hardcodes `uint32` for the
  index buffer because we can't read the aval synchronously at
  prep time. Either add a sibling `indexFormat?: GPUIndexFormat`
  to `BufferView`, or deprecate the dual format and split the type.
- **Mip-map generation.** `prepareAdaptiveTexture` allocates
  `mipLevelCount` slots when `generateMips: true` is passed, but
  only uploads mip 0. Real mip generation needs a compute pass
  (Dawn / Three.js have reference impls).
- ~~**`bytesPerPixelFor` table**~~. DONE: replaced with
  `blockInfoFor(format)` returning `{ width, height, bytesPerBlock }`.
  Linear formats use 1×1 blocks; BC1–BC7, ETC2/EAC, and the full
  ASTC range (4×4 through 12×12) covered. `writeTexture` builds
  bytesPerRow / rowsPerImage off the block grid.
- **Buffer alignment / staging.** Right now `queue.writeBuffer` is
  used unconditionally. For very large uploads or per-frame compute
  outputs, a staging-buffer + `copyBufferToBuffer` path is cheaper.
  Decide later based on profiling.
- **Shader-module sharing across pipelines.** `compileRenderPipeline`
  caches modules per-source string, but two pipelines compiled from
  the same `CompiledEffect` will share the module via that path.
  Verify the cache key picks up correctly.

## Runtime layer (packages/runtime)

- ~~**PreparedRenderObject cache key**.~~ DONE: keyed on
  `(RenderObject, FramebufferSignature)`; pipelines additionally
  use `Effect.id` (wombat.shader build-time stable hash) instead
  of FNV-hashing shader source.
- **Adaptive subscription.** `RenderTask.run(token)` walks the
  command alist on every call, reading every aval through `token`.
  Correct, but unoptimised — re-walks the tree even when nothing
  changed. Future: make `RenderTask` itself an `AdaptiveObject`
  that subscribes only to alist deltas, with cached prepared-leaves.
- ~~**Unordered reordering**.~~ DONE: `Unordered` /
  `UnorderedFromSet` children sort by pipeline (then by group-0
  layout) before encoding. Reduces state-change boundaries.
- ~~**Pass coalescing**.~~ DONE: adjacent `Clear` + `Render` on
  the same FBO aval fuse into one render pass with `loadOp:"clear"`
  attachments.
- ~~**`device.lost` + recovery**.~~ DONE (basic): `Runtime` now
  subscribes to `device.lost` and fires `disposeAll()` when the
  promise resolves. `Runtime.isDeviceLost` + `Runtime.deviceLost`
  are observable. **Not** done: rebuilding all `AdaptiveResource`s
  on a fresh device (which is the actual recovery story); for now
  the user has to construct a new `Runtime` from a new device.
- **Multiple queues / async submission.** Single queue, sync
  submit per `run()`. For async readback or upload, callers go
  through `Custom`.

## Window / browser integration (packages/window — NOT YET CREATED)

- Canvas attachment via `GPUCanvasContext.configure()`.
- `ResizeObserver`-driven swap-chain resize.
- DPR-aware viewport.
- `requestAnimationFrame`-driven loop integrated with
  wombat.adaptive's transaction model so each frame samples a
  coherent state.
- Optional input streaming as `aval<MouseState>` etc.

## Shader integration

- ~~**Real `compileEffect`**.~~ DONE: `core/shader.ts` re-exports
  the real `Effect` / `CompiledEffect` / `ProgramInterface` from
  `@aardworx/wombat.shader-runtime`; `Runtime` defaults to
  `effect.compile({ target: "wgsl" })`; integration test in
  `tests/shader-integration.test.ts` runs full source → WGSL →
  pipeline.
- ~~**Effect IDs as cache keys**.~~ DONE: `compileRenderPipeline`
  accepts `effectId`; `prepareRenderObject` threads `Effect.id`
  through. FNV-hash fallback only for hand-built effects without
  an upstream Effect.
- **Source-map plumbing.** `CompiledStage.sourceMap` is now
  re-exported but not yet surfaced in the rendering layer for
  shader-compile error reporting.
- **Vite plugin examples.** Real users use the
  `vertex(...) / fragment(...)` markers + `@aardworx/wombat.shader-vite`
  plugin to get an `Effect` directly. No example yet.

## Examples / tooling

- No examples yet (`packages/examples/hello-cube`,
  `camera-controller`, `instanced-grid`, `post-processing`,
  `compute-readback`).
- ~~No browser smoke test infra.~~ DONE: `tests-browser/`
  runs under vitest browser-mode + Playwright using the
  **system Chromium** (`/usr/bin/chromium` via `executablePath`)
  with Vulkan flags. Picks the real GPU (NVIDIA RTX 5060 /
  Blackwell on this host) instead of SwiftShader. Pixel-level
  validation via `copyTextureToBuffer` + map readback.
- ~~`Runtime.disposeAll()`~~ — DONE. Tracks every task compiled
  via `Runtime.compile()` / `Runtime.renderTo()` in a registry;
  `disposeAll()` releases each task and blocks future compiles.
  Auto-fires from the `device.lost` handler.

## Upstream wombat.shader bug

- **`liftReturns` not effective via `parseShader → stage(...)` path.**
  When using compileShaderSource, `liftReturns` rewrites bare
  `return new V4f(...)` into a `WriteOutput("outColor", ...)`. But
  going through `parseShader` + `stage(module).compile()` doesn't
  apply the lift — the resulting WGSL has `return vec4<f32>(...)`
  with the wrong return type vs `FsMainOutput`, causing
  `Error while parsing WGSL: return statement type must match its
  function return type, returned 'vec4<f32>', expected 'FsMainOutput'`.
  Workaround: declare the fragment function with an explicit struct
  return (`function fsMain(...): { outColor: V4f } { return { outColor: ... } }`).
  Likely the carrier annotation that liftReturns matches on is
  non-enumerable and gets stripped when stage() spreads the values.
  File against wombat.shader.

## Type-level loose ends

- ~~`Effect` placeholder.~~ DONE: replaced with real type.
- ~~`ProgramInterface` minimum-viable subset.~~ DONE: real
  re-export.
