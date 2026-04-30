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
- **`bytesPerPixelFor` table** in `adaptiveTexture.ts` is minimal —
  extend as new formats are needed (block-compressed formats need a
  different code path entirely).
- **Buffer alignment / staging.** Right now `queue.writeBuffer` is
  used unconditionally. For very large uploads or per-frame compute
  outputs, a staging-buffer + `copyBufferToBuffer` path is cheaper.
  Decide later based on profiling.
- **Shader-module sharing across pipelines.** `compileRenderPipeline`
  caches modules per-source string, but two pipelines compiled from
  the same `CompiledEffect` will share the module via that path.
  Verify the cache key picks up correctly.

## Runtime layer (packages/runtime)

- **PreparedRenderObject cache key** is currently just the
  `RenderObject` reference. Should also key on
  `(effect-id, signature)` so the same RenderObject can render
  into different FBO shapes — pipeline color targets must match.
- **Adaptive subscription.** `RenderTask.run(token)` walks the
  command alist on every call, reading every aval through `token`.
  Correct, but unoptimised — re-walks the tree even when nothing
  changed. Future: make `RenderTask` itself an `AdaptiveObject`
  that subscribes only to alist deltas, with cached prepared-leaves.
- **Unordered reordering.** `Unordered` and `UnorderedFromSet`
  currently emit children in iteration order. Should sort by
  pipeline → bind-group → vertex-buffer to minimise state changes.
- **Pass coalescing.** Adjacent `Clear` + `Render` on the same FBO
  could fold into one render pass with the right `loadOp`s. Not
  done; v0.1 emits one pass per command.
- **`device.lost` + recovery.** The runtime currently has no
  lost-device handling. On lost device all `AdaptiveResource`s
  need rebuilding.
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

- **Real `compileEffect`** wired to `@aardworx/wombat.shader-runtime`.
  Today `Runtime` takes a `compileEffect: (Effect) => CompiledEffect`
  callback; tests use an identity passthrough that lets users put a
  hand-built `CompiledEffect` directly in `RenderObject.effect`.
- **Effect IDs as cache keys.** wombat.shader produces stable
  build-time IDs; once integrated, key the pipeline cache on
  `(effectId, signature.hash, pipelineState.hash)` instead of on
  shader source strings.
- **Source-map plumbing.** `CompiledEffect.stages[].sourceMap`
  is in the shader runtime but not surfaced in our minimal
  `CompiledEffect` placeholder type.

## Examples / tooling

- No examples yet (`packages/examples/hello-cube`,
  `camera-controller`, `instanced-grid`, `post-processing`,
  `compute-readback`).
- No browser smoke test infra. All current tests use a Node-side
  mock; verifying against real WebGPU needs either a Dawn-Node
  binding or a Playwright headless-Chromium pipeline.
- `Runtime.disposeAll()` and other lifetime niceties for the
  page-unload path.

## Type-level loose ends

- `Effect` type in `core/shader.ts` is a placeholder
  `{ readonly __wombatEffect: unique symbol } & object` — replace
  with the real wombat.shader Effect type.
- `CompiledEffect.stages[].stage` is currently typed as `string`;
  tighten to `"vertex" | "fragment" | "compute"`.
- `ProgramInterface` is a minimum-viable subset of the
  wombat.shader interface; align with the real shape once
  integrated.
