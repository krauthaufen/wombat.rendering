# wombat.rendering

WebGPU rendering layer for the Wombat TypeScript stack. Port of
Aardvark.Rendering's lower layers (RenderObject / RenderTask /
runtime / window) on top of WebGPU + wombat.shader. One published
package: `@aardworx/wombat.rendering`.

The scenegraph layer (declarative scene → RenderObjects) lives
**outside** this repo. This package stops at the RenderObject /
ScenePass / Runtime boundary.

## Repository layout

```
packages/
└── rendering/   @aardworx/wombat.rendering
                 src/
                 ├── core/        types only — RenderObject, RenderTree,
                 │                Command, IBuffer/ITexture/ISampler,
                 │                AdaptiveResource, FramebufferSignature,
                 │                RenderContext, shader re-exports
                 ├── resources/   per-device preparers — prepareAdaptiveBuffer/
                 │                Texture/Sampler, prepareUniformBuffer,
                 │                compileRenderPipeline, prepareRenderObject,
                 │                prepareComputeShader, allocateFramebuffer,
                 │                generateMips, installShaderDiagnostics,
                 │                sourceMapDecoder
                 ├── commands/    per-encoder — clear, render, renderMany,
                 │                beginPassDescriptor
                 ├── runtime/     Runtime, compileRenderTask, ScenePass
                 │                (delta-driven walker), renderTo, copy
                 ├── window/      attachCanvas + runFrame
                 └── index.ts     re-exports everything

tests/         57 mock-GPU tests — invariants, cache hits, ordering, lifecycle
tests-browser/ 17 real-GPU tests — pixel correctness via copyTextureToBuffer
examples/
└── hello-triangle/   Vite + the wombat.shader plugin's inline marker workflow
```

Originally split into 5 packages (core / resources / commands /
runtime / window) — collapsed to 1 in 0.1.0. Internal imports inside
`packages/rendering/src/<sub>/` use relative paths
(`../core/index.js` etc.); the package's subpath exports surface
each subdir.

## Tooling

- `npm test` — vitest, 57 mock-GPU tests, ~2s.
- `npm run test:browser` — vitest browser mode, 17 real-GPU tests,
  ~4s. Needs Vulkan + system Chromium (configured in
  `vitest.browser.config.ts` with `executablePath: "/usr/bin/chromium"`).
  Playwright's bundled headless shell falls back to SwiftShader
  because it lacks a compositor.
- `npm run test:all` — both.
- `npm run typecheck` — `tsc -b --noEmit`.
- `npm run build` — `tsc -b`.

The real-GPU tests are validated on NVIDIA RTX 5060 / Blackwell via
Vulkan. `probe-webgpu.test.ts` dumps the adapter info on every
browser-test run.

## Architecture

```
RenderObject ──► prepareRenderObject ──► PreparedRenderObject (cached on
   alist<…>                          (resources)   (effect.id, signature))
       │
       │ wrapped in RenderTree (Leaf/Ordered/Unordered/…)
       ▼
   Command (Render/Clear/Copy/Custom)
       │
   alist<Command> ──► Runtime.compile ──► RenderTask
                            (runtime)             (delta-driven walker)
                            │
                            ▼
                       per-frame: walk deltas, encode pass(es), submit
```

- **core types**: pure interfaces. No WebGPU calls.
- **resources**: per-`GPUDevice` preparers. Each lifts an
  `aval<source>` into an `AdaptiveResource<GPU…>` with explicit
  `acquire`/`release` ref-counting (Aardvark's `IResourceLocation<T>`
  story). Pipeline / module / bind-group caches keyed on
  `(effectId, signature)` and resolved handle identity.
- **commands**: stateless functions on a `GPUCommandEncoder`. Used
  by both the runtime walker and `renderTo`.
- **runtime**: `Runtime` wraps a device + caches; `compileRenderTask`
  builds a `ScenePass` per `Render` command (persistent walker —
  per-frame cost is O(deltas) + O(emitted-leaves), not O(tree)).
- **window**: canvas + rAF loop integrated with adaptive
  transactions via `markFrame()`.

## ScenePass walker

Each `Render` command holds a persistent `ScenePass`. The walker
hierarchy mirrors the `RenderTree` shape:

- `EmptyWalker`, `LeafWalker`, `OrderedWalker`, `UnorderedWalker`
- `AdaptiveWalker` — subscribes to `aval<RenderTree>` and splices
  in a child node-walker on change.
- `OrderedFromListWalker` — subscribes to `alist<RenderTree>`,
  uses `MapExt<Index, NodeWalker>` so deltas in any position cost
  O(log n).
- `UnorderedFromSetWalker` — subscribes to `aset<RenderTree>`,
  emits leaves in arbitrary order (for the state-aware sort).

## Compute primitive

`prepareComputeShader(device, shader)` is **separate** from the
adaptive walker — Aardvark.GPGPU's pattern. The returned
`PreparedComputeShader` exposes `createInputBinding()` →
`ComputeInputBinding`, which is mutable and name-keyed:
`setUniform`, `setBuffer`, `setTexture`, `setStorageTexture`,
`setSampler`. Then `dispatch(binding, groups)` opens an encoder,
encodes the compute pass, submits, awaits — or `encode(encoder, …)`
for batched submission.

No `Compute` command in the alist. The runtime is rendering-only.

## MSAA

`signature.sampleCount > 1` triggers the multisample path in
`allocateFramebuffer`: each color attachment becomes a multisample
texture (RENDER_ATTACHMENT only) plus a single-sample resolve
target (RENDER_ATTACHMENT | TEXTURE_BINDING). `IFramebuffer.colors`
is the multisample view, `IFramebuffer.resolveColors` is the
resolve view. `beginPassDescriptor` wires `resolveTarget`
automatically.

`attachCanvas({ sampleCount: 4 })` does the same for the canvas
swap chain — hidden multisample target, swap-chain texture as
resolve.

## Source maps

Stage compile errors are forwarded by `installShaderDiagnostics`
(via `getCompilationInfo()`) and decoded against
wombat.shader's v3 source maps via `decodePosition(map, line, col)`.
Multi-segment maps from the WGSL emitter give per-Expr granularity
inside lines.

## Lifecycle

- `Runtime` auto-disposes when `device.lost` resolves; user-level
  avals (`cval`/`clist`/`cset`) survive. Recovery = construct a
  new Runtime from a new device and re-compile the same
  `alist<Command>`. Per-device caches use `WeakMap<GPUDevice, …>`
  so they re-key naturally.
- `MockGPU.simulateLost(...)` exposes a manually-resolvable
  `device.lost` for the auto-dispose test.
- `AdaptiveResource<T>.acquire()`/`release()` — ref-counted; first
  acquire calls `create()`, last release calls `destroy()`. The
  `derive()` helper produces a child resource that forwards
  acquire/release to its parent.

## Test imports

Tests import from one of two namespaces depending on what they need:

- **Public API tests** (e.g. `tests/render-attribute-reactivity.test.ts`):
  import from `@aardworx/wombat.rendering.experimental` (note the
  `.experimental` suffix — this is the workspace alias defined in
  `package.json`'s `dependencies`/`exports` map). Only symbols
  re-exported via the package's subpath exports are reachable.
  Subpaths: `/core`, `/resources`, `/commands`, `/runtime`, `/window`.
- **Internal tests** that need symbols not yet on the public surface
  (e.g. `tests/heap-atlas-mip.test.ts` importing `AtlasPool`,
  `tests/heap-eligibility.test.ts` importing `isHeapEligible`):
  import via direct relative paths from the test file:
  `../packages/rendering/src/core/<file>.js`,
  `../packages/rendering/src/runtime/<file>.js`. Use the `.js`
  extension; the TS resolver maps it back to the `.ts` source.

For browser-mode tests in `tests-browser/`, the same rules apply
relative to that directory.

When adding a new test, prefer the public-API path if the symbol
is exported. Otherwise reach into `../packages/rendering/src/...`
directly. Don't try to import from `@aardworx/wombat.rendering`
without the `.experimental` suffix — that resolves to the published
npm package, which is older than the workspace source and missing
work-in-progress symbols.

The internal-import convention is also why `bunx vitest run` works
without npm-publishing the workspace: vitest picks up the relative
paths via TypeScript's path resolution.

## Don'ts

- Don't add new packages or split this one. Subpath exports cover
  the granularity needs.
- Don't introduce a `Compute` command in the alist or wire
  `prepareComputeShader` into the walker. Compute is imperative on
  purpose (Aardvark.GPGPU shape).
- Don't change `Effect.id` semantics — the pipeline cache key
  depends on `(effect.id, signature)` being stable for unchanged
  shaders.
- Don't bypass `RenderContext.encoder` from inside an
  `AdaptiveResource.compute(token)` — that's the channel the
  runtime uses to thread the active encoder into resource
  preparation (e.g. mip generation, renderTo's inner pass).
- Don't `npm publish` from a dirty tree.
