# wombat.rendering

WebGPU rendering layer for the Wombat TypeScript stack — port of
`Aardvark.Rendering`'s lower layers (`RenderObject`, `RenderTask`,
runtime, window/loop) on top of WebGPU. Composes with
[`@aardworx/wombat.shader`](https://github.com/krauthaufen/wombat.shader)
for shader composition + WGSL emit and
[`@aardworx/wombat.adaptive`](https://github.com/krauthaufen/wombat.adaptive)
for incremental data flow.

## Status

v0.1, working end-to-end on real hardware:

- 51 mock tests + 14 real-GPU tests = 65 passing.
- Real-GPU tests run under `vitest --browser` + Playwright with the
  system Chromium picking the actual GPU via Vulkan
  (validated on NVIDIA RTX 5060 / Blackwell).
- `examples/hello-triangle` renders a coloured triangle in real
  Chromium via the `@aardworx/wombat.shader-vite` inline-marker
  workflow.

See [`TODO.md`](TODO.md) for what's still deferred.

## Workspace layout

```
packages/
├── core/        @aardworx/wombat.rendering-core
│                  Types only — RenderObject, RenderTree, Command,
│                  IBuffer / ITexture / ISampler, AdaptiveResource,
│                  FramebufferSignature, RenderContext.
├── resources/   @aardworx/wombat.rendering-resources
│                  Layer 2 (per-device): prepareAdaptiveBuffer /
│                  Texture / Sampler, prepareUniformBuffer,
│                  compileRenderPipeline, prepareRenderObject,
│                  allocateFramebuffer, generateMips,
│                  installShaderDiagnostics.
├── commands/    @aardworx/wombat.rendering-commands
│                  Layer 3 (per-encoder): clear, render,
│                  renderMany, beginPassDescriptor.
├── runtime/     @aardworx/wombat.rendering-runtime
│                  Runtime facade, compileRenderTask, ScenePass
│                  (delta-driven walker), renderTo, copy.
└── window/      @aardworx/wombat.rendering-window
                   attachCanvas + runFrame (rAF loop integrated
                   with wombat.adaptive transactions).

examples/
└── hello-triangle/  Vite + the wombat.shader plugin's inline
                     marker workflow. See its README for the
                     headless verification dance.

tests/         51 mock-GPU tests (call-pattern invariants,
               cache hits, ordering decisions, lifecycle).
tests-browser/ 14 real-GPU tests (pixel correctness via
               copyTextureToBuffer + map readback).
```

## Headline features

- **Adaptive everywhere.** `RenderObject`'s vertex / instance /
  uniform / texture / sampler / storage inputs are name-keyed
  `HashMap<string, aval<T>>`. `AdaptiveResource<T>` extends
  `aval<T>` so resources participate in the adaptive graph
  directly with explicit `acquire` / `release` ref-counting (the
  Aardvark `IResourceLocation<T>` story).
- **Delta-driven walker.** Each `Render` command holds a
  persistent `ScenePass` built once at compile time; per-frame
  cost is O(deltas)+O(emitted-leaves), not O(tree-nodes).
  Subscriptions to `aval<RenderTree>` / `alist<RenderTree>` /
  `aset<RenderTree>` splice node-walkers in place.
- **Real shader integration.** `RenderObject.effect` takes a
  `wombat.shader` `Effect`. `Effect.id` keys the pipeline cache
  (`(RenderObject, FramebufferSignature)` is the prep cache key).
- **`renderTo(scene, opts) → aval<ITexture>`.** Aardvark's
  `RenderTask.renderTo` ported. Acquiring the returned aval
  brings a hidden FBO + render task live; last release tears
  them down. Downstream RenderObjects bind it as a texture and
  the dependency graph encodes the data flow.
- **Walker optimisations:** Clear+Render coalescing, Unordered
  state-aware sort, bind-group cache keyed on resolved handle
  identity, `(effect, signature)` pipeline cache key, real-GPU
  validated.

## Dev

```sh
npm install
npm run build         # tsc -b across the workspace
npm test              # 51 mock-GPU tests, ~2 s
npm run test:browser  # 14 real-GPU tests, ~3 s — needs Vulkan
npm run test:all      # both
```

`npm run test:browser` uses your system Chromium (configured in
`vitest.browser.config.ts`); the Playwright-bundled headless shell
falls back to SwiftShader because it lacks a compositor.

## License

MIT.
