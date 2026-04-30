# wombat.rendering

WebGPU rendering layer for the Wombat TypeScript stack. Port of
`Aardvark.Rendering`'s lower layers (RenderObject / RenderTask /
runtime), with WebGPU as the v0.1 backend.

See `../wombat.rendering-plan.md` for the design plan.

## Status

v0.1 in progress. Vertical slice through the stack works under a
mock GPUDevice — see `TODO.md` for what's still missing.

## Workspace layout

- `packages/core` — types only (`RenderObject`, `RenderTree`,
  `Command`, `FramebufferSignature`, `IBuffer` / `ITexture` /
  `ISampler`, `AdaptiveResource`, `RenderContext`, …).
- `packages/resources` — layer 2: `prepareAdaptiveBuffer`,
  `prepareAdaptiveTexture`, `prepareAdaptiveSampler`,
  `prepareUniformBuffer`, `compileRenderPipeline`,
  `prepareRenderObject`, `allocateFramebuffer`,
  `createFramebufferSignature`.
- `packages/commands` — layer 3: `clear`, `render`, `renderMany`.
- `packages/runtime` — `Runtime` facade, `compileRenderTask`,
  `copy`.
- `packages/window` — canvas attach + render loop. *(planned)*

## Dev

```sh
npm install
npm run build
npm test
```
