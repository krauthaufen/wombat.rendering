# @aardworx/wombat.rendering-resources

Layer 2 of [wombat.rendering](https://github.com/krauthaufen/wombat.rendering) —
per-device resource preparation. Pure functions over a `GPUDevice`; no
encoder, no frame loop.

Exports:

- `prepareAdaptiveBuffer` / `prepareAdaptiveTexture` /
  `prepareAdaptiveSampler` — turn `aval<IBuffer|ITexture|ISampler>` into
  ref-counted `AdaptiveResource<GPUBuffer|GPUTexture|GPUSampler>`.
- `prepareUniformBuffer` — packs `aval<unknown>` into a UBO via the
  `wombat.shader` layout.
- `compileRenderPipeline` — `(Effect, FramebufferSignature) →
  GPURenderPipeline`, cached on `(effect.id, signature.id)`.
- `prepareRenderObject` — full lowering: vertex/instance attrs, UBO,
  textures, samplers, storage buffers; bind-group cache keyed on
  resolved handle identity.
- `allocateFramebuffer` — color + depth attachments matching a
  `FramebufferSignature`.
- `generateMips` — 2× box-filter compute pass; pipelines cached per
  `(device, format)`.
- `installShaderDiagnostics` — surfaces `getCompilationInfo()` warnings
  and errors against the supplied `SourceMap`.
