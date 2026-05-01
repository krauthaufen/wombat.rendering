# @aardworx/wombat.rendering-core

Layer 1 of [wombat.rendering](https://github.com/krauthaufen/wombat.rendering) —
types only. No WebGPU calls live here.

Defines:

- `IBuffer`, `ITexture`, `ISampler` — handle unions
  (`{kind:"gpu",...} | {kind:"host",...}`) the upper layers know how to
  prepare into real `GPU*` resources.
- `AdaptiveResource<T>` extends `aval<T>` — adaptive resource with
  explicit `acquire`/`release` ref-counting (Aardvark's
  `IResourceLocation<T>` story).
- `RenderObject`, `RenderTree`, `Command` — the scene description.
- `FramebufferSignature`, `RenderContext` — render-pass identity +
  per-frame context.
- Re-exports `Effect`, `CompiledEffect`, `ProgramInterface` from
  `@aardworx/wombat.shader-runtime`.

This package is consumed by every other layer; downstream packages
never reach into one another's internals.
