# @aardworx/wombat.rendering-window

Layer 5 of [wombat.rendering](https://github.com/krauthaufen/wombat.rendering) —
canvas + main loop integration.

Exports:

- `attachCanvas(runtime, canvas, opts)` — configures the
  `GPUCanvasContext`, picks a presentation format compatible with the
  runtime's `FramebufferSignature`, and returns an
  `aval<IFramebuffer>` for the current swap-chain texture (re-fires on
  resize).
- `runFrame(runtime, render)` — `requestAnimationFrame` loop wrapped
  in a `wombat.adaptive` transaction so per-frame `cval` updates are
  batched.

Pure browser glue; nothing here knows about `RenderObject` or
`RenderTask`.
