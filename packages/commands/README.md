# @aardworx/wombat.rendering-commands

Layer 3 of [wombat.rendering](https://github.com/krauthaufen/wombat.rendering) —
command-stream functions on a `GPUCommandEncoder`. Stateless; takes an
encoder + already-prepared resources and emits draws / clears / copies.

Exports:

- `clear` — issues a render pass that clears the bound attachments.
- `render` — encodes a single `PreparedRenderObject`.
- `renderMany` — encodes a batch with state-aware sort to minimise
  pipeline / bind-group churn.
- `beginPassDescriptor` — builds a `GPURenderPassDescriptor` from a
  `FramebufferSignature` + clear values.

Higher-level orchestration (delta-driven walking, render task
compilation, `renderTo`) lives in
[`@aardworx/wombat.rendering-runtime`](../runtime).
