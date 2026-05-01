# @aardworx/wombat.rendering-runtime

Layer 4 of [wombat.rendering](https://github.com/krauthaufen/wombat.rendering) —
the runtime façade. Compiles `alist<Command>` into a delta-driven
executor and owns per-device caches and lifetimes.

Exports:

- `Runtime` — top-level entry point. Wraps a `GPUDevice`, hosts caches
  (pipeline, prepared-RO, bind-group), and disposes itself on
  `device.lost`.
- `compileRenderTask` — `alist<Command> → RenderTask`. Per `Render`
  command builds a persistent `ScenePass` so per-frame cost is
  `O(deltas) + O(emitted-leaves)`, not `O(tree-nodes)`.
- `ScenePass` + `NodeWalker` hierarchy — `EmptyWalker`, `LeafWalker`,
  `OrderedWalker`, `UnorderedWalker`, `AdaptiveWalker`,
  `OrderedFromListWalker` (uses `MapExt<Index, NodeWalker>`),
  `UnorderedFromSetWalker`. Subscriptions to `aval<RenderTree>` /
  `alist<RenderTree>` / `aset<RenderTree>` splice node-walkers in
  place.
- `renderTo(scene, opts) → aval<ITexture>` — Aardvark's
  `RenderTask.renderTo` ported. Acquiring the result brings a hidden
  FBO + render task live; last release tears them down.
- `copy` — buffer/texture-to-buffer/texture copies on the runtime's
  encoder.
- Clear+Render coalescing into a single render pass.
