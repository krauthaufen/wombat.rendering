# @aardworx/wombat.rendering

WebGPU rendering layer for the Wombat TypeScript stack — port of
[`Aardvark.Rendering`](https://github.com/aardvark-platform/aardvark.rendering)'s
lower layers (`RenderObject`, `RenderTask`, runtime, window/loop) on
top of WebGPU.

Part of the Wombat TypeScript port of the Aardvark stack:

1. [`@aardworx/wombat.adaptive`](https://github.com/krauthaufen/wombat.adaptive) — incremental adaptive computations (`aval`/`aset`/`alist`/`amap`).
2. [`@aardworx/wombat.base`](https://github.com/krauthaufen/wombat.base) — math/geometry primitives.
3. [`@aardworx/wombat.shader`](https://github.com/krauthaufen/wombat.shader) — TS-as-shader DSL with WGSL/GLSL emit.
4. **`@aardworx/wombat.rendering`** — this repo: RenderObject + RenderTask + window on top of WebGPU and wombat.shader.

A scenegraph layer (declarative scene → RenderObjects) lives outside
this repo.

## Install

```bash
npm install @aardworx/wombat.rendering @aardworx/wombat.shader \
            @aardworx/wombat.adaptive @aardworx/wombat.base
# build-time, for the inline-shader marker workflow:
npm install -D @aardworx/wombat.shader-vite
```

ESM only. Browser (WebGPU). Node ≥ 18 for tooling.

## Quick start

```ts
import { effect, vertex, fragment } from "@aardworx/wombat.shader";
import { V2f, V3f, V4f } from "@aardworx/wombat.base";
import { AList, HashMap, cval } from "@aardworx/wombat.adaptive";
import {
  Runtime, attachCanvas, runFrame,
  RenderTree, IBuffer,
  type Command, type RenderObject, type DrawCall, type BufferView,
} from "@aardworx/wombat.rendering";

const helloTriangle = effect(
  vertex<{ a_position: V2f; a_color: V3f }>(input => ({
    gl_Position: new V4f(input.a_position.x, input.a_position.y, 0.0, 1.0),
    v_color: input.a_color,
  })),
  fragment<{ v_color: V3f }>(input =>
    new V4f(input.v_color.x, input.v_color.y, input.v_color.z, 1.0),
  ),
);

async function main() {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter!.requestDevice();
  const canvas = document.querySelector("canvas")!;
  const win = attachCanvas(device, canvas);
  const runtime = new Runtime({ device });

  const positions = new Float32Array([-0.7, -0.7,  0.7, -0.7,  0,  0.7]);
  const colors    = new Float32Array([1, 0, 0,  0, 1, 0,  0, 0, 1]);
  const obj: RenderObject = {
    effect: helloTriangle,
    pipelineState: { rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } },
    vertexAttributes: HashMap.empty<string, BufferView>()
      .add("a_position", { buffer: IBuffer.fromHost(positions), offset: 0, count: 3, stride: 8,  format: "float32x2" })
      .add("a_color",    { buffer: IBuffer.fromHost(colors),    offset: 0, count: 3, stride: 12, format: "float32x3" }),
    uniforms: HashMap.empty(),
    textures: HashMap.empty(),
    samplers: HashMap.empty(),
    drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
  };

  const commands: Command[] = [{ kind: "Render", output: win.framebuffer, tree: RenderTree.leaf(obj) }];
  const task = runtime.compile(AList.ofList(commands));
  runFrame(task, win);
}
main();
```

## Module map

| Subpath | Contents |
| --- | --- |
| `@aardworx/wombat.rendering` | full surface — most consumers import from here |
| `@aardworx/wombat.rendering/core` | types only — `RenderObject`, `RenderTree`, `Command`, `IBuffer`/`ITexture`/`ISampler`, `AdaptiveResource`, `FramebufferSignature` |
| `@aardworx/wombat.rendering/resources` | per-device resource preparers — `prepareAdaptiveBuffer`/`Texture`/`Sampler`, `prepareUniformBuffer`, `compileRenderPipeline`, `prepareRenderObject`, `prepareComputeShader`, `allocateFramebuffer`, `generateMips`, `installShaderDiagnostics`, source-map decoder |
| `@aardworx/wombat.rendering/commands` | command-stream functions on `GPUCommandEncoder` — `clear`, `render`, `renderMany`, `beginPassDescriptor` |
| `@aardworx/wombat.rendering/runtime` | `Runtime`, `compileRenderTask`, `ScenePass` (delta-driven walker), `renderTo`, `copy` |
| `@aardworx/wombat.rendering/window` | `attachCanvas` + `runFrame` (rAF loop integrated with adaptive transactions) |

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
  `wombat.shader` `Effect`. `Effect.id` keys the pipeline cache;
  `(RenderObject, FramebufferSignature)` is the prep cache key.
- **`renderTo(scene, opts) → aval<ITexture>`.** Aardvark's
  `RenderTask.renderTo` ported. Acquiring the returned aval brings
  a hidden FBO + render task live; last release tears them down.
  Multi-attachment renders use `result.colors()` to bind the whole
  G-buffer at once.
- **Imperative compute.** `prepareComputeShader(device, shader)`
  exposes Aardvark.GPGPU's mutable `ComputeInputBinding` —
  `setUniform`/`setBuffer`/`setTexture`/`setStorageTexture` by
  name, then `dispatch(binding, groups)`. No alist or walker
  involvement.
- **MSAA.** Set `signature.sampleCount > 1` (or
  `attachCanvas({ sampleCount: 4 })`); `allocateFramebuffer`
  allocates multisample color/depth + single-sample resolve
  targets, the pass descriptor wires `resolveTarget`
  automatically.
- **Walker optimisations:** Clear+Render coalescing, Unordered
  state-aware sort, bind-group cache keyed on resolved handle
  identity, `(effect, signature)` pipeline cache key, real-GPU
  validated.

## Status

v0.1, working end-to-end on real hardware.

- 57 mock tests + 17 real-GPU tests = 74 passing.
- Real-GPU tests run under `vitest --browser` + Playwright with the
  system Chromium picking the actual GPU via Vulkan
  (validated on NVIDIA RTX 5060 / Blackwell).
- `examples/hello-triangle` renders a coloured triangle in real
  Chromium via the `@aardworx/wombat.shader-vite` inline-marker
  workflow.

See [`TODO.md`](TODO.md) for what's still deferred.

## Build & test

```sh
npm install
npm run build         # tsc -b
npm test              # 57 mock-GPU tests, ~2 s
npm run test:browser  # 17 real-GPU tests, ~4 s — needs Vulkan + system Chromium
npm run test:all      # both
```

`npm run test:browser` uses your system Chromium (configured in
`vitest.browser.config.ts`); the Playwright-bundled headless shell
falls back to SwiftShader because it lacks a compositor.

## License

MIT.
