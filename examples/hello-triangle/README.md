# hello-triangle

Minimal browser example showing the full wombat.rendering stack:

- `attachCanvas` + `runFrame` from `@aardworx/wombat.rendering-window`
  for canvas binding + a `requestAnimationFrame` loop.
- A real `Effect` built via `parseShader` + `stage` from
  `@aardworx/wombat.shader-runtime` (the inline `vertex(...)` /
  `fragment(...)` Vite-plugin marker workflow is a deferred
  follow-up).
- A `RenderObject` with two vertex attributes (position + color)
  and a `Clear` + `Render` command list driven by the runtime.
- A `cval` cycling the clear color over time, demonstrating the
  reactive frame loop.

## Run

```sh
cd examples/hello-triangle
npm install
npm run dev          # serves http://localhost:5174/
```

Open the URL in any browser with WebGPU enabled (Chrome 113+ /
Edge with hardware acceleration).

## Headless verification

`check.mjs` is a Playwright script that:
1. Launches your system Chromium (`/usr/bin/chromium`) with the
   Vulkan + WebGPU flags so it picks the real GPU instead of
   SwiftShader.
2. Loads the dev server, lets a few rAF ticks run, and saves
   `canvas.png` (raw `toDataURL` from the canvas — works under
   headless) plus `screenshot.png` (Playwright screenshot — note
   that headless Chromium's compositor does not pull WebGPU
   canvas pixels into the screenshot, so this image will be
   black even when rendering succeeds; see `canvas.png` for the
   actual output).

```sh
node check.mjs
```

Expected output:

- `status: rendering on blackwell (nvidia)` (or whatever your
  adapter is)
- `canvas.png` shows a per-vertex-coloured triangle (red bottom
  left, green bottom right, blue top apex) on a dim purple
  background.
