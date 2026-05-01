// Hello-triangle example — uses the wombat.shader Vite plugin's
// inline `vertex(...) / fragment(...)` marker workflow. The
// plugin transforms each marker call into a build-time
// `__wombat_stage(...)` expression that produces a real `Effect`
// at runtime — no `parseShader → stage` plumbing in user code.
//
// To run:
//   cd examples/hello-triangle
//   npm install
//   npm run dev          # serves http://localhost:5174/

import {
  AList,
  HashMap,
  cval,
  transact,
  type aval,
} from "@aardworx/wombat.adaptive";
import { V2f, V3f, V4f } from "@aardworx/wombat.base";
import { effect, fragment, vertex } from "@aardworx/wombat.shader";
import {
  IBuffer,
  RenderTree,
  type BufferView,
  type Command,
  type DrawCall,
  type RenderObject,
  PipelineState,
} from "@aardworx/wombat.rendering/core";
import { Runtime } from "@aardworx/wombat.rendering/runtime";
import { attachCanvas, runFrame } from "@aardworx/wombat.rendering/window";

// The Vite plugin uses the TS type-checker to recover types — any
// of these compose:
//   - lambda parameter / return annotations
//   - marker generic args (`vertex<I, O>(...)`)
//   - bare `V4f` return → `gl_Position` for vertex,
//     `outColor` for fragment
//
// Below: only the input record is annotated; the return type is
// inferred from the lambda body, and the fragment uses bare V4f.
const helloTriangleEffect = effect(
  vertex<{ a_position: V2f; a_color: V3f }>(input => ({
    gl_Position: new V4f(input.a_position.x, input.a_position.y, 0.0, 1.0),
    v_color: input.a_color,
  })),
  fragment<{ v_color: V3f }>(input =>
    new V4f(input.v_color.x, input.v_color.y, input.v_color.z, 1.0),
  ),
);

const status = document.getElementById("status")!;

async function main() {
  if (!("gpu" in navigator)) { status.textContent = "WebGPU not available."; return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { status.textContent = "no GPUAdapter"; return; }
  const device = await adapter.requestDevice();
  device.onuncapturederror = (e) => {
    console.error("GPU error:", e.error.message);
    status.textContent = "GPU error: " + e.error.message;
  };

  const canvas = document.getElementById("gpu") as HTMLCanvasElement;
  const win = attachCanvas(device, canvas, {
    colorAttachmentName: "outColor",
    format: navigator.gpu.getPreferredCanvasFormat(),
  });

  const positions = new Float32Array([-0.6, -0.6, 0.6, -0.6, 0.0, 0.6]);
  const colors    = new Float32Array([1, 0, 0,  0, 1, 0,  0, 0, 1]);

  const obj: RenderObject = {
    effect: helloTriangleEffect,
    pipelineState: PipelineState.constant({ rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } }),
    vertexAttributes: HashMap.empty<string, aval<BufferView>>()
      .add("a_position", cval<BufferView>({
        buffer: IBuffer.fromHost(positions), offset: 0, count: 3, stride: 8, format: "float32x2",
      }))
      .add("a_color", cval<BufferView>({
        buffer: IBuffer.fromHost(colors), offset: 0, count: 3, stride: 12, format: "float32x3",
      })),
    uniforms: HashMap.empty(),
    textures: HashMap.empty(),
    samplers: HashMap.empty(),
    drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
  };

  const runtime = new Runtime({ device });
  const clear = cval(new V4f(0.05, 0.05, 0.07, 1));
  const t0 = performance.now();

  runFrame(win, (token, info) => {
    const t = (info.timestampMs - t0) / 1000;
    transact(() => {
      clear.value = new V4f(
        0.1 + 0.05 * Math.sin(t),
        0.1 + 0.05 * Math.sin(t + 2),
        0.15 + 0.05 * Math.sin(t + 4),
        1,
      );
    });
    runtime.compile(AList.ofArray<Command>([
      { kind: "Clear",  output: win.framebuffer, values: { colors: HashMap.empty<string, V4f>().add("outColor", clear.value) } },
      { kind: "Render", output: win.framebuffer, tree: RenderTree.leaf(obj) },
    ])).run(token);

    if (info.frame === 0) {
      status.textContent = "rendering on " + adapter.info.architecture + " (" + adapter.info.vendor + ")";
    }
  });
}

main().catch(err => {
  console.error(err);
  status.textContent = "init error: " + (err as Error).message;
});
