// Hello-triangle example. Demonstrates the wombat.rendering stack
// end-to-end in a browser:
//   - attachCanvas + runFrame from packages/window
//   - real wombat.shader effect via parseShader + stage
//   - RenderObject with two vertex attributes (position, color)
//   - Per-frame clear color cycling driven by a cval
//
// To run:
//   cd examples/hello-triangle
//   npm install
//   npm run dev
// Then open http://localhost:5174/ in a real browser (Chrome /
// Edge with WebGPU enabled). Headless Chromium via Playwright's
// `chromium_headless_shell` build does not present the swap-chain
// to a screenshot-able compositor, so verifying the example via
// `node check.mjs` will only show a black canvas. The underlying
// rendering works — see tests-browser/ for pixel-level checks
// on real GPU via copyTextureToBuffer readback.

import {
  AList,
  HashMap,
  cval,
  transact,
  type aval,
} from "@aardworx/wombat.adaptive";
import { V4f } from "@aardworx/wombat.base";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader-frontend";
import { stage } from "@aardworx/wombat.shader-runtime";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader-ir";
import {
  IBuffer,
  RenderTree,
  type BufferView,
  type Command,
  type DrawCall,
  type RenderObject,
} from "@aardworx/wombat.rendering-core";
import { Runtime } from "@aardworx/wombat.rendering-runtime";
import { attachCanvas, runFrame } from "@aardworx/wombat.rendering-window";

const Tvec2f: Type = Vec(Tf32, 2);
const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

function buildEffect() {
  const source = `
    function vsMain(input: { a_position: V2f; a_color: V3f }): { gl_Position: V4f; v_color: V3f } {
      return {
        gl_Position: new V4f(input.a_position.x, input.a_position.y, 0.0, 1.0),
        v_color: input.a_color,
      };
    }
    function fsMain(input: { v_color: V3f }): { outColor: V4f } {
      return { outColor: new V4f(input.v_color.x, input.v_color.y, input.v_color.z, 1.0) };
    }
  `;
  const entries: EntryRequest[] = [
    {
      name: "vsMain", stage: "vertex",
      inputs: [
        { name: "a_position", type: Tvec2f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "a_color",    type: Tvec3f, semantic: "Color",    decorations: [{ kind: "Location", value: 1 }] },
      ],
      outputs: [
        { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "v_color",     type: Tvec3f, semantic: "Color",    decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
    {
      name: "fsMain", stage: "fragment",
      inputs:  [{ name: "v_color",  type: Tvec3f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    },
  ];
  return stage(parseShader({ source, entries }));
}

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
    effect: buildEffect(),
    pipelineState: { rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } },
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
