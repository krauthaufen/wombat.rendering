// Real-GPU regression: a draw whose vertex data and draw call change size after
// it is first drawn (FableCube dogfood/poetry.md: a mesh that starts empty and is
// filled later; a buffer that grows draws only its first vertices). The heap
// adapter read the draw call once, so the record kept its first count (0 for an
// empty mesh: never drawn). Covers 0 → N, growth and shrink, non-indexed and
// indexed, on the heap path (a draw call marked heap-safe, as wombat.dom marks a
// mesh whose data changes) and on the legacy path (an unmarked changing one).

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import {
  IBuffer, ElementType, PipelineState, RenderTree,
  type BufferView, type DrawCall, type RenderObject,
} from "@aardworx/wombat.rendering/core";
import { AttributeProvider, UniformProvider } from "../packages/rendering/src/core/provider.js";
import { compileHybridScene } from "../packages/rendering/src/runtime/hybridScene.js";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";
import { makeHeapTestEffect } from "../tests/_heapTestEffect.js";

const W = 40, H = 8;
/** Quad k of `n` columns as two triangles (6 vertices, or 4 + 6 indices). */
function quadVerts(k: number, n: number): number[] {
  const x0 = -1 + (2 * k) / n, x1 = -1 + (2 * (k + 1)) / n;
  return [x0, -1, 0, x1, -1, 0, x0, 1, 0, x0, 1, 0, x1, -1, 0, x1, 1, 0];
}

interface Geo { positions: Float32Array; indices?: Uint32Array }
/** The first `quads` of 4 columns. */
function geo(quads: number, indexed: boolean): Geo {
  if (!indexed) return { positions: new Float32Array(Array.from({ length: quads }, (_, k) => quadVerts(k, 4)).flat()) };
  const pos: number[] = [], idx: number[] = [];
  for (let k = 0; k < quads; k++) {
    const x0 = -1 + (2 * k) / 4, x1 = -1 + (2 * (k + 1)) / 4, b = k * 4;
    pos.push(x0, -1, 0, x1, -1, 0, x0, 1, 0, x1, 1, 0);
    idx.push(b, b + 1, b + 2, b + 2, b + 1, b + 3);
  }
  return { positions: new Float32Array(pos), indices: new Uint32Array(idx) };
}

function ro(g: aval<Geo>, indexed: boolean, heapSafe: boolean): RenderObject {
  const view = (f: (x: Geo) => Float32Array): BufferView =>
    ({ buffer: g.map((x) => IBuffer.fromHost(f(x).buffer)), offset: 0, stride: 12, elementType: ElementType.V3f });
  const drawCall = g.map((x): DrawCall => indexed
    ? { kind: "indexed", indexCount: x.indices!.length, instanceCount: 1, firstIndex: 0, baseVertex: 0, firstInstance: 0 }
    : { kind: "non-indexed", vertexCount: x.positions.length / 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 });
  // as wombat.dom marks a mesh whose data changes (markHeapSafeDrawAVal)
  if (heapSafe) (drawCall as { __sgHeapSafeDraw?: { kind: string } }).__sgHeapSafeDraw = { kind: indexed ? "indexed" : "non-indexed" };
  return {
    effect: makeHeapTestEffect(),
    pipelineState: PipelineState.constant({
      rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" },
      depth: { write: true, compare: "less-equal" },
    }),
    vertexAttributes: AttributeProvider.ofObject({ Positions: view((x) => x.positions), Normals: view((x) => new Float32Array(x.positions.length)) }),
    uniforms: UniformProvider.ofObject({
      ModelTrafo: AVal.constant(Trafo3d.identity),
      ViewProjTrafo: AVal.constant(Trafo3d.identity),
      Color: AVal.constant(new V4f(1, 0, 0, 1)),
      LightLocation: AVal.constant(new V3d(0, 0, 1)),
    }),
    textures: HashMap.empty(),
    samplers: HashMap.empty(),
    ...(indexed ? { indices: { buffer: g.map((x) => IBuffer.fromHost(x.indices!.buffer)), offset: 0, stride: 4, elementType: ElementType.U32 } } : {}),
    drawCall,
  } as unknown as RenderObject;
}

async function run(indexed: boolean, heapSafe: boolean) {
  const device = await requestRealDevice();
  const errors: GPUError[] = [];
  device.onuncapturederror = (e) => errors.push(e.error);
  try {
    const g = cval<Geo>(geo(0, indexed));
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
    const scene = compileHybridScene(device, sig, RenderTree.leaf(ro(g, indexed, heapSafe)), {});
    const colorTex = device.createTexture({ size: { width: W, height: H }, format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const depthTex = device.createTexture({ size: { width: W, height: H }, format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    // how many of the 4 columns are drawn (red)
    const columns = async (): Promise<number> => {
      scene.update(AdaptiveToken.top);
      const enc = device.createCommandEncoder();
      scene.encodeComputePrep(enc, AdaptiveToken.top);
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: colorTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
        depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
      });
      scene.encodeIntoPass(pass, AdaptiveToken.top);
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      if (errors.length > 0) throw new Error("GPU errors: " + errors.map((e) => (e as { message?: string }).message).join("; "));
      const px = await readTexturePixels(device, colorTex);
      let n = 0;
      for (let k = 0; k < 4; k++) { const x = Math.floor((k + 0.5) * W / 4); if (px[(4 * W + x) * 4]! > 100) n++; }
      return n;
    };
    const path = () => (scene.heapTotalDraws() > 0 ? "heap" : "legacy");
    const seen: [string, number][] = [];
    seen.push(["empty", await columns()]);
    for (const q of [2, 4, 1, 3]) {
      transact(() => { g.value = geo(q, indexed); });
      seen.push([`${q}`, await columns()]);
    }
    return { seen, path: path() };
  } finally { device.destroy(); }
}

describe("a draw whose vertex count changes — real GPU", () => {
  for (const indexed of [false, true]) {
    for (const heapSafe of [true, false]) {
      it(`${indexed ? "indexed" : "non-indexed"}, ${heapSafe ? "heap-safe (heap)" : "unmarked"}: 0 → 2 → 4 → 1 → 3 quads`, async () => {
        const { seen, path } = await run(indexed, heapSafe);
        if (heapSafe) expect(path).toBe("heap");
        expect(seen, `path ${path}`).toEqual([["empty", 0], ["2", 2], ["4", 4], ["1", 1], ["3", 3]]);
      });
    }
  }
});
