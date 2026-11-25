// Real-GPU test for UNIFORM COLUMNS / TABLES (core/uniformColumn.ts).
//
// Three draws share one effect and one quad. Their ModelTrafo and Color
// come from rows of ONE table (a struct per row); the effect is the
// ordinary test effect and reads plain uniforms. The scene is built the
// way an application builds it: RenderObjects in a RenderTree, compiled
// by the hybrid scene — once on the heap, once on the classic path, with
// identical pixels expected:
//   - the heap stores the table once and points each draw at its row;
//   - the classic path reads each cell as the aval it is.
// Then: every value changes at once (one array, one mark), one draw moves
// to another row, the table grows, and a draw with a plain aval sits in
// the same bucket as the column-backed ones.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import {
  IBuffer, ElementType, PipelineState, RenderTree,
  type BufferView, type DrawCall, type RenderObject,
} from "@aardworx/wombat.rendering/core";
import { AttributeProvider, UniformProvider } from "../packages/rendering/src/core/provider.js";
import { tableLayout, uniformTable, uniformColumn } from "../packages/rendering/src/core/uniformColumn.js";
import { compileHybridScene } from "../packages/rendering/src/runtime/hybridScene.js";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";
import { makeHeapTestEffect, makeHeapTestEffectMvp } from "../tests/_heapTestEffect.js";
import { buildHeapScene, type HeapDrawSpec } from "../packages/rendering/src/runtime/heapScene.js";

const W = 30, H = 8;
const effect = makeHeapTestEffect();
const FIELDS = { ModelTrafo: "mat4x4<f32>", Color: "vec4<f32>" } as const;
const LAYOUT = tableLayout(FIELDS);
const ROW = LAYOUT.stride / 4;

/** Table data: per row a translation by `x` and a colour. */
function tableData(rows: readonly { x: number; rgb: readonly [number, number, number] }[]): Float32Array {
  const data = new Float32Array(rows.length * ROW);
  rows.forEach((r, i) => {
    Trafo3d.translation(new V3d(r.x, 0, 0)).forward.copyTo(data, i * ROW + LAYOUT.offsets.ModelTrafo / 4);
    data.set([r.rgb[0], r.rgb[1], r.rgb[2], 1], i * ROW + LAYOUT.offsets.Color / 4);
  });
  return data;
}

// the quad covers the LEFT third; a row's translation moves it to its third
const pos = new Float32Array([-1, -1, 0, -1 / 3, -1, 0, -1, 1, 0, -1, 1, 0, -1 / 3, -1, 0, -1 / 3, 1, 0]);
const positions: BufferView = { buffer: AVal.constant(IBuffer.fromHost(pos.buffer)), offset: 0, stride: 12, elementType: ElementType.V3f };
const quadRO = (uniforms: Record<string, aval<unknown>>, debugName: string, fx = effect): RenderObject => ({
  effect: fx,
  debugName,
  pipelineState: PipelineState.constant({
    rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" },
    depth: { write: true, compare: "less-equal" },
  }),
  vertexAttributes: AttributeProvider.ofObject({ Positions: positions }),
  uniforms: UniformProvider.ofObject({
    ViewProjTrafo: AVal.constant(Trafo3d.identity),
    LightLocation: AVal.constant(new V3d(0, 0, 1)),
    ...uniforms,
  }),
  textures: HashMap.empty(),
  samplers: HashMap.empty(),
  drawCall: AVal.constant<DrawCall>({ kind: "non-indexed", vertexCount: 6, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
} as unknown as RenderObject);

describe("uniform columns through the scene (real GPU)", () => {
  for (const heap of [true, false]) {
    it(`${heap ? "heap" : "classic"}: draws take their uniforms from rows of one table`, async () => {
      try { globalThis.localStorage?.clear(); } catch { /* no storage */ }
      const device = await requestRealDevice();
      const errors: GPUError[] = [];
      device.onuncapturederror = (e) => errors.push(e.error);
      try {
        const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
        const third = 2 / 3;
        const data = cval(tableData([
          { x: 0, rgb: [1, 0, 0] },           // row 0: left, red
          { x: third, rgb: [0, 1, 0] },       // row 1: middle, green
          { x: 2 * third, rgb: [0, 0, 1] },   // row 2: right, blue
          { x: 2 * third, rgb: [1, 1, 1] },   // row 3: right, white (unused at first)
        ]));
        const table = uniformTable(FIELDS, data);
        const lastRow = cval(2);
        const tree = RenderTree.ordered(
          RenderTree.leaf(quadRO(table.row(0), "left")),
          RenderTree.leaf(quadRO(table.row(1), "middle")),
          RenderTree.leaf(quadRO(table.row(lastRow), "right")),
        );
        const colorTex = device.createTexture({ size: { width: W, height: H }, format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
        const depthTex = device.createTexture({ size: { width: W, height: H }, format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
        const scene = compileHybridScene(device, sig, tree, { heapEnabled: AVal.constant(heap) });
        const render = async (): Promise<number[][]> => {
          scene.update(AdaptiveToken.top);
          const enc = device.createCommandEncoder();
          scene.encodeComputePrep(enc, AdaptiveToken.top);
          const rp = enc.beginRenderPass({
            colorAttachments: [{ view: colorTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
            depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
          });
          scene.encodeIntoPass(rp, AdaptiveToken.top);
          rp.end();
          device.queue.submit([enc.finish()]);
          await device.queue.onSubmittedWorkDone();
          const px = await readTexturePixels(device, colorTex);
          if (errors.length > 0) {
            throw new Error("GPU errors:\n  " + errors.map(e => (e as { message?: string }).message ?? String(e)).join("\n  "));
          }
          return [0, 1, 2].map((k) => { const o = (4 * W + Math.floor((k + 0.5) * W / 3)) * 4; return [px[o]!, px[o + 1]!, px[o + 2]!]; });
        };
        const R = [255, 0, 0], G = [0, 255, 0], B = [0, 0, 255], Y = [255, 255, 0], C = [0, 255, 255], M = [255, 0, 255], Wh = [255, 255, 255];

        expect(await render(), "each draw reads its own row").toEqual([R, G, B]);
        expect(scene.heapTotalDraws()).toBe(heap ? 3 : 0);

        // every value at once: one new array
        transact(() => { data.value = tableData([
          { x: 0, rgb: [1, 1, 0] }, { x: third, rgb: [0, 1, 1] }, { x: 2 * third, rgb: [1, 0, 1] }, { x: 2 * third, rgb: [1, 1, 1] },
        ]); });
        expect(await render(), "all rows changed together").toEqual([Y, C, M]);

        // one draw moves to another row
        transact(() => { lastRow.value = 3; });
        expect(await render(), "the right draw now reads row 3").toEqual([Y, C, Wh]);

        // rows swap places: positions come from the table too
        transact(() => { data.value = tableData([
          { x: third, rgb: [1, 0, 0] }, { x: 0, rgb: [0, 1, 0] }, { x: 0, rgb: [0, 0, 0] }, { x: 2 * third, rgb: [0, 0, 1] },
        ]); });
        expect(await render(), "the matrices follow as well").toEqual([G, R, B]);

        // the table grows (the allocation moves)
        transact(() => { data.value = tableData([
          { x: 0, rgb: [0, 0, 1] }, { x: third, rgb: [1, 0, 0] }, { x: 0, rgb: [0, 0, 0] }, { x: 2 * third, rgb: [0, 1, 0] },
          { x: 0, rgb: [0, 0, 0] }, { x: 0, rgb: [0, 0, 0] }, { x: 0, rgb: [0, 0, 0] }, { x: 0, rgb: [0, 0, 0] },
        ]); });
        expect(await render(), "after the table grew").toEqual([B, R, G]);
        expect(scene.heapTotalDraws(), "still on the heap").toBe(heap ? 3 : 0);
      } finally { device.destroy(); }
    });

    it(`${heap ? "heap" : "classic"}: column-backed and plain uniforms mix in one scene`, async () => {
      try { globalThis.localStorage?.clear(); } catch { /* no storage */ }
      const device = await requestRealDevice();
      const errors: GPUError[] = [];
      device.onuncapturederror = (e) => errors.push(e.error);
      try {
        const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
        const third = 2 / 3;
        // a plain column: one colour per row, nothing else
        const colors = uniformColumn("vec4<f32>", new Float32Array([1, 0, 0, 1,  0, 1, 0, 1]));
        const at = (x: number): aval<unknown> => AVal.constant(Trafo3d.translation(new V3d(x, 0, 0)));
        const tree = RenderTree.ordered(
          RenderTree.leaf(quadRO({ ModelTrafo: at(0), Color: colors.at(0) }, "left: colour from the column")),
          RenderTree.leaf(quadRO({ ModelTrafo: at(third), Color: colors.at(1) }, "middle: colour from the column")),
          RenderTree.leaf(quadRO({ ModelTrafo: at(2 * third), Color: AVal.constant(new V4f(0, 0, 1, 1)) }, "right: plain aval")),
        );
        const colorTex = device.createTexture({ size: { width: W, height: H }, format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
        const depthTex = device.createTexture({ size: { width: W, height: H }, format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
        const scene = compileHybridScene(device, sig, tree, { heapEnabled: AVal.constant(heap) });
        scene.update(AdaptiveToken.top);
        const enc = device.createCommandEncoder();
        scene.encodeComputePrep(enc, AdaptiveToken.top);
        const rp = enc.beginRenderPass({
          colorAttachments: [{ view: colorTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
          depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
        });
        scene.encodeIntoPass(rp, AdaptiveToken.top);
        rp.end();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        const px = await readTexturePixels(device, colorTex);
        if (errors.length > 0) {
          throw new Error("GPU errors:\n  " + errors.map(e => (e as { message?: string }).message ?? String(e)).join("\n  "));
        }
        const got = [0, 1, 2].map((k) => { const o = (4 * W + Math.floor((k + 0.5) * W / 3)) * 4; return [px[o]!, px[o + 1]!, px[o + 2]!]; });
        expect(got).toEqual([[255, 0, 0], [0, 255, 0], [0, 0, 255]]);
        expect(scene.heapTotalDraws()).toBe(heap ? 3 : 0);
      } finally { device.destroy(); }
    });
  }
});

/** Render a hybrid or heap scene once and sample the three thirds. */
async function thirds(
  device: GPUDevice,
  scene: { update(t: AdaptiveToken): void; encodeComputePrep(e: GPUCommandEncoder, t: AdaptiveToken): void; encodeIntoPass(p: GPURenderPassEncoder, t: AdaptiveToken): void },
): Promise<number[][]> {
  const colorTex = device.createTexture({ size: { width: W, height: H }, format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const depthTex = device.createTexture({ size: { width: W, height: H }, format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  scene.update(AdaptiveToken.top);
  const enc = device.createCommandEncoder();
  scene.encodeComputePrep(enc, AdaptiveToken.top);
  const rp = enc.beginRenderPass({
    colorAttachments: [{ view: colorTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
    depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
  });
  scene.encodeIntoPass(rp, AdaptiveToken.top);
  rp.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const px = await readTexturePixels(device, colorTex);
  return [0, 1, 2].map((k) => { const o = (4 * W + Math.floor((k + 0.5) * W / 3)) * 4; return [px[o]!, px[o + 1]!, px[o + 2]!]; });
}

describe("uniform columns: the harder corners (real GPU)", () => {
  // A draw's header points INSIDE the table's allocation. Compaction moves
  // allocations, so that interior address has to move with its table.
  it("column-backed draws survive an arena compaction", async () => {
    try { globalThis.localStorage?.clear(); } catch { /* no storage */ }
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
      const third = 2 / 3;
      const table = uniformTable(FIELDS, tableData([
        { x: 0, rgb: [1, 0, 0] }, { x: third, rgb: [0, 1, 0] }, { x: 2 * third, rgb: [0, 0, 1] },
      ]));
      const identity = AVal.constant(Trafo3d.identity);
      const scene = buildHeapScene(device, sig, [], { fragmentOutputLayout: { locations: new Map([["outColor", 0]]) } });
      // fillers first (low addresses), of varying size, drawn off screen
      const fillers: number[] = [];
      for (let j = 0; j < 6; j++) {
        const n = 3 + j * 2;
        fillers.push(scene.addDraw({
          effect,
          inputs: {
            Positions: new Float32Array(n * 3).fill(5), ModelTrafo: identity, ViewProjTrafo: identity,
            Color: AVal.constant(new V4f(0.1 * j, 0.2, 0.3, 1)), LightLocation: AVal.constant(new V3d(0, 0, 1)),
          },
          indices: new Uint32Array(n).map((_, i) => i % n),
        } as HeapDrawSpec));
      }
      for (let k = 0; k < 3; k++) {
        scene.addDraw({
          effect,
          inputs: { Positions: pos, ViewProjTrafo: identity, LightLocation: AVal.constant(new V3d(0, 0, 1)), ...table.row(k) },
          indices: new Uint32Array([0, 1, 2, 3, 4, 5]),
        } as HeapDrawSpec);
      }
      const before = await thirds(device, scene);
      expect(before).toEqual([[255, 0, 0], [0, 255, 0], [0, 0, 255]]);

      for (const id of fillers) scene.removeDraw(id);
      scene.update(AdaptiveToken.top);
      const dbg = (scene as unknown as { _debug: { forceCompact(): number; attrWasteBytes(): number } })._debug;
      expect(dbg.attrWasteBytes()).toBeGreaterThan(0);
      expect(dbg.forceCompact()).toBe(0);
      expect(scene.stats.compactions).toBeGreaterThan(0);

      expect(await thirds(device, scene), "after compaction").toEqual(before);
      expect(errors.map(e => (e as { message?: string }).message)).toEqual([]);
    } finally { device.destroy(); }
  });

  // A cell is an aval, so it also works where the heap cannot take the
  // shortcut: here ModelTrafo feeds the GPU-derived ModelViewProjTrafo.
  it("a column cell can feed a derived uniform", async () => {
    try { globalThis.localStorage?.clear(); } catch { /* no storage */ }
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
      const third = 2 / 3;
      const data = cval(tableData([
        { x: 0, rgb: [1, 0, 0] }, { x: third, rgb: [0, 1, 0] }, { x: 2 * third, rgb: [0, 0, 1] },
      ]));
      const table = uniformTable(FIELDS, data);
      const identity = AVal.constant(Trafo3d.identity);
      const mvp = makeHeapTestEffectMvp();
      const tree = RenderTree.ordered(...[0, 1, 2].map((k) =>
        RenderTree.leaf(quadRO({ ViewTrafo: identity, ProjTrafo: identity, ...table.row(k) }, `quad ${k}`, mvp))));
      const scene = compileHybridScene(device, sig, tree, { heapEnabled: AVal.constant(true) });
      expect(await thirds(device, scene)).toEqual([[255, 0, 0], [0, 255, 0], [0, 0, 255]]);
      expect(scene.heapTotalDraws()).toBe(3);
      // move the left and the middle quad onto each other's place
      transact(() => { data.value = tableData([
        { x: third, rgb: [1, 0, 0] }, { x: 0, rgb: [0, 1, 0] }, { x: 2 * third, rgb: [0, 0, 1] },
      ]); });
      expect(await thirds(device, scene), "the derived matrix follows the table").toEqual([[0, 255, 0], [255, 0, 0], [0, 0, 255]]);
      expect(errors.map(e => (e as { message?: string }).message)).toEqual([]);
    } finally { device.destroy(); }
  });
});
