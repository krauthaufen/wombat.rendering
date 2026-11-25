// Real-GPU test for FRUSTUM CULLING on the heap path.
//
// A draw that supplies a bounding box (in the space of its Positions) is
// left out of a frame in which the box, taken through the draw's own
// ModelTrafo and ViewProjTrafo, is outside the view frustum: it stays
// resident, but its vertices are not emitted. A draw without a box is
// always drawn.
//
// What is checked is the number of vertices the megacall actually emits
// (the indirect draw's count, read back from the GPU), plus pixels.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import {
  IBuffer, ElementType, PipelineState, RenderTree,
  type BufferView, type DrawCall, type RenderObject,
} from "@aardworx/wombat.rendering/core";
import { AttributeProvider, UniformProvider } from "../packages/rendering/src/core/provider.js";
import { compileHybridScene } from "../packages/rendering/src/runtime/hybridScene.js";
import { buildHeapScene, type HeapDrawSpec, type BoundsLike } from "../packages/rendering/src/runtime/heapScene.js";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";
import { makeHeapTestEffect } from "../tests/_heapTestEffect.js";

const W = 30, H = 8;
const effect = makeHeapTestEffect();
// a quad over the LEFT third of the view; a draw's ModelTrafo moves it
const pos = new Float32Array([-1, -1, 0, -1 / 3, -1, 0, -1, 1, 0, -1, 1, 0, -1 / 3, -1, 0, -1 / 3, 1, 0]);
const quadBox: BoundsLike = { min: { x: -1, y: -1, z: 0 }, max: { x: -1 / 3, y: 1, z: 0 } };
const third = 2 / 3;
const shift = (x: number): aval<unknown> => AVal.constant(Trafo3d.translation(new V3d(x, 0, 0)));
const color = (r: number, g: number, b: number): aval<unknown> => AVal.constant(new V4f(r, g, b, 1));

type SceneLike = {
  update(t: AdaptiveToken): void;
  encodeComputePrep(e: GPUCommandEncoder, t: AdaptiveToken): void;
  encodeIntoPass(p: GPURenderPassEncoder, t: AdaptiveToken): void;
};
async function render(device: GPUDevice, scene: SceneLike): Promise<number[][]> {
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
/** Vertices the scene's megacalls emitted in the last frame (summed over buckets). */
async function emitted(device: GPUDevice, scene: unknown): Promise<number> {
  const buckets = (scene as { _debug: { bucketsForTest(): readonly { indirectBuf: GPUBuffer | undefined }[] } })._debug.bucketsForTest();
  let total = 0;
  for (const b of buckets) {
    if (b.indirectBuf === undefined) continue;
    const copy = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(b.indirectBuf, 0, copy, 0, 16);
    device.queue.submit([enc.finish()]);
    await copy.mapAsync(GPUMapMode.READ);
    total += new Uint32Array(copy.getMappedRange())[0]!;
    copy.unmap(); copy.destroy();
  }
  return total;
}

describe("heap frustum culling (real GPU)", () => {
  it("a draw whose box is outside the view emits no vertices, and comes back when it is inside", async () => {
    try { globalThis.localStorage?.clear(); } catch { /* no storage */ }
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
      const viewProj = cval(Trafo3d.identity);
      const box = AVal.constant(quadBox);
      const middleBox = cval(quadBox);
      const draw = (x: number, c: aval<unknown>, bounds?: aval<BoundsLike>): HeapDrawSpec => ({
        effect,
        inputs: { Positions: pos, ModelTrafo: shift(x), ViewProjTrafo: viewProj, Color: c, LightLocation: AVal.constant(new V3d(0, 0, 1)) },
        indices: new Uint32Array([0, 1, 2, 3, 4, 5]),
        ...(bounds !== undefined ? { bounds } : {}),
      } as HeapDrawSpec);
      const scene = buildHeapScene(device, sig, [
        draw(0, color(1, 0, 0), box),              // left third, red
        draw(third, color(0, 1, 0), middleBox),    // middle third, green
        draw(10, color(0, 0, 1), box),             // far off to the right, blue
      ], { fragmentOutputLayout: { locations: new Map([["outColor", 0]]) } });

      const R = [255, 0, 0], G = [0, 255, 0], B = [0, 0, 255], K = [0, 0, 0];
      expect(await render(device, scene)).toEqual([R, G, K]);
      expect(await emitted(device, scene), "the far draw is culled").toBe(12);

      // move the view so that only the far draw is inside
      transact(() => { viewProj.value = Trafo3d.translation(new V3d(-(10 - 2 * third), 0, 0)); });
      expect(await render(device, scene)).toEqual([K, K, B]);
      expect(await emitted(device, scene), "now the other two are culled").toBe(6);

      // back; then the middle draw's BOX moves away (its geometry does not):
      // the box is what is tested
      transact(() => { viewProj.value = Trafo3d.identity; });
      expect(await emitted(device, scene)).toBe(6);  // (the previous frame's count, before rendering)
      expect(await render(device, scene)).toEqual([R, G, K]);
      expect(await emitted(device, scene)).toBe(12);
      transact(() => { middleBox.value = { min: { x: 50, y: 50, z: 0 }, max: { x: 51, y: 51, z: 0 } }; });
      expect(await render(device, scene)).toEqual([R, K, K]);
      expect(await emitted(device, scene), "a changed box takes effect").toBe(6);
      expect(errors.map(e => (e as { message?: string }).message)).toEqual([]);
    } finally { device.destroy(); }
  });

  it("a draw without a box is always drawn; a box that straddles the edge is kept", async () => {
    try { globalThis.localStorage?.clear(); } catch { /* no storage */ }
    const device = await requestRealDevice();
    try {
      const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
      const identity = AVal.constant(Trafo3d.identity);
      const draw = (x: number, bounds?: aval<BoundsLike>): HeapDrawSpec => ({
        effect,
        inputs: { Positions: pos, ModelTrafo: shift(x), ViewProjTrafo: identity, Color: color(1, 1, 1), LightLocation: AVal.constant(new V3d(0, 0, 1)) },
        indices: new Uint32Array([0, 1, 2, 3, 4, 5]),
        ...(bounds !== undefined ? { bounds } : {}),
      } as HeapDrawSpec);
      const scene = buildHeapScene(device, sig, [
        draw(10),                                  // off screen, but no box: emitted
        draw(10, AVal.constant(quadBox)),          // off screen with a box: culled
        draw(2 * third + 0.3, AVal.constant(quadBox)),   // half over the right edge: kept
      ], { fragmentOutputLayout: { locations: new Map([["outColor", 0]]) } });
      await render(device, scene);
      expect(await emitted(device, scene)).toBe(12);
    } finally { device.destroy(); }
  });

  // Through render objects: a box that LIES (far away while the geometry is
  // in view) makes the draw disappear, which shows the box reached the heap.
  it("RenderObject.bounds reaches the heap through the scene", async () => {
    try { globalThis.localStorage?.clear(); } catch { /* no storage */ }
    const device = await requestRealDevice();
    try {
      const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
      const view: BufferView = { buffer: AVal.constant(IBuffer.fromHost(pos.buffer)), offset: 0, stride: 12, elementType: ElementType.V3f };
      const ro = (x: number, c: aval<unknown>, bounds?: BoundsLike): RenderObject => ({
        effect,
        pipelineState: PipelineState.constant({
          rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" },
          depth: { write: true, compare: "less-equal" },
        }),
        vertexAttributes: AttributeProvider.ofObject({ Positions: view }),
        uniforms: UniformProvider.ofObject({
          ModelTrafo: shift(x), ViewProjTrafo: AVal.constant(Trafo3d.identity), Color: c,
          LightLocation: AVal.constant(new V3d(0, 0, 1)),
        }),
        textures: HashMap.empty(),
        samplers: HashMap.empty(),
        ...(bounds !== undefined ? { bounds: AVal.constant(bounds) } : {}),
        drawCall: AVal.constant<DrawCall>({ kind: "non-indexed", vertexCount: 6, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
      } as unknown as RenderObject);
      const far: BoundsLike = { min: { x: 50, y: 50, z: 0 }, max: { x: 51, y: 51, z: 0 } };
      const tree = RenderTree.ordered(
        RenderTree.leaf(ro(0, color(1, 0, 0), quadBox)),       // honest box: drawn
        RenderTree.leaf(ro(third, color(0, 1, 0), far)),       // box says "far away": not drawn
        RenderTree.leaf(ro(2 * third, color(0, 0, 1))),        // no box: drawn
      );
      const scene = compileHybridScene(device, sig, tree, { heapEnabled: AVal.constant(true) });
      expect(await render(device, scene)).toEqual([[255, 0, 0], [0, 0, 0], [0, 0, 255]]);
      expect(scene.heapTotalDraws()).toBe(3);
    } finally { device.destroy(); }
  });
});
