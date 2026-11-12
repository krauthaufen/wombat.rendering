// Real-GPU: host textures too large for the shared atlas pages get a page of
// their own (a solo page, docs/heap-large-textures.md) and their draws stay
// heap draws: one GPU texture per source however many draws sample it, one
// bucket per solo page, mixed with atlas draws; a texture that changes (also
// across the cap) and removal (the page goes).

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap, cset, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import {
  IBuffer, ITexture, ISampler, ElementType, PipelineState, RenderTree,
  type BufferView, type DrawCall, type RenderObject,
} from "@aardworx/wombat.rendering/core";
import { AttributeProvider, UniformProvider } from "../packages/rendering/src/core/provider.js";
import { compileHybridScene } from "../packages/rendering/src/runtime/hybridScene.js";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";
import { makeHeapTestEffectTextured } from "../tests/_heapTestEffect.js";

function solid(w: number, h: number, r: number, g: number, b: number): ITexture {
  const data = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) { data[i * 4] = r; data[i * 4 + 1] = g; data[i * 4 + 2] = b; data[i * 4 + 3] = 255; }
  return ITexture.fromRaw({ data, width: w, height: h, format: "rgba8unorm" });
}

/** A quad over columns [x0, x1) of n, sampling `texAval`. */
function quadRO(i: number, n: number, texAval: aval<ITexture>): RenderObject {
  const x0 = -1 + (2 * i) / n, x1 = -1 + (2 * (i + 1)) / n;
  const pos = new Float32Array([x0, -1, 0, x1, -1, 0, x0, 1, 0, x0, 1, 0, x1, -1, 0, x1, 1, 0]);
  const nrm = new Float32Array(18);
  const view = (a: Float32Array): BufferView =>
    ({ buffer: AVal.constant(IBuffer.fromHost(a.buffer)), offset: 0, stride: 12, elementType: ElementType.V3f });
  const sampler = ISampler.fromDescriptor({ magFilter: "nearest", minFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
  return {
    effect: makeHeapTestEffectTextured(),
    pipelineState: PipelineState.constant({
      rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" },
      depth: { write: true, compare: "less-equal" },
    }),
    vertexAttributes: AttributeProvider.ofObject({ Positions: view(pos), Normals: view(nrm) }),
    uniforms: UniformProvider.ofObject({
      ModelTrafo: AVal.constant(Trafo3d.identity),
      ViewProjTrafo: AVal.constant(Trafo3d.identity),
      Color: AVal.constant(new V4f(1, 1, 1, 1)),
      LightLocation: AVal.constant(new V3d(0, 0, 1)),
    }),
    textures: HashMap.empty<string, aval<ITexture>>().add("checker", texAval).add("checker_view", texAval),
    samplers: HashMap.empty<string, aval<ISampler>>().add("checker", AVal.constant(sampler)),
    drawCall: AVal.constant<DrawCall>({ kind: "non-indexed", vertexCount: 6, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
  } as RenderObject;
}

const W = 80, H = 8;

/** A device whose texture creations and destructions of a given width are
 *  counted, a scene over `ros`, and a frame function returning pixel colors. */
async function setup(widths: number[]) {
  const device = await requestRealDevice();
  const errors: GPUError[] = [];
  device.onuncapturederror = (e) => errors.push(e.error);
  const created = new Map<number, number>(), destroyed = new Map<number, number>();
  const createTexture = device.createTexture.bind(device);
  device.createTexture = (d: GPUTextureDescriptor) => {
    const t = createTexture(d);
    const w = (d.size as GPUExtent3DDict).width;
    // a solo page is the texture's width plus its 4 px of gutters
    const k = widths.find((x) => x + 4 === w);
    if (k !== undefined) {
      created.set(k, (created.get(k) ?? 0) + 1);
      const destroy = t.destroy.bind(t);
      t.destroy = () => { destroyed.set(k, (destroyed.get(k) ?? 0) + 1); destroy(); };
    }
    return t;
  };
  const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
  const colorTex = device.createTexture({ size: { width: W, height: H }, format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const depthTex = device.createTexture({ size: { width: W, height: H }, format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const frame = async (scene: ReturnType<typeof compileHybridScene>) => {
    scene.update(AdaptiveToken.top);
    const enc = device.createCommandEncoder();
    scene.encodeComputePrep(enc, AdaptiveToken.top);
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: colorTex.createView(), clearValue: { r: 0, g: 1, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
    });
    scene.encodeIntoPass(pass, AdaptiveToken.top);
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (errors.length > 0) throw new Error("GPU errors: " + errors.map(e => (e as { message?: string }).message).join("; "));
    const px = await readTexturePixels(device, colorTex);
    return (x: number) => { const o = (4 * W + x) * 4; return [px[o]!, px[o + 1]!, px[o + 2]!] as const; };
  };
  return { device, sig, created, destroyed, frame };
}

const isRed = ([r, g, b]: readonly number[]) => r! > 200 && g! < 60 && b! < 60;
const isBlue = ([r, g, b]: readonly number[]) => b! > 200 && r! < 60 && g! < 60;

describe("solo pages: large host textures on the heap — real GPU", () => {
  it("one large texture under 40 draws: one page, heap draws, the right pixels", async () => {
    const { device, sig, created, frame } = await setup([1100]);
    try {
      const tex = AVal.constant(solid(1100, 4, 255, 0, 0));
      const N = 40;
      const scene = compileHybridScene(device, sig, RenderTree.unordered(...Array.from({ length: N }, (_, i) => RenderTree.leaf(quadRO(i, N, tex)))), {});
      const at = await frame(scene);
      await frame(scene);
      expect(scene.__legacyCount(), "no legacy draw").toBe(0);
      expect(scene.heapTotalDraws(), "heap draws").toBe(N);
      expect(created.get(1100), "one solo page").toBe(1);
      for (const x of [1, 40, 78]) expect(isRed(at(x)), `red at ${x}: ${at(x)}`).toBe(true);
    } finally { device.destroy(); }
  });

  it("two large textures and an atlas texture in one scene", async () => {
    const { device, sig, created, destroyed, frame } = await setup([1100, 1200]);
    try {
      const red = AVal.constant(solid(1100, 4, 255, 0, 0));
      const blue = AVal.constant(solid(1200, 4, 0, 0, 255));
      const small = AVal.constant(solid(16, 16, 255, 0, 0));
      const scene = compileHybridScene(device, sig, RenderTree.unordered(
        RenderTree.leaf(quadRO(0, 3, red)), RenderTree.leaf(quadRO(1, 3, blue)), RenderTree.leaf(quadRO(2, 3, small))), {});
      const at = await frame(scene);
      expect(scene.__legacyCount()).toBe(0);
      expect(created.get(1100), `created ${[...created]} destroyed ${[...destroyed]}`).toBe(1);
      expect(created.get(1200)).toBe(1);
      expect(isRed(at(10)), `left ${at(10)}`).toBe(true);
      expect(isBlue(at(40)), `middle ${at(40)}`).toBe(true);
      expect(isRed(at(70)), `right (atlas) ${at(70)}`).toBe(true);
    } finally { device.destroy(); }
  });

  it("a texture that changes: large to large, across the cap both ways", async () => {
    const { device, sig, created, destroyed, frame } = await setup([1100, 1200]);
    try {
      const tex = cval<ITexture>(solid(1100, 4, 255, 0, 0));
      const scene = compileHybridScene(device, sig, RenderTree.unordered(RenderTree.leaf(quadRO(0, 1, tex))), {});
      expect(isRed((await frame(scene))(40))).toBe(true);
      transact(() => { tex.value = solid(1200, 4, 0, 0, 255); });
      let at = await frame(scene);
      await frame(scene);
      expect(isBlue(at(40)), `large → large: ${at(40)}`).toBe(true);
      expect(destroyed.get(1100), "the old page goes").toBe(1);
      transact(() => { tex.value = solid(16, 16, 255, 0, 0); });
      at = await frame(scene);
      expect(isRed(at(40)), `large → atlas: ${at(40)}`).toBe(true);
      transact(() => { tex.value = solid(1100, 4, 0, 0, 255); });
      at = await frame(scene);
      expect(isBlue(at(40)), `atlas → large: ${at(40)}`).toBe(true);
      expect(scene.__legacyCount()).toBe(0);
      expect(created.get(1100)).toBe(2);
    } finally { device.destroy(); }
  });

  it("removal: the page goes with its last draw", async () => {
    const { device, sig, created, destroyed, frame } = await setup([1100]);
    try {
      const tex = AVal.constant(solid(1100, 4, 255, 0, 0));
      const a = RenderTree.leaf(quadRO(0, 2, tex)), b = RenderTree.leaf(quadRO(1, 2, tex));
      const set = cset([a, b]);
      const scene = compileHybridScene(device, sig, RenderTree.unorderedFromSet(set), {});
      await frame(scene);
      expect(created.get(1100)).toBe(1);
      transact(() => { set.remove(a); });
      await frame(scene);
      expect(destroyed.get(1100) ?? 0, "one draw left: kept").toBe(0);
      transact(() => { set.remove(b); });
      await frame(scene);
      await frame(scene);
      expect(destroyed.get(1100), "no draw left: gone (a frame later)").toBe(1);
    } finally { device.destroy(); }
  });
});
