// Real-GPU regression: many legacy-path RenderObjects sampling one host
// texture upload it once.
//
// A texture over the atlas cap (ATLAS_MAX_DIM) sends its draws to the legacy
// renderer, and prepareRenderObject gave every draw its own GPUTexture: a
// 2432² sprite sheet under a few thousand quads (PoEtry's passive tree) ran
// the GPU out of memory. The texture is now one GPU texture per host source
// (per device), whether the draws share the aval or only its value.
//
// The repro is small on purpose: a 1100×4 texture (just over the cap) and
// 40 draws, counting texture creations and uploads.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap, type aval } from "@aardworx/wombat.adaptive";
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

const TW = 1100, TH = 4;

function solidRed(): ITexture {
  const data = new Uint8Array(TW * TH * 4);
  for (let i = 0; i < TW * TH; i++) { data[i * 4] = 255; data[i * 4 + 3] = 255; }
  return ITexture.fromRaw({ data, width: TW, height: TH, format: "rgba8unorm" });
}

function quadRO(i: number, n: number, texAval: aval<ITexture>): RenderObject {
  const x0 = -1 + (2 * i) / n, x1 = -1 + (2 * (i + 1)) / n;
  const pos = new Float32Array([x0, -1, 0, x1, -1, 0, x0, 1, 0, x0, 1, 0, x1, -1, 0, x1, 1, 0]);
  const nrm = new Float32Array(18);
  const view = (a: Float32Array): BufferView =>
    ({ buffer: AVal.constant(IBuffer.fromHost(a.buffer)), offset: 0, stride: 12, elementType: ElementType.V3f });
  const sampler = ISampler.fromDescriptor({
    magFilter: "nearest", minFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
  });
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

async function run(makeAval: (tex: ITexture, i: number) => aval<ITexture>) {
  const device = await requestRealDevice();
  const errors: GPUError[] = [];
  device.onuncapturederror = (e) => errors.push(e.error);
  // count what is created and uploaded at the texture's size
  let created = 0, uploads = 0;
  const createTexture = device.createTexture.bind(device);
  device.createTexture = (d: GPUTextureDescriptor) => {
    const s = d.size as GPUExtent3DDict;
    if (s.width === TW && s.height === TH) created++;
    return createTexture(d);
  };
  const writeTexture = device.queue.writeTexture.bind(device.queue);
  device.queue.writeTexture = ((dst: GPUTexelCopyTextureInfo, data: GPUAllowSharedBufferSource, layout: GPUTexelCopyBufferLayout, size: GPUExtent3DStrict) => {
    if ((size as GPUExtent3DDict).width === TW) uploads++;
    return writeTexture(dst, data, layout, size);
  }) as typeof device.queue.writeTexture;
  try {
    const N = 40;
    const tex = solidRed();
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
    const tree = RenderTree.unordered(...Array.from({ length: N }, (_, i) => RenderTree.leaf(quadRO(i, N, makeAval(tex, i)))));
    const scene = compileHybridScene(device, sig, tree, {});
    const W = 80, H = 8;
    const colorTex = device.createTexture({ size: { width: W, height: H }, format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const depthTex = device.createTexture({ size: { width: W, height: H }, format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    for (let frame = 0; frame < 2; frame++) {
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
    }
    await device.queue.onSubmittedWorkDone();
    if (errors.length > 0) throw new Error("GPU errors: " + errors.map(e => (e as { message?: string }).message).join("; "));
    expect(scene.__legacyCount(), "over the atlas cap: legacy").toBe(N);
    const pixels = await readTexturePixels(device, colorTex);
    for (const x of [1, 40, 78]) expect(pixels[(4 * W + x) * 4], `red at x=${x}`).toBeGreaterThan(200);
    return { created, uploads };
  } finally {
    device.destroy();
  }
}

describe("legacy draws sharing a host texture — real GPU", () => {
  it("one aval under 40 draws: one texture, one upload", async () => {
    let shared: aval<ITexture> | undefined;
    const { created, uploads } = await run((tex) => (shared ??= AVal.constant(tex)));
    expect(created, "GPU textures").toBe(1);
    expect(uploads, "uploads").toBe(1);
  });
  it("40 avals of the same texture: one texture, one upload", async () => {
    const { created, uploads } = await run((tex) => AVal.constant(tex).map((t: ITexture) => t));
    expect(created, "GPU textures").toBe(1);
    expect(uploads, "uploads").toBe(1);
  });
});
