// Real-GPU regression: a HOST texture larger than the atlas cap
// (ATLAS_MAX_DIM = 1024) on a RenderObject in a hybrid scene.
//
// heapEligibility used to admit host textures up to 4096, so such an RO rode
// the heap; the atlas can't take it, heapAdapter routed it to the standalone
// (Tier-L) branch, which only takes `kind: "gpu"` textures and threw
// "standalone path requires ITexture.kind === 'gpu'" from the frame callback
// (a decoded 4096² GLB texture in a 3D Tiles leaf). It must go to the legacy
// renderer instead: a hybrid scene atlas-compiles every textured effect, so
// even an uploaded standalone texture would be sampled through the atlas.
//
// One hybrid scene, two ROs with the same textured effect:
//   - LEFT:  2048×2048 solid-red host texture  → legacy.
//   - RIGHT: 64×64 solid-blue host texture     → heap, atlas.
// Asserts both render with their own texture, and no GPU errors.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap } from "@aardworx/wombat.adaptive";
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

function solid(size: number, r: number, g: number, b: number): ITexture {
  const data = new Uint8Array(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    data[i * 4] = r; data[i * 4 + 1] = g; data[i * 4 + 2] = b; data[i * 4 + 3] = 255;
  }
  return ITexture.fromRaw({ data, width: size, height: size, format: "rgba8unorm" });
}

function quadRO(xMin: number, xMax: number, tex: ITexture): RenderObject {
  const pos = new Float32Array([
    xMin, -1, 0,  xMax, -1, 0,  xMin, 1, 0,
    xMin,  1, 0,  xMax, -1, 0,  xMax, 1, 0,
  ]);
  const nrm = new Float32Array(18);
  const view = (a: Float32Array): BufferView =>
    ({ buffer: AVal.constant(IBuffer.fromHost(a.buffer)), offset: 0, stride: 12, elementType: ElementType.V3f });
  const texAval = AVal.constant(tex);   // one aval under both names
  const sampler = ISampler.fromDescriptor({
    magFilter: "linear", minFilter: "linear",
    addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
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
      // FS = Color * texture(checker, Color.xy): white, sampled at the (solid) corner
      Color: AVal.constant(new V4f(1, 1, 1, 1)),
      LightLocation: AVal.constant(new V3d(0, 0, 1)),
    }),
    // bound under `name` and `name_view`, as wombat.dom's producers do
    textures: HashMap.empty<string, ReturnType<typeof AVal.constant<ITexture>>>()
      .add("checker", texAval).add("checker_view", texAval),
    samplers: HashMap.empty<string, ReturnType<typeof AVal.constant<ISampler>>>().add("checker", AVal.constant(sampler)),
    drawCall: AVal.constant<DrawCall>({ kind: "non-indexed", vertexCount: 6, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
  } as RenderObject;
}

describe("host texture over the atlas cap — real GPU", () => {
  it("a 2048² host texture renders through the legacy path next to an atlas draw", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const sig = createFramebufferSignature({
        colors: { outColor: "rgba8unorm" },
        depthStencil: { format: "depth24plus" },
      });
      const tree = RenderTree.unordered(
        RenderTree.leaf(quadRO(-1, 0, solid(2048, 255, 0, 0))),
        RenderTree.leaf(quadRO(0, 1, solid(64, 0, 0, 255))),
      );
      const scene = compileHybridScene(device, sig, tree, {});

      const W = 32, H = 16;
      const colorTex = device.createTexture({
        size: { width: W, height: H }, format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      const depthTex = device.createTexture({
        size: { width: W, height: H }, format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      scene.update(AdaptiveToken.top);
      const enc = device.createCommandEncoder({ label: "test/standalone-host" });
      scene.encodeComputePrep(enc, AdaptiveToken.top);
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: colorTex.createView(), clearValue: { r: 0, g: 1, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
        depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
      });
      scene.encodeIntoPass(pass, AdaptiveToken.top);
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();

      if (errors.length > 0) {
        throw new Error("GPU errors:\n  " + errors.map(e => (e as { message?: string }).message ?? String(e)).join("\n  "));
      }
      expect(scene.heapTotalDraws(), "the 64² RO rides the heap").toBe(1);
      expect(scene.__legacyCount(), "the 2048² RO goes legacy").toBe(1);

      const pixels = await readTexturePixels(device, colorTex);
      const at = (x: number, y: number) => { const o = (y * W + x) * 4; return [pixels[o]!, pixels[o + 1]!, pixels[o + 2]!]; };
      const [lr, lg, lb] = at(8, 8);
      expect(lr, `left red (legacy 2048² host texture), got ${[lr, lg, lb]} / right ${at(24, 8)}`).toBeGreaterThan(200);
      expect(lg).toBeLessThan(60); expect(lb).toBeLessThan(60);
      const [rr, rg, rb] = at(24, 8);
      expect(rb, "right blue (atlas)").toBeGreaterThan(200);
      expect(rr).toBeLessThan(60); expect(rg).toBeLessThan(60);
    } finally {
      device.destroy();
    }
  });
});
