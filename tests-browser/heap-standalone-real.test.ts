// Real-GPU regression for the STANDALONE (non-atlas) texture path.
//
// A standalone-routed texture keeps its own texture_2d + sampler decls in the
// emitted WGSL; the BGL must agree with those decls' @binding numbers/types.
// A mismatch makes WebGPU reject the pipeline, and an invalid pipeline poisons
// the WHOLE shared render pass — every co-resident heap draw silently vanishes.
//
// This builds ONE heap scene (no atlas) with two draws:
//   - LEFT half:  a standalone-textured RO sampling a solid-red GPUTexture.
//   - RIGHT half: a plain untextured RO emitting solid blue from a uniform.
// It asserts left ≈ red AND right ≈ blue: the standalone draw renders itself
// AND does not corrupt the co-resident heap draw.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import { ITexture, ISampler } from "@aardworx/wombat.rendering/core";
// Workspace source (NOT the published alias) — this exercises the
// standalone-path fixes (BGL binding order + bucket.textures wiring).
import {
  buildHeapScene, type HeapDrawSpec, type HeapTextureSet,
} from "../packages/rendering/src/runtime/heapScene.js";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";
import { makeHeapTestEffect, makeHeapTestEffectTextured } from "../tests/_heapTestEffect.js";

const trafoIdentity = AVal.constant(Trafo3d.identity) as unknown;
const v3 = (x: number, y: number, z: number) => new V3d(x, y, z) as unknown;
const v4 = (x: number, y: number, z: number, w: number) => new V4f(x, y, z, w) as unknown;

describe("heap standalone real-GPU integration", () => {
  it("standalone-textured RO renders AND a co-resident plain heap RO still renders", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const sig = createFramebufferSignature({
        colors: { outColor: "rgba8unorm" },
        depthStencil: { format: "depth24plus" },
      });

      // Solid-red 64×64 GPU texture, bound STANDALONE (kind:"gpu").
      const redTex = device.createTexture({
        size: { width: 64, height: 64, depthOrArrayLayers: 1 },
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      const redData = new Uint8Array(64 * 64 * 4);
      for (let i = 0; i < 64 * 64; i++) { redData[i * 4] = 255; redData[i * 4 + 3] = 255; }
      device.queue.writeTexture(
        { texture: redTex }, redData,
        { bytesPerRow: 64 * 4, rowsPerImage: 64 }, { width: 64, height: 64 },
      );
      const sampler = device.createSampler({
        magFilter: "linear", minFilter: "linear",
        addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
      });
      const standaloneTex: HeapTextureSet & { kind: "standalone" } = {
        kind: "standalone",
        texture: ITexture.fromGPU(redTex),
        sampler: ISampler.fromGPU(sampler),
      };

      const quadPos = (xMin: number, xMax: number) => new Float32Array([
        xMin, -1, 0,  xMax, -1, 0,  xMin, 1, 0,
        xMin,  1, 0,  xMax, -1, 0,  xMax, 1, 0,
      ]);
      const UV = new Float32Array([1,1,0, 1,1,0, 1,1,0, 1,1,0, 1,1,0, 1,1,0]); // sample corner (solid)
      const idx = new Uint32Array([0, 1, 2, 3, 4, 5]);

      const draws: HeapDrawSpec[] = [
        {
          // LEFT: standalone-textured. FS = color * tex; color=(1,1,1,1),
          // tex=red → outColor = red.
          effect: makeHeapTestEffectTextured(),
          inputs: {
            Positions: quadPos(-1, 0), Normals: UV,
            ModelTrafo: trafoIdentity, Color: v4(1, 1, 1, 1),
            ViewProjTrafo: trafoIdentity, LightLocation: v3(0, 0, 1),
          },
          indices: idx,
          textures: standaloneTex,
        },
        {
          // RIGHT: plain untextured heap draw. FS = color → blue.
          effect: makeHeapTestEffect(),
          inputs: {
            Positions: quadPos(0, 1), Normals: UV,
            ModelTrafo: trafoIdentity, Color: v4(0, 0, 1, 1),
            ViewProjTrafo: trafoIdentity, LightLocation: v3(0, 0, 1),
          },
          indices: idx,
        },
      ];

      const scene = buildHeapScene(device, sig, draws, {
        fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
        // NO atlasPool → the textured RO takes the standalone path.
      });

      const W = 32, H = 16;
      const colorTex = device.createTexture({
        size: { width: W, height: H, depthOrArrayLayers: 1 },
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      const depthTex = device.createTexture({
        size: { width: W, height: H, depthOrArrayLayers: 1 },
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });

      scene.update(AdaptiveToken.top);
      const enc = device.createCommandEncoder({ label: "test/standalone-render" });
      scene.encodeComputePrep(enc, AdaptiveToken.top);
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: colorTex.createView(),
          clearValue: { r: 0, g: 1, b: 0, a: 1 }, // green clear → unrendered pixels are obvious
          loadOp: "clear", storeOp: "store",
        }],
        depthStencilAttachment: {
          view: depthTex.createView(),
          depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
        },
      });
      scene.encodeIntoPass(pass);
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();

      const pixels = await readTexturePixels(device, colorTex);
      if (errors.length > 0) {
        const msgs = errors.map(e => (e as { message?: string }).message ?? String(e));
        throw new Error("GPU errors:\n  " + msgs.join("\n  "));
      }

      const at = (x: number, y: number) => {
        const o = (y * W + x) * 4;
        return [pixels[o]!, pixels[o + 1]!, pixels[o + 2]!];
      };
      // LEFT quarter ≈ red (standalone texture rendered).
      const [lr, lg, lb] = at(8, 8);
      expect(lr, "left red").toBeGreaterThan(200);
      expect(lg, "left green").toBeLessThan(60);
      expect(lb, "left blue").toBeLessThan(60);
      // RIGHT quarter ≈ blue (co-resident plain draw rendered → not corrupted).
      const [rr, rg, rb] = at(24, 8);
      expect(rb, "right blue").toBeGreaterThan(200);
      expect(rr, "right red").toBeLessThan(60);
      expect(rg, "right green").toBeLessThan(60);
    } finally {
      device.destroy();
    }
  });
});
