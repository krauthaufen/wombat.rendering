// End-to-end real-GPU test for atlas-tier rendering. Builds a HeapScene
// with megacall + two atlas-routed ROs (each with a distinct solid-
// color 64×64 srgb texture) covering separate halves of a 32×16
// framebuffer; reads back pixels and asserts left ≈ red, right ≈ blue.
//
// Validates the full atlas pipeline:
//   - bucket key folds atlas page identity (both ROs share one bucket).
//   - BGL exposes the atlas binding_arrays (linear/srgb) + sampler.
//   - drawHeader carries pageRef / formatBits / origin / size per RO.
//   - shader rewrite calls atlasSample(...) which walks the embedded
//     atlas region with hardware bilinear and software wrap/format.
//   - atlasPool upload places pixels at the right sub-rect.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken } from "@aardworx/wombat.adaptive";
import { ITexture } from "@aardworx/wombat.rendering/core";
import { ISampler } from "@aardworx/wombat.rendering/core";
import {
  buildHeapScene, type HeapDrawSpec, type HeapTextureSet,
} from "@aardworx/wombat.rendering/runtime";
import { AtlasPool } from "../packages/rendering/src/runtime/textureAtlas/atlasPool.js";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";

import { makeHeapTestEffectTextured } from "../tests/_heapTestEffect.js";

// Stub trafo / vec helpers (avoid wombat.base — see heap-scene-megacall.test.ts).
const IDENTITY44 = (() => { const a = new Float64Array(16); a[0]=1; a[5]=1; a[10]=1; a[15]=1; return a; })();
const trafoIdentity = { forward: { toArray: () => IDENTITY44 } } as unknown;
const v3 = (x: number, y: number, z: number) => ({ x, y, z }) as unknown;
const v4 = (x: number, y: number, z: number, w: number) => ({ x, y, z, w }) as unknown;

describe("heap-atlas real-GPU integration", () => {
  // N independent `texture_2d<f32>` bindings per format, addressed via
  // `switch pageRef` in WGSL. Replaces the prior `binding_array` and
  // `texture_2d_array` shapes (the former needs experimental bindless,
  // the latter requires a GPU-side copy on layer-count grow).
  it("two atlas-variant ROs render solid red / solid blue on left / right halves", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const sig = createFramebufferSignature({
        colors: { outColor: "rgba8unorm" },
        depthStencil: { format: "depth24plus" },
      });

      // 64×64 solid-color rgba8unorm-srgb sources. The srgb format is
      // consumed by the shader's format-dispatch branch; `atlasSrgb`
      // hardware-converts to linear at sample time.
      function makeSolid(r: number, g: number, b: number): Uint8Array {
        const buf = new Uint8Array(64 * 64 * 4);
        for (let i = 0; i < 64 * 64; i++) {
          buf[i * 4 + 0] = r;
          buf[i * 4 + 1] = g;
          buf[i * 4 + 2] = b;
          buf[i * 4 + 3] = 255;
        }
        return buf;
      }
      const redData  = makeSolid(255, 0, 0);
      const blueData = makeSolid(0, 0, 255);
      const redTex  = ITexture.fromRaw({ data: redData,  width: 64, height: 64, format: "rgba8unorm-srgb" });
      const blueTex = ITexture.fromRaw({ data: blueData, width: 64, height: 64, format: "rgba8unorm-srgb" });

      const pool = new AtlasPool(device);
      const redAcq = pool.acquire(
        "rgba8unorm-srgb", AVal.constant(redTex), 64, 64,
        { source: { width: 64, height: 64, host: { kind: "raw", data: redData } } },
      );
      const blueAcq = pool.acquire(
        "rgba8unorm-srgb", AVal.constant(blueTex), 64, 64,
        { source: { width: 64, height: 64, host: { kind: "raw", data: blueData } } },
      );

      const sharedShader = makeHeapTestEffectTextured();
      const sampler = ISampler.fromDescriptor({
        magFilter: "linear", minFilter: "linear",
        addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
      });

      function atlasTextures(acq: typeof redAcq): HeapTextureSet & { kind: "atlas" } {
        return {
          kind: "atlas",
          format: "rgba8unorm-srgb",
          pageId: acq.pageId,
          origin: acq.origin, size: acq.size,
          numMips: acq.numMips,
          sampler, page: acq.page,
          poolRef: acq.ref,
          release: () => pool.release(acq.ref),
        };
      }

      // Two full-quad ROs, one covering left half (x ∈ [-1, 0]),
      // one covering right half (x ∈ [0, 1]). The shared test effect's
      // VS forwards `Color` to `out.color` and the FS uses `color.xy`
      // as the sample UV — so we encode a center-of-quad UV in Color
      // (constant per draw, at 0.5 to land squarely inside each
      // 64×64 sub-rect with `clamp-to-edge` filtering).
      // Six verts (two triangles) covering [xMin..xMax] × [-1..1].
      function quadPos(xMin: number, xMax: number): Float32Array {
        return new Float32Array([
          xMin, -1, 0,
          xMax, -1, 0,
          xMin,  1, 0,
          xMin,  1, 0,
          xMax, -1, 0,
          xMax,  1, 0,
        ]);
      }
      const QUAD_UV = new Float32Array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 1, 0,
        1, 0, 0,
        1, 1, 0,
      ]);
      const idx = new Uint32Array([0, 1, 2, 3, 4, 5]);

      const draws: HeapDrawSpec[] = [
        {
          effect: sharedShader,
          inputs: {
            Positions: quadPos(-1, 0),
            Normals:   QUAD_UV,
            ModelTrafo: trafoIdentity, Color: v4(0.5, 0.5, 1, 1),
            ViewProjTrafo: trafoIdentity, LightLocation: v3(0, 0, 1),
          },
          indices: idx,
          textures: atlasTextures(redAcq),
        },
        {
          effect: sharedShader,
          inputs: {
            Positions: quadPos(0, 1),
            Normals:   QUAD_UV,
            ModelTrafo: trafoIdentity, Color: v4(0.5, 0.5, 1, 1),
            ViewProjTrafo: trafoIdentity, LightLocation: v3(0, 0, 1),
          },
          indices: idx,
          textures: atlasTextures(blueAcq),
        },
      ];

      const scene = buildHeapScene(device, sig, draws, {
        fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
        atlasPool: pool,
      });

      // Verify single bucket — both ROs share the atlas page.
      expect(scene.stats.groups).toBe(1);

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
      const enc = device.createCommandEncoder({ label: "test/atlas-render" });
      scene.encodeComputePrep(enc, AdaptiveToken.top);
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: colorTex.createView(),
          clearValue: { r: 0, g: 1, b: 0, a: 1 }, // green clear so we notice unrendered pixels
          loadOp: "clear", storeOp: "store",
        }],
        depthStencilAttachment: {
          view: depthTex.createView(),
          depthClearValue: 1.0,
          depthLoadOp: "clear", depthStoreOp: "store",
        },
      });
      scene.encodeIntoPass(pass);
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();

      const pixels = await readTexturePixels(device, colorTex);
      if (errors.length > 0) {
        const msgs = errors.map(e => (e as { message?: string }).message ?? String(e));
        // eslint-disable-next-line no-console
        console.error("GPU errors:", msgs);
        throw new Error("GPU errors:\n  " + msgs.join("\n  "));
      }
      expect(errors).toEqual([]);

      // Sample 4 representative pixels per half (avoid edges).
      // The test effect's FS does `outColor = input.color * tex`. With
      // `Color: v4(0.5, 0.5, 0, 1)` the per-pixel UV is (0.5, 0.5) —
      // squarely inside each 64×64 sub-rect — and the brightness factor
      // is 0.5 in r/g, 0 in b. Sampling sRGB red(255,0,0) → linear
      // (~0.214,0,0); writing to rgba8unorm gives ~55. Multiplied by
      // 0.5 → ~27. So the relevant assertion is: red dominates on the
      // left, blue dominates on the right — but the absolute value is
      // small (the color uniform halves it). Compare channel ratios
      // instead of absolute thresholds.
      function pixelAt(x: number, y: number): [number, number, number] {
        const i = (y * W + x) * 4;
        return [pixels[i]!, pixels[i + 1]!, pixels[i + 2]!];
      }
      // Left half (x ≈ 4..12): red channel non-zero, blue channel zero.
      for (const [x, y] of [[4, 8], [8, 4], [12, 12]] as const) {
        const [r, g, b] = pixelAt(x, y);
        expect(r, `left@(${x},${y}) expected red signal (got ${r}/${g}/${b})`).toBeGreaterThan(10);
        expect(b, `left@(${x},${y}) expected no blue (got ${r}/${g}/${b})`).toBeLessThan(5);
        void g;
      }
      // Right half (x ≈ 20..28): blue channel non-zero, red channel zero.
      for (const [x, y] of [[20, 8], [24, 4], [28, 12]] as const) {
        const [r, g, b] = pixelAt(x, y);
        expect(r, `right@(${x},${y}) expected no red (got ${r}/${g}/${b})`).toBeLessThan(5);
        expect(b, `right@(${x},${y}) expected blue signal (got ${r}/${g}/${b})`).toBeGreaterThan(10);
        void g;
      }

      scene.dispose();
      colorTex.destroy();
      depthTex.destroy();
    } finally {
      device.destroy();
    }
  }, 30000);
});
