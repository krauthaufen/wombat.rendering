// Real-GPU test for atlas texture-aval reactivity. A heap scene with
// two atlas-routed ROs each driven by a `cval<ITexture>` (left starts
// red, right starts blue) renders red/blue halves; after swapping the
// left RO's cval to point at the blue texture, both halves render blue.
//
// Validates the full reactive path:
//   - heapScene subscribes sceneObj to each atlas RO's `sourceAval`.
//   - sourceAval marks route through `inputChanged` → `atlasAvalDirty`.
//   - `update(token)` drains the dirty set, calls `pool.repack`,
//     rewrites the drawHeader fields (pageRef/origin/size).
//   - `release` closures keep working after the swap (no leaks).
//
// Mirrors `heap-atlas-real.test.ts`'s scaffolding; only the swap +
// re-render portion is new.

import { describe, expect, it } from "vitest";
import { AdaptiveToken, cval, transact } from "@aardworx/wombat.adaptive";
import { ITexture } from "@aardworx/wombat.rendering.experimental/core";
import { ISampler } from "@aardworx/wombat.rendering.experimental/core";
import {
  buildHeapScene, type HeapDrawSpec, type HeapTextureSet,
} from "@aardworx/wombat.rendering.experimental/runtime";
import { AtlasPool } from "../packages/rendering/src/runtime/textureAtlas/atlasPool.js";
import { createFramebufferSignature } from "@aardworx/wombat.rendering.experimental/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";

import { makeHeapTestEffectTextured } from "../tests/_heapTestEffect.js";

const IDENTITY44 = (() => { const a = new Float64Array(16); a[0]=1; a[5]=1; a[10]=1; a[15]=1; return a; })();
const trafoIdentity = { forward: { toArray: () => IDENTITY44 } } as unknown;
const v3 = (x: number, y: number, z: number) => ({ x, y, z }) as unknown;
const v4 = (x: number, y: number, z: number, w: number) => ({ x, y, z, w }) as unknown;

describe("heap-atlas texture-aval reactivity (real GPU)", () => {
  it("swapping a cval<ITexture> rewrites the drawHeader and re-renders", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const sig = createFramebufferSignature({
        colors: { outColor: "rgba8unorm" },
        depthStencil: { format: "depth24plus" },
      });

      function makeSolid(r: number, g: number, b: number): Uint8Array {
        const buf = new Uint8Array(64 * 64 * 4);
        for (let i = 0; i < 64 * 64; i++) {
          buf[i * 4 + 0] = r; buf[i * 4 + 1] = g;
          buf[i * 4 + 2] = b; buf[i * 4 + 3] = 255;
        }
        return buf;
      }
      const redData  = makeSolid(255, 0, 0);
      const blueData = makeSolid(0, 0, 255);
      const redTex  = ITexture.fromRaw({ data: redData,  width: 64, height: 64, format: "rgba8unorm-srgb" });
      const blueTex = ITexture.fromRaw({ data: blueData, width: 64, height: 64, format: "rgba8unorm-srgb" });

      // cval-driven sources. Both ROs route through their own cval —
      // when we transact-set leftCval = blueTex, only the left RO's
      // drawHeader should be rewritten (and the right RO is untouched).
      const leftCval  = cval<ITexture>(redTex);
      const rightCval = cval<ITexture>(blueTex);

      const pool = new AtlasPool(device);

      // Per-aval current-ref cell (mirrors heapAdapter's cell): so the
      // `release` closure always frees the latest sub-rect even after
      // a repack. For this hand-built spec we wire it inline.
      function buildAtlas(
        cv: typeof leftCval,
        initial: ITexture,
        initialData: Uint8Array,
      ): HeapTextureSet & { kind: "atlas" } {
        const acq = pool.acquire(
          "rgba8unorm-srgb", cv, 64, 64,
          { source: { width: 64, height: 64, host: { kind: "raw", data: initialData } } },
        );
        const cell = { ref: acq.ref };
        void initial;
        const sampler = ISampler.fromDescriptor({
          magFilter: "linear", minFilter: "linear",
          addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
        });
        return {
          kind: "atlas",
          format: "rgba8unorm-srgb",
          pageId: acq.pageId,
          origin: acq.origin, size: acq.size,
          numMips: acq.numMips,
          sampler, page: acq.page,
          poolRef: acq.ref,
          release: () => pool.release(cell.ref),
          sourceAval: cv,
          repack: (newTex: ITexture) => {
            const next = pool.repack(cv, newTex);
            cell.ref = next.ref;
            return next;
          },
        };
      }

      const sharedShader = makeHeapTestEffectTextured();

      function quadPos(xMin: number, xMax: number): Float32Array {
        return new Float32Array([
          xMin, -1, 0, xMax, -1, 0, xMin,  1, 0,
          xMin,  1, 0, xMax, -1, 0, xMax,  1, 0,
        ]);
      }
      const QUAD_UV = new Float32Array([
        0,0,0, 1,0,0, 0,1,0, 0,1,0, 1,0,0, 1,1,0,
      ]);
      const idx = new Uint32Array([0, 1, 2, 3, 4, 5]);

      const draws: HeapDrawSpec[] = [
        {
          effect: sharedShader,
          inputs: {
            Positions: quadPos(-1, 0), Normals: QUAD_UV,
            ModelTrafo: trafoIdentity, Color: v4(0.5, 0.5, 1, 1),
            ViewProjTrafo: trafoIdentity, LightLocation: v3(0, 0, 1),
          },
          indices: idx,
          textures: buildAtlas(leftCval, redTex, redData),
        },
        {
          effect: sharedShader,
          inputs: {
            Positions: quadPos(0, 1), Normals: QUAD_UV,
            ModelTrafo: trafoIdentity, Color: v4(0.5, 0.5, 1, 1),
            ViewProjTrafo: trafoIdentity, LightLocation: v3(0, 0, 1),
          },
          indices: idx,
          textures: buildAtlas(rightCval, blueTex, blueData),
        },
      ];

      const scene = buildHeapScene(device, sig, draws, {
        fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
        megacall: false,
        atlasPool: pool,
      });

      const W = 32, H = 16;
      function makeFb(): { color: GPUTexture; depth: GPUTexture } {
        return {
          color: device.createTexture({
            size: { width: W, height: H, depthOrArrayLayers: 1 },
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
          }),
          depth: device.createTexture({
            size: { width: W, height: H, depthOrArrayLayers: 1 },
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
          }),
        };
      }
      async function renderAndRead(): Promise<Uint8Array> {
        const { color, depth } = makeFb();
        scene.update(AdaptiveToken.top);
        const enc = device.createCommandEncoder({ label: "test/atlas-reactivity" });
        scene.encodeComputePrep(enc, AdaptiveToken.top);
        const pass = enc.beginRenderPass({
          colorAttachments: [{
            view: color.createView(),
            clearValue: { r: 0, g: 1, b: 0, a: 1 },
            loadOp: "clear", storeOp: "store",
          }],
          depthStencilAttachment: {
            view: depth.createView(),
            depthClearValue: 1.0,
            depthLoadOp: "clear", depthStoreOp: "store",
          },
        });
        scene.encodeIntoPass(pass);
        pass.end();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        const px = await readTexturePixels(device, color);
        color.destroy(); depth.destroy();
        return px;
      }

      function pixelAt(px: Uint8Array, x: number, y: number): [number, number, number] {
        const i = (y * W + x) * 4;
        return [px[i]!, px[i + 1]!, px[i + 2]!];
      }

      // Initial state: left red, right blue.
      const before = await renderAndRead();
      for (const [x, y] of [[4, 8], [12, 12]] as const) {
        const [r, , b] = pixelAt(before, x, y);
        expect(r, `pre-swap left@(${x},${y}) red`).toBeGreaterThan(10);
        expect(b, `pre-swap left@(${x},${y}) no blue`).toBeLessThan(5);
      }
      for (const [x, y] of [[20, 8], [28, 12]] as const) {
        const [r, , b] = pixelAt(before, x, y);
        expect(r, `pre-swap right@(${x},${y}) no red`).toBeLessThan(5);
        expect(b, `pre-swap right@(${x},${y}) blue`).toBeGreaterThan(10);
      }

      // Swap: route the left RO's atlas to the blue texture.
      transact(() => { leftCval.value = blueTex; });

      // Re-render — both halves should now be blue.
      const after = await renderAndRead();
      for (const [x, y] of [[4, 8], [12, 12], [20, 8], [28, 12]] as const) {
        const [r, , b] = pixelAt(after, x, y);
        expect(r, `post-swap@(${x},${y}) no red`).toBeLessThan(5);
        expect(b, `post-swap@(${x},${y}) blue`).toBeGreaterThan(10);
      }

      expect(errors).toEqual([]);
      scene.dispose();
    } finally {
      device.destroy();
    }
  }, 30000);
});
