// Real-GPU: the texture atlas's array growth refused for lack of memory.
//
// A heap scene with two atlas draws: red on the left (page 0) and blue on the
// right, whose texture needs page 1 and so a growth of the array to 2 layers.
// The device wrapper makes that allocation fail the way an out-of-memory one
// does on a real GPU: the texture it returns is invalid and the out-of-memory
// error scope around it reports an error, asynchronously. Asserted:
//   - once the refusal is in, the scene renders again without device errors:
//     the left half red (before the fix the invalid array stayed in the heap's
//     bind group and nothing at all rendered, the failure pro3d-web's Shaler
//     scene showed on a GPU with about 1 GB free);
//   - the blue draw, whose placement was lost and does not fit, samples
//     nothing (black, not the green clear: it still draws);
//   - once room frees in the kept page, the blue draw is placed again and the
//     right half turns blue.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import { ITexture } from "@aardworx/wombat.rendering/core";
import { ISampler } from "@aardworx/wombat.rendering/core";
import { AtlasPool, buildHeapScene, type HeapDrawSpec, type HeapTextureSet } from "@aardworx/wombat.rendering/runtime";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";
import { makeHeapTestEffectTextured } from "../tests/_heapTestEffect.js";

const trafoIdentity = AVal.constant(Trafo3d.identity) as unknown;
const v3 = (x: number, y: number, z: number) => new V3d(x, y, z) as unknown;
const v4 = (x: number, y: number, z: number, w: number) => new V4f(x, y, z, w) as unknown;

/** Refuses the next atlas array allocation of `layers` layers as out of memory. */
function refuseAtlasGrowth(device: GPUDevice, layers: number): void {
  const create = device.createTexture.bind(device);
  const push = device.pushErrorScope.bind(device);
  const pop = device.popErrorScope.bind(device);
  let refuse = false, refused = false;
  const d = device as unknown as Record<string, unknown>;
  d.createTexture = (desc: GPUTextureDescriptor) => {
    const size = desc.size as GPUExtent3DDict;
    if (desc.label?.startsWith("atlas/") && size.depthOrArrayLayers === layers && !refused) {
      refused = true;
      refuse = true;
      // an invalid texture, as a failed allocation leaves (its own
      // validation error swallowed here)
      push("validation");
      const t = create({ ...desc, size: { ...size, width: device.limits.maxTextureDimension2D * 2 } });
      void pop();
      return t;
    }
    return create(desc);
  };
  // the scope popped right after the refused allocation reports it
  d.popErrorScope = () => {
    const failed = refuse;
    refuse = false;
    return pop().then((e) => (failed && e === null
      ? { message: "simulated: vkAllocateMemory failed with VK_ERROR_OUT_OF_DEVICE_MEMORY" } as GPUError
      : e));
  };
}

function solid(r: number, g: number, b: number, w: number, h: number): Uint8Array {
  const buf = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) { buf[i * 4] = r; buf[i * 4 + 1] = g; buf[i * 4 + 2] = b; buf[i * 4 + 3] = 255; }
  return buf;
}

describe("heap atlas: growth refused for lack of memory (real GPU)", () => {
  it("the scene keeps rendering; the lost texture is placed again once there is room", async () => {
    // (the package's own settings, routed through globalThis)
    (globalThis as Record<string, unknown>).__wombatAtlasPageSize = 2048;
    (globalThis as Record<string, unknown>).__wombatAtlasGrowthLayers = 1;
    const device = await requestRealDevice();
    const errors: string[] = [];
    device.onuncapturederror = (e) => errors.push((e.error as { message?: string }).message ?? String(e.error));
    try {
      refuseAtlasGrowth(device, 2);
      const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
      const pool = new AtlasPool(device);
      const S = 1000; // four of them fill a 2048² page
      const place = (data: Uint8Array) => {
        const tex = ITexture.fromRaw({ data, width: S, height: S, format: "rgba8unorm" });
        const av = AVal.constant(tex);
        const host = { kind: "raw" as const, data };
        const acq = pool.acquire("rgba8unorm", av, S, S, { source: { width: S, height: S, host } })!;
        return { acq, av, host };
      };
      const red = place(solid(255, 0, 0, S, S));
      const fillers = [place(solid(9, 9, 9, S, S)), place(solid(9, 9, 9, S, S)), place(solid(9, 9, 9, S, S))];
      const blue = place(solid(0, 0, 255, S, S));
      expect(red.acq.pageId).toBe(0);
      expect(blue.acq.pageId).toBe(1);

      const sampler = ISampler.fromDescriptor({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
      const textures = (p: ReturnType<typeof place>): HeapTextureSet & { kind: "atlas" } => ({
        kind: "atlas", format: "rgba8unorm", pageId: p.acq.pageId, origin: p.acq.origin, size: p.acq.size,
        numMips: p.acq.numMips, sampler, page: p.acq.page, poolRef: p.acq.ref,
        release: () => pool.release(p.acq.ref),
        sourceAval: p.av as never, host: p.host,
      } as HeapTextureSet & { kind: "atlas" });
      const quad = (x0: number, x1: number) => new Float32Array([x0, -1, 0, x1, -1, 0, x0, 1, 0, x0, 1, 0, x1, -1, 0, x1, 1, 0]);
      const uv = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]);
      const idx = new Uint32Array([0, 1, 2, 3, 4, 5]);
      const draw = (x0: number, x1: number, p: ReturnType<typeof place>): HeapDrawSpec => ({
        effect: makeHeapTestEffectTextured(),
        inputs: { Positions: quad(x0, x1), Normals: uv, ModelTrafo: trafoIdentity, Color: v4(0.5, 0.5, 1, 1), ViewProjTrafo: trafoIdentity, LightLocation: v3(0, 0, 1) },
        indices: idx,
        textures: textures(p),
      });
      const scene = buildHeapScene(device, sig, [draw(-1, 0, red), draw(0, 1, blue)], {
        fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
        atlasPool: pool,
      });

      const W = 32, H = 16;
      const color = device.createTexture({ size: { width: W, height: H }, format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      const depth = device.createTexture({ size: { width: W, height: H }, format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
      const frame = async (): Promise<[number, number, number][]> => {
        scene.update(AdaptiveToken.top);
        const enc = device.createCommandEncoder();
        scene.encodeComputePrep(enc, AdaptiveToken.top);
        const pass = enc.beginRenderPass({
          colorAttachments: [{ view: color.createView(), clearValue: { r: 0, g: 1, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
          depthStencilAttachment: { view: depth.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
        });
        scene.encodeIntoPass(pass);
        pass.end();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        const px = await readTexturePixels(device, color);
        const at = (x: number, y: number): [number, number, number] => { const i = (y * W + x) * 4; return [px[i]!, px[i + 1]!, px[i + 2]!]; };
        return [at(8, 8), at(24, 8)];
      };

      // the refusal arrives asynchronously; the first frame may use the invalid array
      await frame();
      for (let i = 0; i < 20 && pool.capFor("rgba8unorm") !== 1; i++) await new Promise((r) => setTimeout(r, 10));
      expect(pool.capFor("rgba8unorm")).toBe(1);
      expect(pool.isLive(blue.acq.ref)).toBe(false);
      errors.length = 0;

      const [l1, r1] = await frame();
      expect(errors, "no device errors once the growth is undone").toEqual([]);
      expect(l1[0], `left red (got ${l1})`).toBeGreaterThan(60);
      expect(l1[1]).toBeLessThan(10);
      // the blue draw draws, sampling nothing: black, not the green clear
      expect(r1, "right: drawn, untextured").toEqual([0, 0, 0]);

      // room: the fillers go; the next frame places the blue texture on page 0
      for (const f of fillers) pool.release(f.acq.ref);
      pool.evictIdle();
      const [l2, r2] = await frame();
      expect(errors).toEqual([]);
      expect(l2[0]).toBeGreaterThan(60);
      expect(r2[2], `right blue (got ${r2})`).toBeGreaterThan(60);
      expect(r2[0]).toBeLessThan(10);

      scene.dispose();
      color.destroy();
      depth.destroy();
    } finally {
      device.destroy();
      delete (globalThis as Record<string, unknown>).__wombatAtlasPageSize;
      delete (globalThis as Record<string, unknown>).__wombatAtlasGrowthLayers;
    }
  }, 30000);
});
