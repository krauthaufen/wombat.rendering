// Per-frame resource-leak regression guard (real GPU). Wombat had draw-level
// counters + validateHeap() but no always-on net-live count of GPU handles
// and no long-run golden — so a per-FRAME leak inside one long-lived scene
// (the kind that survives the per-scene lifetime tests) could slip through.
//
// We wrap the GPUDevice to count net-live GPUBuffer / GPUTexture handles
// (both have an explicit destroy()), drive many rounds of add → compact →
// remove churn against ONE long-lived scene, and assert the live counts stay
// flat across the back half of the run. A leak — a removeDraw that forgets a
// buffer, or the compaction scratch buffer not being destroyed — shows up as
// a count that climbs with the round index.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import { buildHeapScene, type HeapDrawSpec } from "@aardworx/wombat.rendering/runtime";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";
import { makeHeapTestEffect } from "../tests/_heapTestEffect.js";

interface LiveStats { liveBuffers: number; liveTextures: number }

/** Proxy a GPUDevice so createBuffer/createTexture bump a net-live counter and
 *  the returned object's destroy() decrements it (guarded against double-free).
 *  Everything else delegates to the real device. */
function instrument(device: GPUDevice): { device: GPUDevice; stats: LiveStats } {
  const stats: LiveStats = { liveBuffers: 0, liveTextures: 0 };
  const wrapDestroy = (obj: { destroy(): void }, dec: () => void): void => {
    const orig = obj.destroy.bind(obj);
    let dead = false;
    Object.defineProperty(obj, "destroy", {
      configurable: true,
      value: () => { if (!dead) { dead = true; dec(); } orig(); },
    });
  };
  const proxy = new Proxy(device, {
    get(target, prop) {
      if (prop === "createBuffer") {
        return (desc: GPUBufferDescriptor) => {
          const b = target.createBuffer(desc);
          stats.liveBuffers++;
          wrapDestroy(b, () => { stats.liveBuffers--; });
          return b;
        };
      }
      if (prop === "createTexture") {
        return (desc: GPUTextureDescriptor) => {
          const t = target.createTexture(desc);
          stats.liveTextures++;
          wrapDestroy(t, () => { stats.liveTextures--; });
          return t;
        };
      }
      // Read directly off `target` (not via Reflect with the proxy as
      // receiver) so GPUDevice accessors (queue / limits / lost / …) run with
      // their real `this`. Bind methods to the real device.
      const v = (target as unknown as Record<string | symbol, unknown>)[prop];
      return typeof v === "function" ? (v as (...a: unknown[]) => unknown).bind(target) : v;
    },
  });
  return { device: proxy, stats };
}

const effect = makeHeapTestEffect();
const trafoId = AVal.constant(Trafo3d.identity) as unknown;

/** A triangle of `verts` vertices centred at (cx,cy) — distinct vertex count
 *  drives both attribute- and index-allocation size variety. */
function spec(verts: number, cx: number, cy: number, col: V4f): HeapDrawSpec {
  const positions = new Float32Array(verts * 3);
  for (let i = 0; i < verts; i++) {
    const a = (i / verts) * Math.PI * 2;
    positions[i * 3] = cx + Math.cos(a) * 0.2;
    positions[i * 3 + 1] = cy + Math.sin(a) * 0.2;
  }
  const idx = new Uint32Array(verts); for (let i = 0; i < verts; i++) idx[i] = i;
  return {
    effect,
    inputs: {
      Positions: positions,
      Normals: new Float32Array(verts * 3).fill(0),
      ModelTrafo: trafoId,
      Color: col as unknown,
      ViewProjTrafo: trafoId,
      LightLocation: new V3d(0, 0, 1) as unknown,
    },
    indices: idx,
  };
}

describe("heap per-frame resource-leak guard (real GPU)", () => {
  it("net-live GPU handles stay flat across add → compact → remove churn", async () => {
    const device0 = await requestRealDevice();
    const { device, stats } = instrument(device0);
    const errors: GPUError[] = [];
    device0.onuncapturederror = (e) => errors.push(e.error);
    try {
      const sig = createFramebufferSignature({
        colors: { outColor: "rgba8unorm" },
        depthStencil: { format: "depth24plus" },
      });
      // Tiny compaction floor so even small fragmentation triggers it.
      const scene = buildHeapScene(device, sig, [], {
        fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
        compactionWasteFloorBytes: 0,
      });
      const dbg = (scene as unknown as { _debug: { forceCompact(): number } })._debug;

      // One permanent survivor keeps the bucket alive (Count ≥ 1) across the run.
      scene.addDraw(spec(3, 0, 0, new V4f(1, 1, 1, 1)));

      const W = 32, H = 32;
      const colorTex = device.createTexture({
        size: { width: W, height: H }, format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      const depthTex = device.createTexture({
        size: { width: W, height: H }, format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      const renderOnce = (): void => {
        scene.update(AdaptiveToken.top);
        const enc = device.createCommandEncoder();
        scene.encodeComputePrep(enc, AdaptiveToken.top);
        const pass = enc.beginRenderPass({
          colorAttachments: [{ view: colorTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
          depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" },
        });
        scene.encodeIntoPass(pass);
        pass.end();
        device.queue.submit([enc.finish()]);
      };

      const ROUNDS = 30;
      const samples: LiveStats[] = [];
      for (let r = 0; r < ROUNDS; r++) {
        // t1 then t2 (t2 lands above t1 in the bump arena).
        const t1 = scene.addDraw(spec(3, -0.4, -0.4, new V4f(1, 0, 0, 1)));
        scene.update(AdaptiveToken.top);
        const t2 = scene.addDraw(spec(6, 0.4, -0.4, new V4f(0, 1, 0, 1)));
        scene.update(AdaptiveToken.top);
        // Free t1 → a hole below t2; force a compaction that relocates t2
        // (and the survivor) downward — creates + destroys a scratch buffer.
        scene.removeDraw(t1);
        scene.update(AdaptiveToken.top);
        dbg.forceCompact();
        // Remove t2 too → back to the single survivor baseline.
        scene.removeDraw(t2);
        renderOnce();
        await device.queue.onSubmittedWorkDone();
        samples.push({ ...stats });
      }

      // Back-half flatness: once warmed up (buffers sized, freelists primed),
      // the net-live counts must not climb with the round index.
      const back = samples.slice(ROUNDS / 2);
      const baseBuffers = back[0]!.liveBuffers;
      const baseTextures = back[0]!.liveTextures;
      // Guard against a vacuous pass: the instrument must actually be tracking
      // the scene's handles (arena chunks, drawHeap/drawTable/scan buffers, the
      // framebuffer textures, …), so a no-op wrapper can't make 0 === 0 pass.
      expect(baseBuffers).toBeGreaterThan(0);
      expect(baseTextures).toBeGreaterThan(0);
      for (const s of back) {
        expect(s.liveBuffers).toBe(baseBuffers);
        expect(s.liveTextures).toBe(baseTextures);
      }
      // And the survivor still renders something.
      const px = await readTexturePixels(device, colorTex);
      let nonClear = 0;
      for (let i = 0; i < px.length; i += 4) if (px[i] || px[i + 1] || px[i + 2]) nonClear++;
      expect(nonClear).toBeGreaterThan(0);
      expect(errors).toEqual([]);

      scene.dispose();
      colorTex.destroy();
      depthTex.destroy();
    } finally {
      device0.destroy();
    }
  }, 60000);
});
