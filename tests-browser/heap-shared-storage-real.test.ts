// Real-GPU test for cross-heap sharing (Part 2) — the shadow-mapping enabler.
// Two independent heap scenes (think: a shadow pass and a color pass, here
// two passes with the same effect but different per-draw colour) draw the SAME
// geometry through ONE shared `HeapStorage`. The shared geometry must be
// stored ONCE (deduped + refcounted across scenes), both scenes must render
// correctly, and a compaction triggered by one scene must re-seat the OTHER
// scene's refs too (the per-scene remap-callback registry).

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import { buildHeapScene, createHeapStorage, type HeapDrawSpec, type HeapStorage } from "@aardworx/wombat.rendering/runtime";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";
import { makeHeapTestEffect } from "../tests/_heapTestEffect.js";

const effect = makeHeapTestEffect();
const trafoId = AVal.constant(Trafo3d.identity) as unknown;
const sharedUniforms = { ModelTrafo: trafoId, ViewProjTrafo: trafoId, LightLocation: new V3d(0, 0, 1) as unknown };

const N = 8;
const PAD = 256; // big positions (~3 KB each) so geometry dominates the arena

/** N shared geometry avals (positions), created ONCE so both scenes key the
 *  pool on the SAME aval identity ⇒ one allocation, refcount 2. */
const sharedPositions: unknown[] = [];
for (let i = 0; i < N; i++) {
  const cx = ((i % 4) + 0.5) / 4 * 2 - 1, cy = (Math.floor(i / 4) + 0.5) / 2 * 2 - 1;
  const p = new Float32Array(PAD * 3);
  p.set([cx - 0.12, cy - 0.12, 0, cx + 0.12, cy - 0.12, 0, cx, cy + 0.12, 0], 0);
  for (let v = 3; v < PAD; v++) p[v * 3] = i + v;
  sharedPositions.push(AVal.constant(p));
}
const sharedIndices = AVal.constant(new Uint32Array([0, 1, 2])) as unknown;

function specs(color: V4f): HeapDrawSpec[] {
  const out: HeapDrawSpec[] = [];
  for (let i = 0; i < N; i++) {
    out.push({
      effect,
      inputs: { Positions: sharedPositions[i], Color: color as unknown, ...sharedUniforms },
      indices: sharedIndices as never,
    });
  }
  return out;
}

const sig = createFramebufferSignature({
  colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" },
});
const fbOpts = { fragmentOutputLayout: { locations: new Map([["outColor", 0]]) } };

function makeScene(device: GPUDevice, storage: HeapStorage, color: V4f): ReturnType<typeof buildHeapScene> {
  return buildHeapScene(device, sig, specs(color), { ...fbOpts, storage });
}

async function renderScene(device: GPUDevice, scene: ReturnType<typeof buildHeapScene>): Promise<Uint8Array> {
  const W = 64, H = 64;
  const colorTex = device.createTexture({ size: { width: W, height: H }, format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const depthTex = device.createTexture({ size: { width: W, height: H }, format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
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
  await device.queue.onSubmittedWorkDone();
  const px = await readTexturePixels(device, colorTex);
  colorTex.destroy(); depthTex.destroy();
  return px;
}

/** Coverage mask: which pixels are non-clear (a triangle drew there). */
function coverage(px: Uint8Array): boolean[] {
  const m: boolean[] = [];
  for (let i = 0; i < px.length; i += 4) m.push(px[i]! !== 0 || px[i + 1]! !== 0 || px[i + 2]! !== 0);
  return m;
}

describe("heap shared storage across two heaps (real GPU)", () => {
  it("dedupes geometry, renders both, and survives a cross-scene compaction", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const storage = createHeapStorage(device);

      // Scene A (red) on the shared store.
      const sceneA = makeScene(device, storage, new V4f(1, 0, 0, 1));
      sceneA.update(AdaptiveToken.top);
      const bytesA = sceneA.stats.geometryBytes;

      // Scene B (blue) on the SAME store with the SAME geometry avals.
      const sceneB = makeScene(device, storage, new V4f(0, 0, 1, 1));
      sceneB.update(AdaptiveToken.top);
      const bytesAB = sceneA.stats.geometryBytes; // shared arena total

      // Geometry is shared: adding scene B grew the arena by only its small
      // per-draw uniforms, NOT another ~24 KB copy of the N big positions.
      const geomBytes = N * PAD * 3 * 4; // ~24 KB
      expect(bytesAB - bytesA).toBeLessThan(geomBytes / 2);

      // Both render; same geometry ⇒ identical coverage, different colour.
      const aPx = await renderScene(device, sceneA);
      const bPx = await renderScene(device, sceneB);
      const aCov = coverage(aPx), bCov = coverage(bPx);
      let drawn = 0, mismatch = 0;
      for (let i = 0; i < aCov.length; i++) { if (aCov[i]) drawn++; if (aCov[i] !== bCov[i]) mismatch++; }
      expect(drawn).toBeGreaterThan(0);
      expect(mismatch).toBe(0); // same shapes
      // A is red, B is blue at a covered pixel.
      const k = aCov.findIndex(Boolean) * 4;
      expect(aPx[k]).toBeGreaterThan(0); expect(aPx[k + 2]).toBe(0); // red
      expect(bPx[k + 2]).toBeGreaterThan(0); expect(bPx[k]).toBe(0); // blue

      // Force a compaction via scene A — it relocates the SHARED geometry, and
      // the registry must re-seat scene B's refs too. Then both must still
      // render identically to before.
      (sceneA as unknown as { _debug: { forceCompact(): number } })._debug.forceCompact();
      const aPx2 = await renderScene(device, sceneA);
      const bPx2 = await renderScene(device, sceneB);
      let aDiff = 0, bDiff = 0;
      for (let i = 0; i < aPx.length; i++) { if (aPx[i] !== aPx2[i]) aDiff++; if (bPx[i] !== bPx2[i]) bDiff++; }
      expect(aDiff).toBe(0); // scene A unchanged after compaction
      expect(bDiff).toBe(0); // scene B (the OTHER heap) re-seated correctly

      expect(errors).toEqual([]);
      sceneA.dispose();
      sceneB.dispose();
      storage.dispose();
    } finally {
      device.destroy();
    }
  }, 60000);
});
