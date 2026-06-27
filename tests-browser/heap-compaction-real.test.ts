// Real-GPU end-to-end test for waste-triggered attribute-arena
// compaction. Renders a set of "survivor" draws twice:
//   (1) a reference scene containing only the survivors, and
//   (2) a churn scene that allocates the survivors interleaved with
//       varying-size "filler" draws, removes the fillers (fragmenting
//       the arena), then FORCES a compaction that relocates the
//       survivors' live allocations to fill the holes.
// If every cached arena ref (uniform-pool entries, drawHeader cells,
// Pos/Nor caches) is re-seated correctly and the GPU byte-move is right,
// the two framebuffers are pixel-identical. A wrong remap corrupts a
// survivor's Positions / Normals / Color and the pixels diverge.

import { describe, expect, it } from "vitest";
import { AdaptiveToken } from "@aardworx/wombat.adaptive";
import { buildHeapScene, type HeapDrawSpec } from "@aardworx/wombat.rendering/runtime";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";
import { makeHeapTestEffect } from "../tests/_heapTestEffect.js";

const IDENTITY44 = (() => { const a = new Float64Array(16); a[0]=1; a[5]=1; a[10]=1; a[15]=1; return a; })();
// PACKER_MAT4 reads `.forward` then calls `m.copyTo(dst, off)` (narrows
// f64→f32 via `dst.set`). Stub a M44d-shaped value with that method.
const matIdentity = { copyTo: (dst: Float32Array, off: number) => { dst.set(IDENTITY44, off); } };
const trafoIdentity = { forward: matIdentity } as unknown;
const v3 = (x: number, y: number, z: number) => ({ x, y, z }) as unknown;
const v4 = (x: number, y: number, z: number, w: number) => ({ x, y, z, w }) as unknown;

/** A small triangle (3 verts) centred at (cx,cy) in clip space. Fresh
 *  Float32Array each call ⇒ a distinct aval ⇒ its own arena allocation. */
function tri(cx: number, cy: number, s: number): Float32Array {
  return new Float32Array([cx - s, cy - s, 0,  cx + s, cy - s, 0,  cx, cy + s, 0]);
}
/** Distinct varying-length geometry for a filler (drives size drift). */
function fillerGeom(verts: number): { positions: Float32Array; idx: Uint32Array } {
  const positions = new Float32Array(verts * 3);
  for (let i = 0; i < verts; i++) { positions[i * 3] = -2; positions[i * 3 + 1] = -2; } // off-screen
  const idx = new Uint32Array(verts); for (let i = 0; i < verts; i++) idx[i] = i;
  return { positions, idx };
}

const effect = makeHeapTestEffect();
// Derived uniforms off: this test drives the arena BYTE-MOVE + the core ref
// holders (uniform pool, drawHeader cells, Pos/Nor caches). The §7 derived
// and modes-master remaps are covered by focused unit tests
// (records.remapHostHeap / partition.remapUniformRefs) since exercising them
// here would need real Trafo3d avals, which pull wombat.base into the browser
// bundle (poly2tri vs vite dep-optimiser — see the megacall test's note).
const fbOpts = {
  fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
  enableDerivedUniforms: false,
} as const;

/** The three on-screen survivors, distinct region + colour. Built once so
 *  the reference and churn scenes use identical values. */
function survivorSpecs(): HeapDrawSpec[] {
  const regions = [
    { c: [-0.5, -0.5] as const, col: v4(1, 0, 0, 1) },
    { c: [ 0.5, -0.5] as const, col: v4(0, 1, 0, 1) },
    { c: [ 0.0,  0.5] as const, col: v4(0, 0, 1, 1) },
  ];
  return regions.map(({ c, col }) => {
    const positions = tri(c[0], c[1], 0.25);
    return {
      effect,
      inputs: {
        Positions: positions,
        Normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        ModelTrafo: trafoIdentity,
        Color: col,
        ViewProjTrafo: trafoIdentity,
        LightLocation: v3(0, 0, 1),
      },
      indices: new Uint32Array([0, 1, 2]),
    };
  });
}

async function renderScene(device: GPUDevice, scene: ReturnType<typeof buildHeapScene>): Promise<Uint8Array> {
  const W = 64, H = 64;
  const colorTex = device.createTexture({
    size: { width: W, height: H }, format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const depthTex = device.createTexture({
    size: { width: W, height: H }, format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  scene.update(AdaptiveToken.top);
  const enc = device.createCommandEncoder();
  scene.encodeComputePrep(enc, AdaptiveToken.top);
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view: colorTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 },
      loadOp: "clear", storeOp: "store",
    }],
    depthStencilAttachment: {
      view: depthTex.createView(), depthClearValue: 1.0,
      depthLoadOp: "clear", depthStoreOp: "store",
    },
  });
  scene.encodeIntoPass(pass);
  pass.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const pixels = await readTexturePixels(device, colorTex);
  colorTex.destroy();
  depthTex.destroy();
  return pixels;
}

describe("heap attribute-arena compaction (real GPU)", () => {
  it("survivors render identically after fragment + forced compaction", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const sig = createFramebufferSignature({
        colors: { outColor: "rgba8unorm" },
        depthStencil: { format: "depth24plus" },
      });

      // (1) Reference: survivors only.
      const refScene = buildHeapScene(device, sig, survivorSpecs(), fbOpts);
      const refPixels = await renderScene(device, refScene);

      // (2) Churn: fillers first (low offsets), then survivors (higher
      //     offsets), so removing the fillers leaves holes BELOW the
      //     survivors — compaction must relocate the survivors down.
      const scene = buildHeapScene(device, sig, [], fbOpts);
      const fillerIds: number[] = [];
      for (let j = 0; j < 6; j++) {
        const { positions, idx } = fillerGeom(3 + j * 2); // varying size ⇒ drift
        fillerIds.push(scene.addDraw({
          effect,
          inputs: {
            Positions: positions,
            Normals: new Float32Array((3 + j * 2) * 3).fill(0),
            ModelTrafo: trafoIdentity,
            Color: v4(0.1 * j, 0.2, 0.3, 1),
            ViewProjTrafo: trafoIdentity,
            LightLocation: v3(0, 0, 1),
          },
          indices: idx,
        }));
      }
      for (const s of survivorSpecs()) scene.addDraw(s);

      // Realize everything, then remove the fillers (fragments the arena).
      scene.update(AdaptiveToken.top);
      for (const id of fillerIds) scene.removeDraw(id);
      scene.update(AdaptiveToken.top);

      const dbg = (scene as unknown as {
        _debug: { forceCompact(): number; attrWasteBytes(): number };
      })._debug;
      // Fillers fragment the unified arena — distinct Positions/Normals/Color
      // AND distinct varying-length index arrays (indices are arena regions now).
      expect(dbg.attrWasteBytes()).toBeGreaterThan(0);
      const residual = dbg.forceCompact();                // relocate survivors (uniforms, attrs, indices)
      expect(residual).toBe(0);                           // arena fully packed
      expect(scene.stats.compactions).toBeGreaterThan(0);

      const churnPixels = await renderScene(device, scene);

      expect(errors).toEqual([]);
      expect(churnPixels.length).toBe(refPixels.length);
      // Pixel-identical ⇒ every survivor ref survived relocation.
      let diffs = 0;
      for (let i = 0; i < refPixels.length; i++) if (refPixels[i] !== churnPixels[i]) diffs++;
      expect(diffs).toBe(0);
      // Sanity: the survivors actually drew something (not an all-clear match).
      let nonClear = 0;
      for (let i = 0; i < refPixels.length; i += 4) {
        if (refPixels[i] !== 0 || refPixels[i + 1] !== 0 || refPixels[i + 2] !== 0) nonClear++;
      }
      expect(nonClear).toBeGreaterThan(0);

      refScene.dispose();
      scene.dispose();
    } finally {
      device.destroy();
    }
  }, 30000);
});
