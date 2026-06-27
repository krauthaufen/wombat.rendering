// Real-GPU test for multi-page (multi-chunk) group placement. A heap scene
// whose data exceeds one storage buffer's cap must spread across several
// arena pages — each draw's whole group (uniforms + attributes + index data)
// placed wholly on ONE page, rendered as one sub-draw per page, with the
// plain single-buffer gather (no shader-side chunk switch).
//
// We render the SAME scene twice: once with a large page cap (1 page) and
// once with a tiny 64 KiB cap that forces several pages, and assert the two
// framebuffers are pixel-identical. A wrong page binding, a mis-placed group,
// or a wrong `_indexStart` would diverge.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import { buildHeapScene, type HeapDrawSpec } from "@aardworx/wombat.rendering/runtime";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";
import { makeHeapTestEffect } from "../tests/_heapTestEffect.js";

const effect = makeHeapTestEffect();
const trafoId = AVal.constant(Trafo3d.identity) as unknown;
const shared = {
  ViewProjTrafo: trafoId,
  LightLocation: new V3d(0, 0, 1) as unknown,
};

const N = 64;
const PAD_VERTS = 256; // big distinct positions arrays (~3 KB each) overflow a 64 KiB page

/** Draw i: a small triangle at the ORIGIN placed into grid cell i and rotated
 *  by a distinct angle via a per-draw `ModelTrafo` (a §7 derived passthrough →
 *  the derive must compose it into THIS draw's page). The big distinct position
 *  array pads the arena so a few groups overflow a 64 KiB page. If the per-page
 *  derive mis-routed, a page>0 draw would get the wrong Model → wrong cell /
 *  orientation → the 1-page-vs-N-page comparison diverges. */
function specFor(i: number): HeapDrawSpec {
  const cols = 8, rows = 8;
  const cx = ((i % cols) + 0.5) / cols * 2 - 1;
  const cy = (Math.floor(i / cols) + 0.5) / rows * 2 - 1;
  const s = 0.11;
  const positions = new Float32Array(PAD_VERTS * 3);
  positions.set([-s, -s, 0,  s, -s, 0,  0, s, 0], 0); // triangle at origin; Model places it
  for (let v = 3; v < PAD_VERTS; v++) positions[v * 3] = i + v; // distinct padding
  // Distinct Model: translate into cell i, then rotate by a per-draw angle.
  const model = AVal.constant(
    Trafo3d.translation(new V3d(cx, cy, 0)).mul(Trafo3d.rotation(new V3d(0, 0, 1), i * 0.37)),
  ) as unknown;
  const col = new V4f(((i * 37) % 256) / 255, ((i * 91) % 256) / 255, ((i * 53) % 256) / 255, 1);
  return {
    effect,
    inputs: { Positions: positions, Color: col as unknown, ModelTrafo: model, ...shared },
    indices: new Uint32Array([0, 1, 2]),
  };
}

async function render(device: GPUDevice, maxChunkBytes: number): Promise<{ px: Uint8Array; pages: number }> {
  const sig = createFramebufferSignature({
    colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" },
  });
  const draws: HeapDrawSpec[] = [];
  for (let i = 0; i < N; i++) draws.push(specFor(i));
  const scene = buildHeapScene(device, sig, draws, {
    fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
    maxChunkBytes,
  });
  const W = 128, H = 128;
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
    colorAttachments: [{ view: colorTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
    depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" },
  });
  scene.encodeIntoPass(pass);
  pass.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const px = await readTexturePixels(device, colorTex);
  const pages = (scene as unknown as { _debug: { pageCount(): number } })._debug.pageCount();
  scene.dispose();
  colorTex.destroy();
  depthTex.destroy();
  return { px, pages };
}

describe("heap multi-page group placement (real GPU)", () => {
  it("renders identically across 1 page and many 64 KiB pages", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const ref = await render(device, 1 << 28);        // huge cap → 1 page
      const paged = await render(device, 64 * 1024);    // 64 KiB cap → many pages

      expect(ref.pages).toBe(1);
      expect(paged.pages).toBeGreaterThan(1);            // group placement rolled to new pages
      expect(errors).toEqual([]);
      expect(paged.px.length).toBe(ref.px.length);
      let diffs = 0;
      for (let i = 0; i < ref.px.length; i++) if (ref.px[i] !== paged.px[i]) diffs++;
      expect(diffs).toBe(0);                             // pixel-identical across page layouts
      let nonClear = 0;
      for (let i = 0; i < ref.px.length; i += 4) if (ref.px[i] || ref.px[i + 1] || ref.px[i + 2]) nonClear++;
      expect(nonClear).toBeGreaterThan(0);
    } finally {
      device.destroy();
    }
  }, 60000);
});
