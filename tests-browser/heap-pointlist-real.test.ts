// Real-GPU regression for NON-INDEXED and POINT-LIST heap draws.
//
// Found via pointshare-wombat: a point-list heap RO (non-indexed,
// `vertexCount` given) uploads fine, the megacall submits its indirect
// draw, no validation errors fire — and zero fragments come out. This
// test bisects the two axes involved:
//
//   1. indexed triangle-list      — control, the exercised path
//   2. NON-indexed triangle-list  — the `vertexCount` path, no indices
//   3. NON-indexed point-list     — the point-cloud case
//
// Each case renders into its own small target and asserts colored
// pixels actually landed.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import {
  buildHeapScene, type HeapDrawSpec,
} from "../packages/rendering/src/runtime/heapScene.js";
import { PipelineState } from "../packages/rendering/src/core/pipelineState.js";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";
import { makeHeapTestEffect } from "../tests/_heapTestEffect.js";

const trafoIdentity = AVal.constant(Trafo3d.identity) as unknown;
const v3 = (x: number, y: number, z: number) => new V3d(x, y, z) as unknown;
const v4 = (x: number, y: number, z: number, w: number) => new V4f(x, y, z, w) as unknown;

// fullscreen-ish quad, 6 plain vertices (usable indexed via 0..5 too)
const quadPos = new Float32Array([
  -1, -1, 0,   1, -1, 0,  -1, 1, 0,
  -1,  1, 0,   1, -1, 0,   1, 1, 0,
]);
const quadNrm = new Float32Array(18).fill(0).map((_, i) => (i % 3 === 2 ? 1 : 0));

function pointGrid(n: number): Float32Array {
  // n×n points spread across clip space, offset to pixel centers-ish
  const out = new Float32Array(n * n * 3);
  let o = 0;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      out[o++] = ((x + 0.5) / n) * 2 - 1;
      out[o++] = ((y + 0.5) / n) * 2 - 1;
      out[o++] = 0;
    }
  }
  return out;
}

async function renderOne(spec: Omit<HeapDrawSpec, "effect">): Promise<{
  coloredPixels: number; total: number; errors: string[];
}> {
  const device = await requestRealDevice();
  const errors: string[] = [];
  device.onuncapturederror = (e) =>
    errors.push((e.error as { message?: string }).message ?? String(e.error));
  try {
    const sig = createFramebufferSignature({
      colors: { outColor: "rgba8unorm" },
      depthStencil: { format: "depth24plus" },
    });
    const draws: HeapDrawSpec[] = [{
      effect: makeHeapTestEffect(),
      ...spec,
    } as HeapDrawSpec];
    const scene = buildHeapScene(device, sig, draws, {
      fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
    });

    const W = 16, H = 16;
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
    const enc = device.createCommandEncoder();
    scene.encodeComputePrep(enc, AdaptiveToken.top);
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: colorTex.createView(),
        clearValue: { r: 0, g: 1, b: 0, a: 1 }, // green clear
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
    // count pixels that are ≈ the draw color (blue) instead of the clear
    let colored = 0;
    for (let i = 0; i < W * H; i++) {
      const r = pixels[i * 4]!, g = pixels[i * 4 + 1]!, b = pixels[i * 4 + 2]!;
      if (b > 200 && g < 60 && r < 60) colored++;
    }
    scene.dispose();
    return { coloredPixels: colored, total: W * H, errors };
  } finally {
    device.destroy();
  }
}

const blueInputs = {
  ModelTrafo: trafoIdentity, Color: v4(0, 0, 1, 1),
  ViewProjTrafo: trafoIdentity, LightLocation: v3(0, 0, 1),
};

describe("heap non-indexed + point-list real-GPU", () => {
  it("indexed triangle-list renders (control)", async () => {
    const r = await renderOne({
      inputs: { Positions: quadPos, Normals: quadNrm, ...blueInputs },
      indices: new Uint32Array([0, 1, 2, 3, 4, 5]),
    });
    expect(r.errors, "gpu errors").toEqual([]);
    expect(r.coloredPixels, "colored pixels (indexed tri)").toBeGreaterThan(r.total * 0.9);
  });

  it("NON-indexed triangle-list renders (vertexCount path)", async () => {
    const r = await renderOne({
      inputs: { Positions: quadPos, Normals: quadNrm, ...blueInputs },
      vertexCount: 6,
    });
    expect(r.errors, "gpu errors").toEqual([]);
    expect(r.coloredPixels, "colored pixels (non-indexed tri)").toBeGreaterThan(r.total * 0.9);
  });

  it("NON-indexed point-list renders", async () => {
    const n = 8; // 64 points over a 16x16 target
    const r = await renderOne({
      inputs: { Positions: pointGrid(n), Normals: new Float32Array(n * n * 3), ...blueInputs },
      vertexCount: n * n,
      pipelineState: PipelineState.constant({
        rasterizer: { topology: "point-list", cullMode: "none", frontFace: "ccw" },
        depth: { write: true, compare: "less" },
      }),
    });
    expect(r.errors, "gpu errors").toEqual([]);
    expect(r.coloredPixels, "colored pixels (points)").toBeGreaterThanOrEqual(n * n * 0.8);
  });

  it("INDEXED point-list renders", async () => {
    const n = 8;
    const r = await renderOne({
      inputs: { Positions: pointGrid(n), Normals: new Float32Array(n * n * 3), ...blueInputs },
      indices: new Uint32Array(Array.from({ length: n * n }, (_, i) => i)),
      pipelineState: PipelineState.constant({
        rasterizer: { topology: "point-list", cullMode: "none", frontFace: "ccw" },
        depth: { write: true, compare: "less" },
      }),
    });
    expect(r.errors, "gpu errors").toEqual([]);
    expect(r.coloredPixels, "colored pixels (indexed points)").toBeGreaterThanOrEqual(n * n * 0.8);
  });
});
