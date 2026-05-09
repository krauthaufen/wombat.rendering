// End-to-end test for the megacall heap-scene integration. Builds a
// HeapScene with megacall=true via the public API, adds a few draws
// of varying indexCount, drives update + encodeComputePrep, and
// verifies the GPU prefix-sum populated drawTable + indirect args.
// Then runs encodeIntoPass and asserts the framebuffer is non-empty.

import { describe, expect, it } from "vitest";
import { AdaptiveToken } from "@aardworx/wombat.adaptive";
import {
  buildHeapScene,
  type HeapDrawSpec,
} from "@aardworx/wombat.rendering.experimental/runtime";
import { createFramebufferSignature } from "@aardworx/wombat.rendering.experimental/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";
import { makeHeapTestEffect } from "../tests/_heapTestEffect.js";

// Avoid `import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base"` because
// wombat.base pulls in poly2tri which references `global` and chokes
// vite's dep optimiser. Stub the duck-typed shapes the heap-scene
// packers consume: PACKER_MAT4 reads `forward.toArray()` (row-major
// Float64), PACKER_VEC3 reads `.x/.y/.z`, PACKER_VEC4 reads `.x/.y/.z/.w`.
const IDENTITY44 = (() => { const a = new Float64Array(16); a[0]=1; a[5]=1; a[10]=1; a[15]=1; return a; })();
const trafoIdentity = { forward: { toArray: () => IDENTITY44 } } as unknown;
const v3 = (x: number, y: number, z: number) => ({ x, y, z }) as unknown;
const v4 = (x: number, y: number, z: number, w: number) => ({ x, y, z, w }) as unknown;

async function readU32(device: GPUDevice, buf: GPUBuffer, byteCount: number): Promise<Uint32Array> {
  const staging = device.createBuffer({ size: byteCount, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, byteCount);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

// Ortho-ish single-triangle geometry for each draw, all inside
// clip-space [-1,1]. Just need each draw's indexCount to differ so we
// can verify the prefix-sum.
function triGeom(): { positions: Float32Array; normals: Float32Array } {
  return {
    positions: new Float32Array([
      -1, -1, 0,
       3, -1, 0,
      -1,  3, 0,
    ]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
  };
}

describe("heap-scene megacall integration", () => {
  it("scans drawTable and writes indirect for 3 draws; renders non-empty framebuffer", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const sig = createFramebufferSignature({
        colors: { outColor: "rgba8unorm" },
        depthStencil: { format: "depth24plus" },
      });

      // Three draws with distinct indexCounts: 3, 6, 12 → prefix [0, 3, 9].
      const indexCounts = [3, 6, 12];
      const geom = triGeom();
      // Single shader object → all draws share one bucket.
      const sharedShader = makeHeapTestEffect();
      const draws: HeapDrawSpec[] = indexCounts.map((n) => {
        // Build an index list of length `n` over the 3 verts (cycle).
        const idx = new Uint32Array(n);
        for (let i = 0; i < n; i++) idx[i] = i % 3;
        return {
          effect: sharedShader,
          inputs: {
            Positions:     geom.positions,
            Normals:       geom.normals,
            ModelTrafo:    trafoIdentity,
            Color:         v4(1, 0, 0, 1),
            ViewProjTrafo: trafoIdentity,
            LightLocation: v3(0, 0, 1),
          },
          indices: idx,
        };
      });

      const scene = buildHeapScene(device, sig, draws, {
        fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
        megacall: true,
      });

      // Drive update + encodeComputePrep + submit.
      scene.update(AdaptiveToken.top);
      const enc1 = device.createCommandEncoder({ label: "test/prep" });
      scene.encodeComputePrep(enc1, AdaptiveToken.top);
      device.queue.submit([enc1.finish()]);
      await device.queue.onSubmittedWorkDone();

      // Read indirect + drawTable via the test-only debug surface.
      const debug = (scene as unknown as { _debug: { bucketsForTest(): readonly {
        indirectBuf: GPUBuffer | undefined;
        drawTableBuf: GPUBuffer | undefined;
        firstDrawInTileBuf: GPUBuffer | undefined;
        totalEmitEstimate: number;
        recordCount: number;
      }[] } })._debug;
      const dbgBuckets = debug.bucketsForTest();
      expect(dbgBuckets.length).toBe(1);
      const b = dbgBuckets[0]!;
      expect(b.recordCount).toBe(3);
      const indirect = await readU32(device, b.indirectBuf!, 16);
      const expectedTotal = indexCounts.reduce((a, c) => a + c, 0);
      expect(indirect[0]).toBe(expectedTotal);   // total emit count
      expect(indirect[1]).toBe(1);               // instanceCount
      const drawTable = await readU32(device, b.drawTableBuf!, b.recordCount * 16);
      let acc = 0;
      for (let i = 0; i < indexCounts.length; i++) {
        const firstEmit = drawTable[i * 4 + 0]!;
        const drawIdx   = drawTable[i * 4 + 1]!;
        const indexCnt  = drawTable[i * 4 + 3]!;
        expect(firstEmit).toBe(acc);
        expect(drawIdx).toBe(i);
        expect(indexCnt).toBe(indexCounts[i]!);
        acc += indexCounts[i]!;
      }

      // Allocate a real framebuffer.
      const W = 32, H = 32;
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

      const enc2 = device.createCommandEncoder({ label: "test/render" });
      scene.encodeComputePrep(enc2, AdaptiveToken.top);
      const pass = enc2.beginRenderPass({
        colorAttachments: [{
          view: colorTex.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
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
      device.queue.submit([enc2.finish()]);
      await device.queue.onSubmittedWorkDone();

      const pixels = await readTexturePixels(device, colorTex);
      // Find any non-clear-color pixel.
      let nonClear = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i]! !== 0 || pixels[i + 1]! !== 0 || pixels[i + 2]! !== 0) nonClear++;
      }
      expect(errors).toEqual([]);
      expect(nonClear).toBeGreaterThan(0);

      scene.dispose();
      colorTex.destroy();
      depthTex.destroy();
    } finally {
      device.destroy();
    }
  }, 30000);

  it("scans correctly for many draws (100) and after dynamic addDraw", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const sig = createFramebufferSignature({
        colors: { outColor: "rgba8unorm" },
        depthStencil: { format: "depth24plus" },
      });
      const sharedShader = makeHeapTestEffect();
      const geom = triGeom();
      const indexCounts: number[] = [];
      for (let i = 0; i < 100; i++) indexCounts.push(((i * 7) % 11) + 3);
      const draws: HeapDrawSpec[] = indexCounts.map((n) => {
        const idx = new Uint32Array(n);
        for (let i = 0; i < n; i++) idx[i] = i % 3;
        return {
          effect: sharedShader,
          inputs: {
            Positions:     geom.positions,
            Normals:       geom.normals,
            ModelTrafo:    trafoIdentity,
            Color:         v4(1, 0, 0, 1),
            ViewProjTrafo: trafoIdentity,
            LightLocation: v3(0, 0, 1),
          },
          indices: idx,
        };
      });

      const scene = buildHeapScene(device, sig, draws, {
        fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
        megacall: true,
      });

      scene.update(AdaptiveToken.top);
      let enc = device.createCommandEncoder({ label: "test/prep-many" });
      scene.encodeComputePrep(enc, AdaptiveToken.top);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();

      const debug = (scene as unknown as { _debug: { bucketsForTest(): readonly {
        indirectBuf: GPUBuffer | undefined;
        drawTableBuf: GPUBuffer | undefined;
        firstDrawInTileBuf: GPUBuffer | undefined;
        totalEmitEstimate: number;
        recordCount: number;
      }[] } })._debug;
      const dbgBuckets = debug.bucketsForTest();
      expect(dbgBuckets.length).toBe(1);
      const b = dbgBuckets[0]!;
      expect(b.recordCount).toBe(indexCounts.length);
      const indirect = await readU32(device, b.indirectBuf!, 16);
      const expectedTotal = indexCounts.reduce((a, c) => a + c, 0);
      expect(indirect[0]).toBe(expectedTotal);
      const drawTable = await readU32(device, b.drawTableBuf!, b.recordCount * 16);
      let acc = 0;
      for (let i = 0; i < indexCounts.length; i++) {
        expect(drawTable[i * 4 + 0]).toBe(acc);
        expect(drawTable[i * 4 + 3]).toBe(indexCounts[i]!);
        acc += indexCounts[i]!;
      }

      // Now add a dynamic draw and re-scan; verify prefix updates.
      const idx = new Uint32Array(9); for (let i = 0; i < 9; i++) idx[i] = i % 3;
      scene.addDraw({
        effect: sharedShader,
        inputs: {
          Positions: geom.positions, Normals: geom.normals,
          ModelTrafo: trafoIdentity, Color: v4(0, 1, 0, 1),
          ViewProjTrafo: trafoIdentity, LightLocation: v3(0, 0, 1),
        },
        indices: idx,
      });
      scene.update(AdaptiveToken.top);
      enc = device.createCommandEncoder({ label: "test/prep-after-add" });
      scene.encodeComputePrep(enc, AdaptiveToken.top);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      const indirect2 = await readU32(device, b.indirectBuf!, 16);
      expect(indirect2[0]).toBe(expectedTotal + 9);
      const drawTable2 = await readU32(device, b.drawTableBuf!, (indexCounts.length + 1) * 16);
      expect(drawTable2[indexCounts.length * 4 + 0]).toBe(expectedTotal);
      expect(drawTable2[indexCounts.length * 4 + 3]).toBe(9);

      // Render and check non-empty.
      const W = 32, H = 32;
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
      enc = device.createCommandEncoder({ label: "test/render-many" });
      scene.encodeComputePrep(enc, AdaptiveToken.top);
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: colorTex.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
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
      let nonClear = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i]! !== 0 || pixels[i + 1]! !== 0 || pixels[i + 2]! !== 0) nonClear++;
      }
      expect(errors).toEqual([]);
      expect(nonClear).toBeGreaterThan(0);

      scene.dispose();
      colorTex.destroy();
      depthTex.destroy();
    } finally {
      device.destroy();
    }
  }, 30000);

  it("populates firstDrawInTile with correct (lo, hi) per tile and sentinel slot", async () => {
    const TILE_K = 64;
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const sig = createFramebufferSignature({
        colors: { outColor: "rgba8unorm" },
        depthStencil: { format: "depth24plus" },
      });
      const sharedShader = makeHeapTestEffect();
      const geom = triGeom();
      // Mix of small and large indexCounts so we cover both single-tile
      // records and records that span many tiles. Total emit lands well
      // above 256 so we exercise multiple tiles.
      const indexCounts: number[] = [];
      for (let i = 0; i < 50; i++) indexCounts.push(((i * 17) % 23) + 3);
      indexCounts.push(300);
      indexCounts.push(150);
      for (let i = 0; i < 30; i++) indexCounts.push(((i * 13) % 19) + 5);
      const draws: HeapDrawSpec[] = indexCounts.map((n) => {
        const idx = new Uint32Array(n);
        for (let i = 0; i < n; i++) idx[i] = i % 3;
        return {
          effect: sharedShader,
          inputs: {
            Positions: geom.positions, Normals: geom.normals,
            ModelTrafo: trafoIdentity, Color: v4(0.5, 0.5, 1, 1),
            ViewProjTrafo: trafoIdentity, LightLocation: v3(0, 0, 1),
          },
          indices: idx,
        };
      });

      const scene = buildHeapScene(device, sig, draws, {
        fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
        megacall: true,
      });

      scene.update(AdaptiveToken.top);
      const enc = device.createCommandEncoder({ label: "test/firstDrawInTile" });
      scene.encodeComputePrep(enc, AdaptiveToken.top);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();

      const debug = (scene as unknown as { _debug: { bucketsForTest(): readonly {
        indirectBuf: GPUBuffer | undefined;
        drawTableBuf: GPUBuffer | undefined;
        firstDrawInTileBuf: GPUBuffer | undefined;
        totalEmitEstimate: number;
        recordCount: number;
      }[] } })._debug;
      const dbgBuckets = debug.bucketsForTest();
      expect(dbgBuckets.length).toBe(1);
      const b = dbgBuckets[0]!;
      const numRecords = b.recordCount;
      expect(numRecords).toBe(indexCounts.length);

      // CPU reference prefix-sum.
      const firstEmitRef: number[] = [];
      let acc = 0;
      for (const c of indexCounts) { firstEmitRef.push(acc); acc += c; }
      const totalEmit = acc;
      const numTiles = Math.ceil(totalEmit / TILE_K);

      const fdtBytes = (numTiles + 1) * 4;
      const fdt = await readU32(device, b.firstDrawInTileBuf!, fdtBytes);

      for (let t = 0; t < numTiles; t++) {
        const lo = fdt[t]!;
        // For each tile, the stored slot must satisfy:
        //   firstEmit[lo] <= t*TILE_K, AND
        //   if lo+1 < numRecords, firstEmit[lo+1] > t*TILE_K.
        const tileStart = t * TILE_K;
        expect(lo).toBeLessThan(numRecords);
        expect(firstEmitRef[lo]!).toBeLessThanOrEqual(tileStart);
        if (lo + 1 < numRecords) {
          expect(firstEmitRef[lo + 1]!).toBeGreaterThan(tileStart);
        }
      }
      // Sentinel slot.
      expect(fdt[numTiles]).toBe(numRecords);

      expect(errors).toEqual([]);
      scene.dispose();
    } finally {
      device.destroy();
    }
  }, 30000);
});
