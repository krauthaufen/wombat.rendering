// Real-GPU test for the GPU-resident atlas source (adapter-packed textures,
// e.g. the cube face-grid): `AtlasSource.gpuTexture` uploads into a page via
// copyTextureToTexture, and — crucially — an evicted + re-acquired entry
// RE-UPLOADS the same pixels (the retained GPUTexture is copied afresh).
// Without the re-upload the re-acquired sub-rect would come back blank.

import { describe, expect, it } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { ITexture } from "@aardworx/wombat.rendering/core";
import { AtlasPool } from "../packages/rendering/src/runtime/textureAtlas/atlasPool.js";
import { requestRealDevice } from "./_realGpu.js";

// Read a W×H sub-rect at (x,y) of layer `pageId` of an atlas page array.
async function readSubRect(
  device: GPUDevice, page: GPUTexture, pageId: number,
  x: number, y: number, w: number, h: number,
): Promise<Uint8Array> {
  const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
  const staging = device.createBuffer({
    size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer(
    { texture: page, origin: { x, y, z: pageId } },
    { buffer: staging, bytesPerRow, rowsPerImage: h },
    { width: w, height: h, depthOrArrayLayers: 1 },
  );
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(staging.getMappedRange().slice(0));
  staging.unmap(); staging.destroy();
  const out = new Uint8Array(w * h * 4);
  for (let r = 0; r < h; r++) out.set(padded.subarray(r * bytesPerRow, r * bytesPerRow + w * 4), r * w * 4);
  return out;
}

describe("atlas GPU-source upload + eviction re-upload", () => {
  it("acquire → evict → re-acquire yields the same pixels", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const S = 64;
      // Solid-magenta GPU source (stands in for the adapter-packed grid).
      const src = device.createTexture({
        size: { width: S, height: S }, format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
      });
      const data = new Uint8Array(S * S * 4);
      for (let i = 0; i < S * S; i++) { data[i * 4] = 255; data[i * 4 + 2] = 255; data[i * 4 + 3] = 255; } // magenta
      device.queue.writeTexture({ texture: src }, data, { bytesPerRow: S * 4, rowsPerImage: S }, { width: S, height: S });

      const pool = new AtlasPool(device);
      const aval = AVal.constant(ITexture.fromGPU(src));

      const acq1 = pool.acquire("rgba8unorm", aval, S, S, {
        source: { width: S, height: S, gpuTexture: src },
      });
      expect(acq1).not.toBeNull();
      await device.queue.onSubmittedWorkDone();

      const px1 = await readSubRect(device, acq1!.page.texture, acq1!.pageId, acq1!.origin.x + 8, acq1!.origin.y + 8, 4, 4);
      expect([px1[0], px1[1], px1[2]]).toEqual([255, 0, 255]); // magenta uploaded

      // Release → idle in LRU, then force eviction so the entry is gone.
      pool.release(acq1!.ref);
      pool.evictIdle();

      // Re-acquire the SAME aval — a fresh acquire (entry evicted) that must
      // re-upload from the retained GPUTexture.
      const acq2 = pool.acquire("rgba8unorm", aval, S, S, {
        source: { width: S, height: S, gpuTexture: src },
      });
      expect(acq2).not.toBeNull();
      await device.queue.onSubmittedWorkDone();

      const px2 = await readSubRect(device, acq2!.page.texture, acq2!.pageId, acq2!.origin.x + 8, acq2!.origin.y + 8, 4, 4);
      expect([px2[0], px2[1], px2[2]]).toEqual([255, 0, 255]); // SAME pixels after re-acquire

      expect(errors).toEqual([]);
    } finally {
      device.destroy();
    }
  });
});
