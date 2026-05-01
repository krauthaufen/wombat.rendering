// Mip-map generation on real GPU. Allocate a 4×4 rgba8unorm
// texture with 3 mip levels, fill mip 0 with a known pattern,
// run generateMips to populate mips 1 + 2, and read both back to
// verify the 2× downsample produces the expected averages.

import { describe, expect, it } from "vitest";
import { generateMips } from "@aardworx/wombat.rendering/resources";
import { requestRealDevice } from "./_realGpu.js";

async function readMip(device: GPUDevice, tex: GPUTexture, mip: number): Promise<Uint8Array> {
  const w = Math.max(1, tex.width >> mip);
  const h = Math.max(1, tex.height >> mip);
  const bpr = Math.ceil((w * 4) / 256) * 256;
  const staging = device.createBuffer({
    size: bpr * h,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer(
    { texture: tex, mipLevel: mip },
    { buffer: staging, bytesPerRow: bpr, rowsPerImage: h },
    { width: w, height: h, depthOrArrayLayers: 1 },
  );
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    out.set(padded.subarray(y * bpr, y * bpr + w * 4), y * w * 4);
  }
  return out;
}

describe("generateMips — real GPU", () => {
  it("populates mip 1 + mip 2 of a 4×4 rgba8unorm texture from mip 0", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const tex = device.createTexture({
        size: { width: 4, height: 4, depthOrArrayLayers: 1 },
        format: "rgba8unorm",
        mipLevelCount: 3,
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC
             | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      });

      // Mip 0: uniform red (255, 0, 0, 255) on every pixel.
      const mip0 = new Uint8Array(4 * 4 * 4);
      for (let i = 0; i < 16; i++) {
        mip0[i * 4 + 0] = 255;
        mip0[i * 4 + 1] = 0;
        mip0[i * 4 + 2] = 0;
        mip0[i * 4 + 3] = 255;
      }
      device.queue.writeTexture(
        { texture: tex, mipLevel: 0 },
        mip0,
        { bytesPerRow: 16, rowsPerImage: 4 },
        { width: 4, height: 4, depthOrArrayLayers: 1 },
      );

      // Generate mips 1 and 2.
      const enc = device.createCommandEncoder();
      generateMips(device, enc, tex);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();

      const mip1 = await readMip(device, tex, 1);
      const mip2 = await readMip(device, tex, 2);

      // Each downsample averages 4 identical red pixels → still red.
      expect(mip1.length).toBe(2 * 2 * 4);
      for (let i = 0; i < 4; i++) {
        expect(mip1[i * 4 + 0]).toBe(255);
        expect(mip1[i * 4 + 1]).toBe(0);
        expect(mip1[i * 4 + 2]).toBe(0);
        expect(mip1[i * 4 + 3]).toBe(255);
      }
      expect(mip2.length).toBe(1 * 1 * 4);
      expect(mip2[0]).toBe(255);
      expect(mip2[1]).toBe(0);
      expect(mip2[2]).toBe(0);
      expect(mip2[3]).toBe(255);

      expect(errors).toEqual([]);
      tex.destroy();
    } finally {
      device.destroy();
    }
  }, 30000);

  it("averages a 2×2 checkerboard down to (127, 127, 127, 255)", async () => {
    const device = await requestRealDevice();
    try {
      const tex = device.createTexture({
        size: { width: 2, height: 2, depthOrArrayLayers: 1 },
        format: "rgba8unorm",
        mipLevelCount: 2,
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC
             | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      });
      // 2 black + 2 white pixels.
      const mip0 = new Uint8Array([
        255, 255, 255, 255,    0,   0,   0, 255,
          0,   0,   0, 255,  255, 255, 255, 255,
      ]);
      device.queue.writeTexture(
        { texture: tex, mipLevel: 0 }, mip0,
        { bytesPerRow: 8, rowsPerImage: 2 },
        { width: 2, height: 2, depthOrArrayLayers: 1 },
      );
      const enc = device.createCommandEncoder();
      generateMips(device, enc, tex);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();

      const mip1 = await readMip(device, tex, 1);
      // Average of 255+0+0+255 = 510/4 = 127.5 → 127 after rounding to u8.
      expect(Math.abs(mip1[0]! - 127)).toBeLessThan(2);
      expect(Math.abs(mip1[1]! - 127)).toBeLessThan(2);
      expect(Math.abs(mip1[2]! - 127)).toBeLessThan(2);
      expect(mip1[3]).toBe(255);
      tex.destroy();
    } finally {
      device.destroy();
    }
  }, 30000);
});
