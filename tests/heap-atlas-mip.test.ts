// AtlasPool — mip pyramid (1.5×1 Iliffe layout) structural tests.
//
// Verifies the pool reserves the correct rect size for mipped vs
// non-mipped acquisitions, that mixing mip-modes coexists in one
// page, and that release frees the full pyramid footprint.
//
// No GPU mip-correctness is verified here (that needs the shader
// path; future PR). We only assert AtlasPool's allocator behaviour.

import { describe, expect, it } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { ITexture } from "../packages/rendering/src/core/texture.js";
import { AtlasPool, mipOffsetInPyramid, defaultMipCount } from "../packages/rendering/src/runtime/textureAtlas/atlasPool.js";
import { MockGPU } from "./_mockGpu.js";

// MockGPU doesn't pull in WebGPU's typed-constant globals; AtlasPool's
// `device.createTexture({ usage })` reads them from the global scope.
if (typeof (globalThis as { GPUTextureUsage?: unknown }).GPUTextureUsage === "undefined") {
  (globalThis as Record<string, unknown>).GPUTextureUsage = {
    COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04,
    STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10,
  };
}

const fakeAval = (w: number, h: number) =>
  AVal.constant(ITexture.fromRaw({
    data: new Uint8Array(w * h * 4),
    width: w,
    height: h,
    format: "rgba8unorm",
  }));

function packerUsedArea(pool: AtlasPool): number {
  let total = 0;
  for (const p of pool.pagesFor("rgba8unorm")) {
    for (const [, b] of p.packing.used) {
      const w = b.max.x - b.min.x + 1;
      const h = b.max.y - b.min.y + 1;
      total += w * h;
    }
  }
  return total;
}

describe("AtlasPool — mip pyramid layout", () => {
  it("a mipped acquire reserves a 1.5W × H rect", () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(gpu.device);
    const w = 256, h = 256;
    pool.acquire("rgba8unorm", fakeAval(w, h), w, h, { wantsMips: true });
    // Reserved area: ceil(w * 1.5) * h
    expect(packerUsedArea(pool)).toBe(Math.ceil(w * 1.5) * h);
  });

  it("non-mipped acquire reserves only W × H", () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(gpu.device);
    const w = 200, h = 100;
    pool.acquire("rgba8unorm", fakeAval(w, h), w, h);
    expect(packerUsedArea(pool)).toBe(w * h);
  });

  it("4 mipped + 2 non-mipped textures fit one page", () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(gpu.device);
    // 4 × 256² mipped + 2 × 256² plain — easy to fit in one 4096² page.
    for (let i = 0; i < 4; i++) {
      pool.acquire("rgba8unorm", fakeAval(256, 256), 256, 256, { wantsMips: true });
    }
    for (let i = 0; i < 2; i++) {
      pool.acquire("rgba8unorm", fakeAval(256, 256), 256, 256);
    }
    expect(pool.pagesFor("rgba8unorm").length).toBe(1);
    // 4 × (256*1.5*256) + 2 × (256*256) = 4*98304 + 2*65536 = 524288
    expect(packerUsedArea(pool)).toBe(4 * Math.ceil(256 * 1.5) * 256 + 2 * 256 * 256);
  });

  it("releasing a mipped texture frees the full 1.5W × H rect", () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(gpu.device);
    const w = 128, h = 128;
    const acq = pool.acquire("rgba8unorm", fakeAval(w, h), w, h, { wantsMips: true });
    expect(packerUsedArea(pool)).toBe(Math.ceil(w * 1.5) * h);
    pool.release(acq.ref);
    expect(packerUsedArea(pool)).toBe(0);
  });

  it("acquire returns mip-0 origin/size in normalized atlas coords", () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(gpu.device);
    const w = 256, h = 256;
    const acq = pool.acquire("rgba8unorm", fakeAval(w, h), w, h, { wantsMips: true });
    expect(acq.numMips).toBe(defaultMipCount(w, h));
    // Size of mip 0 is W/4096, H/4096 (normalized).
    expect(acq.size.x).toBeCloseTo(w / 4096);
    expect(acq.size.y).toBeCloseTo(h / 4096);
    // Origin must lie inside [0,1).
    expect(acq.origin.x).toBeGreaterThanOrEqual(0);
    expect(acq.origin.y).toBeGreaterThanOrEqual(0);
    expect(acq.origin.x).toBeLessThan(1);
    expect(acq.origin.y).toBeLessThan(1);
  });

  it("mipOffsetInPyramid: mip 0 at (0,0); mips 1..N stack vertically at x=W", () => {
    const W = 256, H = 256;
    expect(mipOffsetInPyramid(W, H, 0)).toEqual({ x: 0, y: 0 });
    expect(mipOffsetInPyramid(W, H, 1)).toEqual({ x: W, y: 0 });
    // Mip 2 below mip 1 → y = H/2 = 128.
    expect(mipOffsetInPyramid(W, H, 2)).toEqual({ x: W, y: 128 });
    // Mip 3 below mip 2 → y = 128 + 64 = 192.
    expect(mipOffsetInPyramid(W, H, 3)).toEqual({ x: W, y: 192 });
  });

  it("identity sharing: same aval returns same ref/origin under wantsMips", () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(gpu.device);
    const av = fakeAval(64, 64);
    const a = pool.acquire("rgba8unorm", av, 64, 64, { wantsMips: true });
    const b = pool.acquire("rgba8unorm", av, 64, 64, { wantsMips: true });
    expect(a.ref).toBe(b.ref);
    expect(a.origin.x).toBe(b.origin.x);
    expect(a.origin.y).toBe(b.origin.y);
    // First release decrements; second frees.
    pool.release(a.ref);
    expect(packerUsedArea(pool)).toBeGreaterThan(0);
    pool.release(b.ref);
    expect(packerUsedArea(pool)).toBe(0);
  });
});
