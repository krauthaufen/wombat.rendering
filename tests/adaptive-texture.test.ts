// prepareAdaptiveTexture — verify lifecycle and upload behaviour
// against a mock GPUDevice.

import { describe, expect, it } from "vitest";
import { AdaptiveToken, cval, transact } from "@aardworx/wombat.adaptive";
import { ITexture } from "@aardworx/wombat.rendering/core";
import { prepareAdaptiveTexture } from "@aardworx/wombat.rendering/resources";
import { MockGPU, type MockTexture } from "./_mockGpu.js";

function asMock(t: GPUTexture): MockTexture {
  return t as unknown as MockTexture;
}

describe("prepareAdaptiveTexture", () => {
  it("raw host source: allocates and uploads", () => {
    const gpu = new MockGPU();
    const data = new Uint8Array(4 * 4 * 4); // 4x4 rgba8 = 64 bytes
    const src = cval(ITexture.fromRaw({ data, width: 4, height: 4, format: "rgba8unorm" }));
    const r = prepareAdaptiveTexture(gpu.device, src, { label: "albedo" });
    r.acquire();
    const t = asMock(r.getValue(AdaptiveToken.top));
    expect(t.id).toBe(1);
    expect(t.descriptor.format).toBe("rgba8unorm");
    expect(gpu.writeTextureCalls).toHaveLength(1);
    expect(gpu.writeTextureCalls[0]!.dataLayout.bytesPerRow).toBe(16);
    expect(gpu.writeTextureCalls[0]!.size).toEqual({ width: 4, height: 4, depthOrArrayLayers: 1 });
    r.release();
    expect(t.destroyed).toBe(true);
  });

  it("same-shape update reuses the texture", () => {
    const gpu = new MockGPU();
    const src = cval(ITexture.fromRaw({ data: new Uint8Array(16), width: 2, height: 2, format: "rgba8unorm" }));
    const r = prepareAdaptiveTexture(gpu.device, src, {});
    r.acquire();
    const a = asMock(r.getValue(AdaptiveToken.top));
    transact(() => {
      src.value = ITexture.fromRaw({ data: new Uint8Array(16), width: 2, height: 2, format: "rgba8unorm" });
    });
    const b = asMock(r.getValue(AdaptiveToken.top));
    expect(a.id).toBe(b.id);
    expect(gpu.textures).toHaveLength(1);
    expect(gpu.writeTextureCalls).toHaveLength(2);
    r.release();
  });

  it("size change reallocates", () => {
    const gpu = new MockGPU();
    const src = cval(ITexture.fromRaw({ data: new Uint8Array(16), width: 2, height: 2, format: "rgba8unorm" }));
    const r = prepareAdaptiveTexture(gpu.device, src, {});
    r.acquire();
    const a = asMock(r.getValue(AdaptiveToken.top));
    transact(() => {
      src.value = ITexture.fromRaw({ data: new Uint8Array(64), width: 4, height: 4, format: "rgba8unorm" });
    });
    const b = asMock(r.getValue(AdaptiveToken.top));
    expect(a.id).not.toBe(b.id);
    expect(gpu.textures).toHaveLength(2);
    expect(gpu.textures[0]!.destroyed).toBe(true);
    r.release();
  });

  it("gpu source: passes through, no allocation", () => {
    const gpu = new MockGPU();
    const userTex = { __user: true } as unknown as GPUTexture;
    const src = cval(ITexture.fromGPU(userTex));
    const r = prepareAdaptiveTexture(gpu.device, src, {});
    r.acquire();
    expect(r.getValue(AdaptiveToken.top)).toBe(userTex);
    expect(gpu.textures).toHaveLength(0);
    r.release();
  });
});
