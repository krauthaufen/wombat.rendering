// AtlasPool under GPU memory pressure: a growth of a format's array whose
// allocation the device refuses (out of memory, reported asynchronously by
// popErrorScope) is undone — the old array stays the bound one, the format
// is capped there, the placements made on the refused array are lost (and
// their holders told), and later acquisitions take the full-atlas path
// (null) instead of landing in an invalid texture. A confirmed growth frees
// the old array only once confirmed. Also: atlas sizes from device limits.
//
// Before: the refused array was bound anyway; on a real device that made the
// heap's bind group invalid and every heap draw vanished (pro3d-web's Shaler
// scene on a GPU with ~1 GB free).

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { ITexture, type HostTextureSource } from "../packages/rendering/src/core/texture.js";
import {
  AtlasPool, atlasLimitsFor, setAtlasPageSize, setAtlasGrowthLayers, setAtlasMaxPages,
} from "../packages/rendering/src/runtime/textureAtlas/atlasPool.js";
import { onDeviceError, deviceErrors } from "../packages/rendering/src/runtime/deviceErrors.js";
import { MockGPU } from "./_mockGpu.js";

if (typeof (globalThis as { GPUTextureUsage?: unknown }).GPUTextureUsage === "undefined") {
  (globalThis as Record<string, unknown>).GPUTextureUsage = {
    COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04,
    STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10,
  };
}

class OutOfMemory extends Error { override name = "GPUOutOfMemoryError"; }

/** The mock device with error scopes: createTexture of an array with
 *  `refuseLayers` layers is refused (out of memory) by the scope around it. */
function scopedDevice(gpu: MockGPU, refuseLayers: Set<number>): GPUDevice {
  const d = gpu.device as unknown as Record<string, unknown>;
  const stack: { error: Error | null }[] = [];
  const create = d.createTexture as (desc: GPUTextureDescriptor) => GPUTexture;
  d.pushErrorScope = (_filter: string) => { stack.push({ error: null }); };
  d.popErrorScope = () => Promise.resolve(stack.pop()!.error);
  d.createTexture = (desc: GPUTextureDescriptor) => {
    const t = create(desc);
    const size = desc.size as GPUExtent3DDict;
    if (refuseLayers.has(size.depthOrArrayLayers ?? 1) && stack.length > 0) {
      stack[stack.length - 1]!.error = new OutOfMemory(`vkAllocateMemory failed (${desc.label})`);
    }
    return t;
  };
  return gpu.device;
}

/** A 1000² texture: four fit one 2048² page (1004² with gutters). */
function tile(): { av: ReturnType<typeof AVal.constant<ITexture>>; host: HostTextureSource } {
  const tex = ITexture.fromRaw({ data: new Uint8Array(4), width: 1000, height: 1000, format: "rgba8unorm" });
  if (tex.kind !== "host") throw new Error("expected host texture");
  return { av: AVal.constant(tex), host: tex.source };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

describe("AtlasPool out of memory", () => {
  let errors: ReturnType<typeof vi.spyOn>;
  beforeEach(() => {
    setAtlasPageSize(2048);
    setAtlasGrowthLayers(1);
    setAtlasMaxPages(16);
    errors = vi.spyOn(console, "error").mockImplementation(() => {});
  });
  afterEach(() => {
    delete (globalThis as Record<string, unknown>).__wombatAtlasPageSize;
    delete (globalThis as Record<string, unknown>).__wombatAtlasGrowthLayers;
    delete (globalThis as Record<string, unknown>).__wombatAtlasMaxPages;
    errors.mockRestore();
  });

  it("a refused growth is undone: old array kept, format capped, its placements lost and reported", async () => {
    const gpu = new MockGPU();
    const device = scopedDevice(gpu, new Set([2]));
    const pool = new AtlasPool(device);
    const added: number[] = [];
    const lost: number[] = [];
    pool.onPageAdded("rgba8unorm", (p) => added.push(p));
    pool.onLayersLost("rgba8unorm", (p) => lost.push(p));
    const reported: string[] = [];
    onDeviceError(device, (e) => reported.push(`${e.kind} ${e.recovered}`));

    const first = [tile(), tile(), tile(), tile()].map((t) =>
      pool.acquire("rgba8unorm", t.av, 1000, 1000, { source: { width: 1000, height: 1000, host: t.host } })!);
    expect(first.every((a) => a.pageId === 0)).toBe(true);
    await flush();
    const one = pool.textureFor("rgba8unorm")!;
    expect((one as unknown as { descriptor: GPUTextureDescriptor }).descriptor.size).toMatchObject({ depthOrArrayLayers: 1 });

    // the fifth needs page 1: the array grows to 2 layers, which the device refuses
    const t5 = tile();
    const fifth = pool.acquire("rgba8unorm", t5.av, 1000, 1000, { source: { width: 1000, height: 1000, host: t5.host } })!;
    expect(fifth.pageId).toBe(1);
    const two = pool.textureFor("rgba8unorm")!;
    expect(two).not.toBe(one);
    // the old array lives until the device has answered
    expect((one as unknown as { destroyed: boolean }).destroyed).toBe(false);
    await flush();

    expect(pool.textureFor("rgba8unorm")).toBe(one);
    expect((two as unknown as { destroyed: boolean }).destroyed).toBe(true);
    expect((one as unknown as { destroyed: boolean }).destroyed).toBe(false);
    expect(pool.capFor("rgba8unorm")).toBe(1);
    expect(pool.pagesFor("rgba8unorm").length).toBe(1);
    expect(pool.isLive(fifth.ref)).toBe(false);
    expect(first.every((a) => pool.isLive(a.ref))).toBe(true);
    expect(lost).toEqual([1]);
    expect(added.at(-1)).toBe(0); // bind groups rebuilt with the kept array
    expect(reported).toEqual(["out-of-memory true"]);
    expect(deviceErrors(device).at(-1)!.message).toMatch(/could not allocate 2 layers/);
    expect(errors).toHaveBeenCalledWith(expect.stringMatching(/GPU device error \(out-of-memory, recovered\)/));

    // full now: the next texture takes the full-atlas path, no growth
    const t6 = tile();
    const textures = gpu.textures.length;
    expect(pool.acquire("rgba8unorm", t6.av, 1000, 1000, { source: { width: 1000, height: 1000, host: t6.host } })).toBeNull();
    expect(gpu.textures.length).toBe(textures);

    // room again (a texture released and evicted): the lost one is placed again
    pool.release(first[0]!.ref);
    pool.release(first[0]!.ref);
    const again = pool.acquire("rgba8unorm", t5.av, 1000, 1000, { source: { width: 1000, height: 1000, host: t5.host } })!;
    expect(again.pageId).toBe(0);
    expect(pool.isLive(again.ref)).toBe(true);
  });

  it("placements made on a refused array in existing pages are lost too", async () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(scopedDevice(gpu, new Set([2])));
    const a = tile(), b = tile();
    const pa = pool.acquire("rgba8unorm", a.av, 1000, 1000, { source: { width: 1000, height: 1000, host: a.host } })!;
    // fill page 0 with three more, the fourth forces the growth
    const fill = [tile(), tile(), tile()].map((t) => pool.acquire("rgba8unorm", t.av, 1000, 1000, { source: { width: 1000, height: 1000, host: t.host } })!);
    await flush();
    const grown = pool.acquire("rgba8unorm", tile().av, 1000, 1000)!;
    expect(grown.pageId).toBe(1);
    // before the device answers, a slot frees on page 0 and a new texture takes it:
    // it is uploaded into the refused array only
    pool.release(fill[0]!.ref); pool.release(fill[0]!.ref); pool.evictIdle();
    const pb = pool.acquire("rgba8unorm", b.av, 1000, 1000, { source: { width: 1000, height: 1000, host: b.host } })!;
    expect(pb.pageId).toBe(0);
    await flush();
    expect(pool.isLive(pa.ref)).toBe(true);
    expect(pool.isLive(pb.ref)).toBe(false);
    expect(pool.isLive(grown.ref)).toBe(false);
  });

  it("a confirmed growth frees the old array only once confirmed", async () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(scopedDevice(gpu, new Set()));
    for (let i = 0; i < 4; i++) pool.acquire("rgba8unorm", tile().av, 1000, 1000);
    await flush();
    const one = pool.textureFor("rgba8unorm")!;
    expect(pool.acquire("rgba8unorm", tile().av, 1000, 1000)!.pageId).toBe(1);
    expect((one as unknown as { destroyed: boolean }).destroyed).toBe(false);
    await flush();
    expect((one as unknown as { destroyed: boolean }).destroyed).toBe(true);
    expect(pool.capFor("rgba8unorm")).toBe(16);
    // one layer per growth: the new array has two
    expect((pool.textureFor("rgba8unorm") as unknown as { descriptor: GPUTextureDescriptor }).descriptor.size).toMatchObject({ depthOrArrayLayers: 2 });
  });

  it("atlas sizes from device limits", () => {
    expect(atlasLimitsFor({ maxTextureDimension2D: 8192, maxTextureArrayLayers: 256 })).toEqual({ pageSize: 4096, maxPages: 16 });
    expect(atlasLimitsFor({ maxTextureDimension2D: 16384, maxTextureArrayLayers: 2048 })).toEqual({ pageSize: 4096, maxPages: 16 });
    expect(atlasLimitsFor({ maxTextureDimension2D: 2048, maxTextureArrayLayers: 8 })).toEqual({ pageSize: 2048, maxPages: 8 });
    expect(atlasLimitsFor(undefined)).toEqual({ pageSize: 4096, maxPages: 16 });
  });
});
