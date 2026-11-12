// §5b behavioural test: AtlasPool dedups distinct constant avals
// that wrap the same inner HostTextureSource. Two separate `acquire`
// calls with two different `aval<ITexture>` instances (both
// `isConstant === true`) but the same underlying source resolve to
// one sub-rect. Changing (non-constant) avals share by source too:
// N draws that each read "their" texture from a cell of their own
// (PoEtry's passive tree: 8,207 draws, 13 sheets) take one sub-rect
// per image, and a changing aval leaves the shared one when its value
// changes (`repack`), with exactly the refs it holds.

import { describe, expect, it } from "vitest";
import { AVal, cval } from "@aardworx/wombat.adaptive";
import { ITexture, type HostTextureSource } from "../packages/rendering/src/core/texture.js";
import { AtlasPool } from "../packages/rendering/src/runtime/textureAtlas/atlasPool.js";
import { MockGPU } from "./_mockGpu.js";

if (typeof (globalThis as { GPUTextureUsage?: unknown }).GPUTextureUsage === "undefined") {
  (globalThis as Record<string, unknown>).GPUTextureUsage = {
    COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04,
    STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10,
  };
}

function makeRawTexture(): { tex: ITexture; host: HostTextureSource } {
  const tex = ITexture.fromRaw({
    data: new Uint8Array(64 * 64 * 4),
    width: 64, height: 64,
    format: "rgba8unorm",
  });
  // ITexture.fromRaw stores the descriptor as `source` of kind "raw".
  if (tex.kind !== "host") throw new Error("expected host texture");
  return { tex, host: tex.source };
}

describe("[plugin/behavioural] AtlasPool §5b dedup", () => {
  it("two distinct constant avals wrapping the same host source → one sub-rect", () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(gpu.device);
    const { tex, host } = makeRawTexture();

    const av1 = AVal.constant(tex);
    const av2 = AVal.constant(tex);
    expect(av1).not.toBe(av2);
    expect(av1.isConstant).toBe(true);

    const a = pool.acquire("rgba8unorm", av1, 64, 64, {
      source: { width: 64, height: 64, host },
    });
    const b = pool.acquire("rgba8unorm", av2, 64, 64, {
      source: { width: 64, height: 64, host },
    });
    // Same sub-rect, same ref → deduped.
    expect(b.ref).toBe(a.ref);
    expect(b.pageId).toBe(a.pageId);

    // Both releases must drop the entry — but only after eviction,
    // since released entries stay in the LRU for resurrection.
    pool.release(a.ref);
    expect(pool.entry(av1)).toBeDefined();
    pool.release(b.ref);
    expect(pool.entry(av1)).toBeDefined();
    pool.evictIdle();
    expect(pool.entry(av1)).toBeUndefined();
    expect(pool.entry(av2)).toBeUndefined();
  });

  it("different host sources with same dimensions → distinct sub-rects", () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(gpu.device);
    const { tex: t1, host: h1 } = makeRawTexture();
    const { tex: t2, host: h2 } = makeRawTexture();
    expect(h1).not.toBe(h2);

    const av1 = AVal.constant(t1);
    const av2 = AVal.constant(t2);
    const a = pool.acquire("rgba8unorm", av1, 64, 64, {
      source: { width: 64, height: 64, host: h1 },
    });
    const b = pool.acquire("rgba8unorm", av2, 64, 64, {
      source: { width: 64, height: 64, host: h2 },
    });
    expect(b.ref).not.toBe(a.ref);
  });
});

describe("AtlasPool: changing avals share a sub-rect by source", () => {
  const src = (host: HostTextureSource) => ({ source: { width: 64, height: 64, host } });

  it("many changing avals with the same host source → one sub-rect", () => {
    const pool = new AtlasPool(new MockGPU().device);
    const { tex, host } = makeRawTexture();
    const avals = Array.from({ length: 50 }, () => cval(tex));
    const refs = avals.map((av) => pool.acquire("rgba8unorm", av, 64, 64, src(host))!.ref);
    expect(new Set(refs).size).toBe(1);
    for (const av of avals) pool.release(refs[0]!, av);
    expect(pool.evictIdle()).toBe(1);
  });

  it("an aval whose value changes leaves the shared sub-rect with its own refs only", () => {
    const pool = new AtlasPool(new MockGPU().device);
    const a = makeRawTexture(), b = makeRawTexture();
    const av1 = cval(a.tex), av2 = cval(a.tex);
    const r1 = pool.acquire("rgba8unorm", av1, 64, 64, src(a.host))!.ref;
    expect(pool.acquire("rgba8unorm", av2, 64, 64, src(a.host))!.ref).toBe(r1);
    // av1 is used by three draws, av2 by one
    expect(pool.incRef(r1, av1)).toBe(true);
    expect(pool.incRef(r1, av1)).toBe(true);

    const moved = pool.repack(av1, b.tex)!;
    expect(moved.ref).not.toBe(r1);
    expect(pool.entry(av1)!.ref).toBe(moved.ref);
    expect(pool.entry(av2)!.ref).toBe(r1);          // untouched for the other aval

    // av2's single release empties the old sub-rect …
    pool.release(r1, av2);
    expect(pool.evictIdle()).toBe(1);
    expect(pool.isLive(r1)).toBe(false);
    // … and the new one holds av1's three refs: it outlives two releases.
    pool.release(moved.ref, av1);
    pool.release(moved.ref, av1);
    expect(pool.evictIdle()).toBe(0);
    pool.release(moved.ref, av1);
    expect(pool.evictIdle()).toBe(1);
  });

  it("avals that change to the same new source share again (placeholder → loaded image)", () => {
    const pool = new AtlasPool(new MockGPU().device);
    const checker = makeRawTexture(), image = makeRawTexture();
    const avals = Array.from({ length: 20 }, () => cval(checker.tex));
    const first = avals.map((av) => pool.acquire("rgba8unorm", av, 64, 64, src(checker.host))!.ref);
    expect(new Set(first).size).toBe(1);
    const second = avals.map((av) => pool.repack(av, image.tex)!.ref);
    expect(new Set(second).size).toBe(1);
    expect(second[0]).not.toBe(first[0]);
    // the placeholder's sub-rect went with its last aval, the image's holds all 20
    expect(pool.isLive(first[0]!)).toBe(false);
    expect(pool.evictIdle()).toBe(0);
    for (const av of avals) pool.release(second[0]!, av);
    expect(pool.evictIdle()).toBe(1);
  });

  it("an aval that ticks with the same source keeps the shared sub-rect", () => {
    const pool = new AtlasPool(new MockGPU().device);
    const { tex, host } = makeRawTexture();
    const av1 = cval(tex), av2 = cval(tex);
    const r = pool.acquire("rgba8unorm", av1, 64, 64, src(host))!.ref;
    pool.acquire("rgba8unorm", av2, 64, 64, src(host));
    expect(pool.repack(av1, tex)!.ref).toBe(r);
    expect(pool.entry(av2)!.ref).toBe(r);
    pool.release(r, av1); pool.release(r, av2);
    expect(pool.evictIdle()).toBe(1);
  });

  it("the last aval on a sub-rect that changes frees it for the new image", () => {
    const pool = new AtlasPool(new MockGPU().device);
    const a = makeRawTexture(), b = makeRawTexture();
    const av = cval(a.tex);
    const r = pool.acquire("rgba8unorm", av, 64, 64, src(a.host))!.ref;
    const moved = pool.repack(av, b.tex)!;
    expect(pool.isLive(r)).toBe(false);
    pool.release(moved.ref, av);
    expect(pool.evictIdle()).toBe(1);
  });
});
