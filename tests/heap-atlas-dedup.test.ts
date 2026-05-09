// §5b behavioural test: AtlasPool dedups distinct constant avals
// that wrap the same inner HostTextureSource. Two separate `acquire`
// calls with two different `aval<ITexture>` instances (both
// `isConstant === true`) but the same underlying source resolve to
// one sub-rect. Reactive (non-constant) avals fall through to
// identity-only as before.

import { describe, expect, it } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
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

    // Both releases must drop the entry.
    pool.release(a.ref);
    expect(pool.entry(av1)).toBeDefined();
    pool.release(b.ref);
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
