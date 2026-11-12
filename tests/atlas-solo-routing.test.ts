// Which page a host texture gets (docs/heap-large-textures.md): the shared
// atlas pages up to ATLAS_MAX_DIM, a solo page of its own above that while
// its reserved rect (gutters, and with mips the column beside mip 0) fits
// SOLO_MAX_DIM, else none (the legacy renderer).

import { describe, expect, it } from "vitest";
import { AtlasPool, ATLAS_MAX_DIM, SOLO_MAX_DIM, reservedSize } from "../packages/rendering/src/runtime/textureAtlas/atlasPool.js";

describe("atlas / solo / legacy routing by size", () => {
  it("up to the cap: the shared pages, not solo", () => {
    expect(AtlasPool.eligibleSize(ATLAS_MAX_DIM, ATLAS_MAX_DIM)).toBe(true);
    expect(AtlasPool.soloSize(ATLAS_MAX_DIM, ATLAS_MAX_DIM)).toBe(false);
  });
  it("over the cap: a solo page", () => {
    expect(AtlasPool.eligibleSize(ATLAS_MAX_DIM + 1, 8)).toBe(false);
    expect(AtlasPool.soloSize(ATLAS_MAX_DIM + 1, 8)).toBe(true);
    expect(AtlasPool.soloSize(2432, 2432, true)).toBe(true);
    expect(AtlasPool.soloSize(4096, 4096, true)).toBe(true);
  });
  it("a solo page must fit its reserved rect: mips count", () => {
    const w = SOLO_MAX_DIM - 4;
    expect(reservedSize(w, 4, 1).w).toBe(SOLO_MAX_DIM);
    expect(AtlasPool.soloSize(w, 4, false)).toBe(true);
    expect(AtlasPool.soloSize(w, 4, true)).toBe(false);
    expect(AtlasPool.soloSize(SOLO_MAX_DIM, 4, false)).toBe(false);
  });
  it("no size: neither", () => {
    expect(AtlasPool.soloSize(0, 4)).toBe(false);
  });
});
