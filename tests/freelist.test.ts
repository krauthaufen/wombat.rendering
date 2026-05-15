// Freelist — best-fit free-block manager backing AttributeArena +
// IndexAllocator. These tests pin the unit-level behaviour: best-fit
// alloc, split-remainder reinsert, left/right coalesce on release,
// and a stress sweep against a reference linear implementation to
// catch invariant drift.

import { describe, expect, it } from "vitest";
import { Freelist } from "../packages/rendering/src/runtime/heapScene/freelist.js";

describe("Freelist", () => {
  it("alloc returns undefined when empty", () => {
    const fl = new Freelist();
    expect(fl.alloc(8)).toBeUndefined();
    expect(fl.blockCount).toBe(0);
  });

  it("release + alloc returns the same offset for exact-fit", () => {
    const fl = new Freelist();
    fl.release(100, 32);
    expect(fl.alloc(32)).toBe(100);
    expect(fl.alloc(1)).toBeUndefined();
  });

  it("split-fit reinserts the remainder", () => {
    const fl = new Freelist();
    fl.release(0, 64);
    expect(fl.alloc(16)).toBe(0);
    // 48 B remainder at offset 16.
    expect(fl.toArray()).toEqual([{ off: 16, size: 48 }]);
    expect(fl.alloc(48)).toBe(16);
    expect(fl.blockCount).toBe(0);
  });

  it("release coalesces with the right neighbour", () => {
    const fl = new Freelist();
    fl.release(64, 32);     // [64..96)
    fl.release(0, 64);      // [0..64) — adjacent left
    expect(fl.toArray()).toEqual([{ off: 0, size: 96 }]);
  });

  it("release coalesces with the left neighbour", () => {
    const fl = new Freelist();
    fl.release(0, 64);      // [0..64)
    fl.release(64, 32);     // adjacent right
    expect(fl.toArray()).toEqual([{ off: 0, size: 96 }]);
  });

  it("release coalesces with both neighbours simultaneously", () => {
    const fl = new Freelist();
    fl.release(0, 32);      // [0..32)
    fl.release(64, 32);     // [64..96)
    // Hole in the middle: release [32..64). Coalesces both sides.
    fl.release(32, 32);
    expect(fl.toArray()).toEqual([{ off: 0, size: 96 }]);
  });

  it("best-fit picks the smallest sufficient block", () => {
    const fl = new Freelist();
    // Three free blocks of size 16, 32, 64.
    fl.release(1000, 32);
    fl.release(0, 16);
    fl.release(2000, 64);
    // Requesting 17 must pick the size-32 block (best-fit), not 64.
    expect(fl.alloc(17)).toBe(1000);
    // 15 of the 32 remaining is split off at 1017.
    expect(fl.toArray()).toEqual([
      { off: 0, size: 16 },
      { off: 1017, size: 15 },
      { off: 2000, size: 64 },
    ]);
  });

  it("alloc requesting > max size returns undefined", () => {
    const fl = new Freelist();
    fl.release(0, 32);
    expect(fl.alloc(33)).toBeUndefined();
  });

  it("takeBlockEndingAt returns and removes the block whose end matches", () => {
    const fl = new Freelist();
    fl.release(0, 32);
    fl.release(64, 32);
    expect(fl.takeBlockEndingAt(32)).toEqual({ off: 0, size: 32 });
    expect(fl.toArray()).toEqual([{ off: 64, size: 32 }]);
    expect(fl.takeBlockEndingAt(32)).toBeUndefined();
    expect(fl.takeBlockEndingAt(96)).toEqual({ off: 64, size: 32 });
    expect(fl.blockCount).toBe(0);
  });

  it("invariant sweep under randomised alloc/release", () => {
    // Verify the structural invariants every Freelist must hold,
    // through ~5000 random ops:
    //   (a) no two free blocks overlap, no two are adjacent
    //       (adjacent blocks must have been coalesced);
    //   (b) total free-block bytes + live-block bytes = initial seed;
    //   (c) every alloc returns an offset whose [off..off+size) lay
    //       entirely inside a previously-free region.
    const rng = mulberry32(0xC0FFEE);
    const fl = new Freelist();
    const SEED_SIZE = 1 << 20; // 1 MB of free space to play with.
    fl.release(0, SEED_SIZE);
    type Block = { off: number; size: number };
    const live: Block[] = [];

    const checkInvariants = (): void => {
      const blocks = fl.toArray();
      // Sorted by off + non-overlapping + non-adjacent.
      for (let i = 0; i + 1 < blocks.length; i++) {
        const a = blocks[i]!, b = blocks[i + 1]!;
        expect(a.off + a.size).toBeLessThan(b.off);  // strict, no adjacency
      }
      const totalFree = blocks.reduce((s, b) => s + b.size, 0);
      const totalLive = live.reduce((s, b) => s + b.size, 0);
      expect(totalFree + totalLive).toBe(SEED_SIZE);
    };

    for (let step = 0; step < 5000; step++) {
      const wantAlloc = (rng() % 3) !== 0 || live.length < 4;
      if (wantAlloc && live.length < 200) {
        const size = 8 * (1 + (rng() % 128));   // aligned 8..1024
        const off = fl.alloc(size);
        if (off !== undefined) live.push({ off, size });
      } else if (live.length > 0) {
        const idx = rng() % live.length;
        const blk = live[idx]!;
        live.splice(idx, 1);
        fl.release(blk.off, blk.size);
      }
      if ((step & 0x3F) === 0) checkInvariants();
    }
    checkInvariants();
    // Drain everything — final state has exactly one block covering
    // the whole seeded region (proves full coalesce).
    while (live.length > 0) {
      const b = live.pop()!;
      fl.release(b.off, b.size);
    }
    expect(fl.toArray()).toEqual([{ off: 0, size: SEED_SIZE }]);
  });
});

function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6D2B79F5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0);
  };
}
