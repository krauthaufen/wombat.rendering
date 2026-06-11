// DirtyRanges — disjoint dirty-interval tracking behind the arena
// shadow flushes. The single min..max span it replaces re-uploaded
// nearly the whole arena under scattered edits (68k-object CAD bench:
// 16–28 ms/frame for a few KB of actual dirty bytes).

import { describe, expect, it } from "vitest";
import { DirtyRanges } from "../packages/rendering/src/runtime/heapScene/dirtyRanges.js";

const drained = (d: DirtyRanges): Array<[number, number]> => {
  const out: Array<[number, number]> = [];
  d.drain((s, e) => out.push([s, e]));
  return out;
};

describe("DirtyRanges", () => {
  it("keeps distant writes as separate ranges", () => {
    const d = new DirtyRanges(16, 64);
    d.add(0, 8);
    d.add(1000, 1008);
    d.add(500_000, 500_004);
    expect(drained(d)).toEqual([[0, 8], [1000, 1008], [500_000, 500_004]]);
  });

  it("merges overlapping and gap-adjacent writes", () => {
    const d = new DirtyRanges(16, 64);
    d.add(100, 120);
    d.add(110, 140);          // overlap
    d.add(150, 160);          // gap 10 ≤ 16 → merge
    d.add(200, 210);          // gap 40 > 16 → separate
    expect(drained(d)).toEqual([[100, 160], [200, 210]]);
  });

  it("merges a write bridging several existing ranges", () => {
    const d = new DirtyRanges(0, 64);
    d.add(0, 10);
    d.add(100, 110);
    d.add(200, 210);
    d.add(5, 205);            // spans all three
    expect(drained(d)).toEqual([[0, 210]]);
  });

  it("inserts in sorted order regardless of write order", () => {
    const d = new DirtyRanges(0, 64);
    d.add(900, 910);
    d.add(100, 110);
    d.add(500, 510);
    expect(drained(d)).toEqual([[100, 110], [500, 510], [900, 910]]);
  });

  it("compacts to maxRanges by merging closest pairs", () => {
    const d = new DirtyRanges(0, 4);
    // 6 ranges; gaps: 10, 1, 10, 1, 10 — the two gap-1 pairs merge first
    d.add(0, 10);
    d.add(20, 30);
    d.add(31, 40);
    d.add(50, 60);
    d.add(61, 70);
    d.add(80, 90);
    expect(d.count).toBeLessThanOrEqual(4);
    expect(drained(d)).toEqual([[0, 10], [20, 40], [50, 70], [80, 90]]);
  });

  it("drain clears state", () => {
    const d = new DirtyRanges();
    d.add(0, 4);
    drained(d);
    expect(d.isEmpty).toBe(true);
    d.add(8, 12);
    expect(drained(d)).toEqual([[8, 12]]);
  });

  it("ignores empty intervals", () => {
    const d = new DirtyRanges();
    d.add(5, 5);
    expect(d.isEmpty).toBe(true);
  });

  it("scattered-edit shape: k random small writes stay k ranges, not one span", () => {
    const d = new DirtyRanges(4096, 64);
    // 32 writes of 64B spread 100KB apart — the old span tracker
    // would flush ~3.2MB; ranges flush 32 × 64B.
    for (let i = 0; i < 32; i++) d.add(i * 100_000, i * 100_000 + 64);
    let total = 0;
    d.drain((s, e) => { total += e - s; });
    expect(total).toBe(32 * 64);
  });
});
