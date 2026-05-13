import { describe, test, expect } from "vitest";
import { partitionCPU } from "@aardworx/wombat.rendering/runtime";

describe("derivedModes/partition (CPU reference)", () => {
  test("single slot: trivial partition", () => {
    const r = partitionCPU({
      slot:      [0, 0, 0],
      emitCount: [10, 20, 30],
      slotCount: 1,
    });
    expect(Array.from(r.slotCounts)).toEqual([3]);
    expect(Array.from(r.slotOffsets)).toEqual([0, 3]);
    expect(Array.from(r.slotTotalEmit)).toEqual([60]);
    expect(Array.from(r.slotRecords)).toEqual([0, 1, 2]);
    // slot 0's prefix: [0, 10, 30, 60]
    expect(Array.from(r.slotEmitPrefix.slice(0, 4))).toEqual([0, 10, 30, 60]);
  });

  test("two slots: records group correctly, stable within slot", () => {
    const r = partitionCPU({
      slot:      [0, 1, 0, 1, 0],
      emitCount: [3, 5, 7, 11, 13],
      slotCount: 2,
    });
    expect(Array.from(r.slotCounts)).toEqual([3, 2]);
    expect(Array.from(r.slotOffsets)).toEqual([0, 3, 5]);
    expect(Array.from(r.slotTotalEmit)).toEqual([3 + 7 + 13, 5 + 11]);
    // Slot 0 holds records [0, 2, 4]; slot 1 holds [1, 3].
    expect(Array.from(r.slotRecords)).toEqual([0, 2, 4, 1, 3]);
    // Slot 0 prefix: [0, 3, 10, 23] at indices 0..3
    expect(Array.from(r.slotEmitPrefix.slice(0, 4))).toEqual([0, 3, 10, 23]);
    // Slot 1 prefix: [0, 5, 16] at indices 4..6 (3 records * 1 slot offset before)
    // The prefix layout: slot s starts at slotOffsets[s] + s.
    // slot 1 starts at 3 + 1 = 4, holds 3 entries (count+1).
    expect(Array.from(r.slotEmitPrefix.slice(4, 7))).toEqual([0, 5, 16]);
  });

  test("empty slot in the middle is OK (zero count, zero emit)", () => {
    const r = partitionCPU({
      slot:      [0, 2, 2],
      emitCount: [1, 2, 4],
      slotCount: 3,
    });
    expect(Array.from(r.slotCounts)).toEqual([1, 0, 2]);
    expect(Array.from(r.slotOffsets)).toEqual([0, 1, 1, 3]);
    expect(Array.from(r.slotTotalEmit)).toEqual([1, 0, 6]);
  });

  test("rejects out-of-range slot", () => {
    expect(() => partitionCPU({
      slot: [0, 5], emitCount: [1, 1], slotCount: 2,
    })).toThrow(/out-of-range/);
  });

  test("rejects mismatched array lengths", () => {
    expect(() => partitionCPU({
      slot: [0, 0], emitCount: [1], slotCount: 1,
    })).toThrow(/!=/);
  });

  test("large-ish case stays consistent: each slot's prefix ends at slotTotalEmit", () => {
    const slot:      number[] = [];
    const emitCount: number[] = [];
    const S = 4;
    const N = 100;
    for (let i = 0; i < N; i++) {
      slot.push(i % S);
      emitCount.push(((i * 13) % 7) + 1);
    }
    const r = partitionCPU({ slot, emitCount, slotCount: S });
    for (let s = 0; s < S; s++) {
      const c = r.slotCounts[s]!;
      const start = r.slotOffsets[s]! + s;
      expect(r.slotEmitPrefix[start]).toBe(0);
      expect(r.slotEmitPrefix[start + c]).toBe(r.slotTotalEmit[s]);
    }
    // sum of slotTotalEmit equals sum of emitCount
    let sumIn = 0; for (const e of emitCount) sumIn += e;
    let sumOut = 0; for (let s = 0; s < S; s++) sumOut += r.slotTotalEmit[s]!;
    expect(sumOut).toBe(sumIn);
  });
});
