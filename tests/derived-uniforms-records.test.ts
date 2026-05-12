import { describe, it, expect } from "vitest";
import {
  RecordsBuffer, SlotTag, makeHandle, handleTag, handlePayload,
} from "../packages/rendering/src/runtime/derivedUniforms/records.js";

const C = (slot: number) => makeHandle(SlotTag.Constituent, slot);
const H = (byte: number) => makeHandle(SlotTag.HostHeap, byte);

function recordAt(rb: RecordsBuffer, i: number): number[] {
  const s = rb.strideWords;
  return Array.from(rb.data.subarray(i * s, i * s + s));
}

describe("derived-uniforms: records buffer", () => {
  it("packs records at the current stride with zero-padded tail slots", () => {
    const rb = new RecordsBuffer();
    const o = {};
    const i0 = rb.add(o, 3, H(64), [C(0), C(1)]);
    expect(i0).toBe(0);
    expect(rb.recordCount).toBe(1);
    expect(recordAt(rb, 0)).toEqual([3, H(64), C(0), C(1), 0]); // MIN_STRIDE_U32 === 5
    rb.add(o, 7, H(128), [C(2)]);
    expect(recordAt(rb, 1)).toEqual([7, H(128), C(2), 0, 0]);
  });

  it("handle pack/unpack round-trips", () => {
    const h = makeHandle(SlotTag.Globals, 12345);
    expect(handleTag(h)).toBe(SlotTag.Globals);
    expect(handlePayload(h)).toBe(12345);
    expect(() => makeHandle(SlotTag.HostHeap, (1 << 29))).toThrow();
  });

  it("grows the stride when a higher-arity record is added, re-packing existing rows", () => {
    const rb = new RecordsBuffer();
    const o = {};
    rb.add(o, 1, H(0), [C(0)]);
    expect(rb.strideWords).toBe(5);
    const v0 = rb.layoutVersion;
    rb.add(o, 2, H(16), [C(0), C(1), C(2), C(3), C(4), C(5)]); // arity 6 ⇒ stride 8
    expect(rb.strideWords).toBe(8);
    expect(rb.layoutVersion).toBe(v0 + 1);
    expect(recordAt(rb, 0)).toEqual([1, H(0), C(0), 0, 0, 0, 0, 0]);
    expect(recordAt(rb, 1)).toEqual([2, H(16), C(0), C(1), C(2), C(3), C(4), C(5)]);
  });

  it("swap-removes all records of an owner, keeping others intact", () => {
    const rb = new RecordsBuffer();
    const a = {}, b = {}, c = {};
    rb.add(a, 10, H(0), [C(0)]);   // 0
    rb.add(b, 20, H(4), [C(1)]);   // 1
    rb.add(a, 11, H(8), [C(2)]);   // 2
    rb.add(c, 30, H(12), [C(3)]);  // 3
    rb.add(b, 21, H(16), [C(4)]);  // 4
    expect(rb.recordCount).toBe(5);

    rb.removeAllForOwner(a);
    expect(rb.recordCount).toBe(3);
    // b's and c's records must all still be present (rule ids 20, 21, 30 in some order).
    const ids = new Set<number>();
    for (let i = 0; i < rb.recordCount; i++) ids.add(recordAt(rb, i)[0]);
    expect(ids).toEqual(new Set([20, 21, 30]));

    rb.removeAllForOwner(b);
    expect(rb.recordCount).toBe(1);
    expect(recordAt(rb, 0)[0]).toBe(30);

    rb.removeAllForOwner(c);
    expect(rb.recordCount).toBe(0);
    expect(rb.data.length).toBe(0);
  });

  it("removing an owner that owns the tail block works", () => {
    const rb = new RecordsBuffer();
    const a = {}, b = {};
    rb.add(a, 1, H(0), [C(0)]);
    rb.add(b, 2, H(4), [C(1)]);
    rb.add(b, 3, H(8), [C(2)]);
    rb.add(b, 4, H(12), [C(3)]);
    rb.removeAllForOwner(b);
    expect(rb.recordCount).toBe(1);
    expect(recordAt(rb, 0)[0]).toBe(1);
    // a is still removable afterwards
    rb.removeAllForOwner(a);
    expect(rb.recordCount).toBe(0);
  });

  it("re-adding after removal reuses freed capacity", () => {
    const rb = new RecordsBuffer();
    const a = {};
    for (let i = 0; i < 100; i++) rb.add(a, i, H(i * 4), [C(i)]);
    rb.removeAllForOwner(a);
    expect(rb.recordCount).toBe(0);
    const b = {};
    rb.add(b, 999, H(0), [C(0)]);
    expect(rb.recordCount).toBe(1);
    expect(recordAt(rb, 0)).toEqual([999, H(0), C(0), 0, 0]);
  });
});
