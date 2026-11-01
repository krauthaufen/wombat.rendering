// A tightly packed host buffer uploads its own bytes: the arena write sees
// the caller's ArrayBuffer, with no copy in JS.
import { describe, expect, it } from "vitest";
import { writeAllocation, type PoolPack } from "../packages/rendering/src/runtime/heapScene/pools.js";
import { packHostU32Bits } from "../packages/rendering/src/runtime/heapScene.js";

function recorder() {
  const writes: { off: number; data: Uint8Array }[] = [];
  return { writes, arena: { write: (_c: number, off: number, data: Uint8Array) => { writes.push({ off, data }); } } };
}

describe("writeAllocation", () => {
  const source = new Float32Array([1, 2, 3, 4, 5, 6]);
  const copying: PoolPack = (v, dst, off) => dst.set(v as Float32Array, off);

  it("hands over the caller's bytes when the pack has them", () => {
    const direct: PoolPack = (v, dst, off) => dst.set(v as Float32Array, off);
    direct.bytes = (v) => new Uint8Array((v as Float32Array).buffer);
    const { writes, arena } = recorder();
    writeAllocation(arena as never, 0, 32, source, 24, 7, 2, direct);
    expect(writes.length).toBe(2);
    expect([...new Uint32Array(writes[0]!.data.buffer)]).toEqual([7, 2, 12, 0]);
    expect(writes[0]!.off).toBe(32);
    expect(writes[1]!.off).toBe(48);
    expect(writes[1]!.data.buffer).toBe(source.buffer);
  });

  it("copies when the bytes aren't the whole payload", () => {
    const short: PoolPack = (v, dst, off) => dst.set((v as Float32Array).subarray(0, 3), off);
    short.bytes = (v) => new Uint8Array((v as Float32Array).buffer);
    const { writes, arena } = recorder();
    writeAllocation(arena as never, 0, 0, source, 12, 7, 1, short);
    expect(writes.length).toBe(1);
    expect(writes[0]!.data.buffer).not.toBe(source.buffer);
    expect([...new Float32Array(writes[0]!.data.buffer, 16, 3)]).toEqual([1, 2, 3]);
  });

  it("writes header and data together without bytes", () => {
    const { writes, arena } = recorder();
    writeAllocation(arena as never, 0, 0, source, 24, 7, 2, copying);
    expect(writes.length).toBe(1);
    expect([...new Float32Array(writes[0]!.data.buffer, 16, 6)]).toEqual([1, 2, 3, 4, 5, 6]);
  });
});

describe("packHostU32Bits", () => {
  const colors = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
  const written = (data: unknown) => {
    const dst = new Float32Array(2);
    packHostU32Bits({ kind: "host", data }, dst, 0);
    return [...new Uint8Array(dst.buffer)];
  };

  it("reads a typed array", () => {
    expect(written(colors)).toEqual([...colors]);
  });

  it("reads a raw ArrayBuffer", () => {
    expect(written(colors.buffer)).toEqual([...colors]);
  });
});
