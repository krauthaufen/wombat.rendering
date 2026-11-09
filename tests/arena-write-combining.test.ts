// AttributeArena write-combining: adjacent writes go up as one
// writeBuffer; what a GPU-side operation reads is uploaded first; the
// buffer's contents are as with one writeBuffer per write.

import { describe, expect, it } from "vitest";
import { MockGPU } from "./_mockGpu.js";
import { AttributeArena } from "../packages/rendering/src/runtime/heapScene/pools.js";
import { GrowBuffer } from "../packages/rendering/src/runtime/heapScene/growBuffer.js";

/** A mock whose buffers hold bytes: writes copy at call time (as WebGPU
 *  does), copies run in recording order; every upload and copy is logged. */
function bytesGpu() {
  const gpu = new MockGPU();
  const bytes = new Map<unknown, Uint8Array>();
  const log: string[] = [];
  const of = (b: unknown, size: number) => {
    let a = bytes.get(b);
    if (a === undefined || a.length < size) {
      const n = new Uint8Array(Math.max(size, 1 << 16));
      if (a !== undefined) n.set(a);
      a = n; bytes.set(b, a);
    }
    return a;
  };
  const q = gpu.device.queue as unknown as { writeBuffer: Function };
  const write = q.writeBuffer.bind(q);
  q.writeBuffer = (b: unknown, off: number, data: ArrayBufferView, dataOff = 0, size?: number) => {
    const src = new Uint8Array(data.buffer, data.byteOffset + dataOff, size ?? data.byteLength - dataOff);
    of(b, off + src.length).set(src, off);
    log.push(`write ${off} ${src.length}`);
    write(b, off, data, dataOff, size);
  };
  const enc = gpu.createCommandEncoder.bind(gpu);
  gpu.createCommandEncoder = () => {
    const e = enc() as unknown as { copyBufferToBuffer: Function };
    const copy = e.copyBufferToBuffer.bind(e);
    e.copyBufferToBuffer = (s: unknown, so: number, d: unknown, dof: number, n: number) => {
      of(d, dof + n).set(of(s, so + n).subarray(so, so + n), dof);
      log.push(`copy ${n}`);
      copy(s, so, d, dof, n);
    };
    return e as unknown as GPUCommandEncoder;
  };
  return { gpu, bytes: (b: GPUBuffer, n: number) => of(b, n).slice(0, n), log };
}

const fill = (n: number, v: number) => new Uint8Array(n).fill(v);

describe("AttributeArena write-combining", () => {
  it("adjacent writes go up as one, staged until something reads the buffer", () => {
    const { gpu, bytes, log } = bytesGpu();
    const arena = new AttributeArena(gpu.device, new GrowBuffer(gpu.device, "a", GPUBufferUsage.STORAGE, 64 * 1024));
    const want = new Uint8Array(100 * 48);
    for (let i = 0; i < 100; i++) {
      const ref = arena.alloc(32); // 16 B header + 32 B data: 48 B, adjacent
      expect(ref).toBe(i * 48);
      arena.write(ref, fill(16, i));
      arena.write(ref + 16, fill(32, 255 - i));
      want.set(fill(16, i), ref); want.set(fill(32, 255 - i), ref + 16);
    }
    expect(log.length).toBe(0);
    const buf = arena.buffer; // reading the buffer uploads what is staged
    expect(log).toEqual([`write 0 ${100 * 48}`]);
    expect(bytes(buf, want.length)).toEqual(want);
  });

  it("writes that are not adjacent keep their order", () => {
    const { gpu, bytes, log } = bytesGpu();
    const arena = new AttributeArena(gpu.device, new GrowBuffer(gpu.device, "a", GPUBufferUsage.STORAGE, 64 * 1024));
    arena.write(0, fill(16, 1));
    arena.write(64, fill(16, 2));
    arena.write(0, fill(16, 3)); // overwrites the first, after it
    arena.write(16, fill(16, 4)); // adjacent to the last
    arena.flush(gpu.device);
    expect(log).toEqual(["write 0 16", "write 64 16", "write 0 32"]);
    const b = bytes(arena.buffer, 80);
    expect([...b.subarray(0, 16)]).toEqual([...fill(16, 3)]);
    expect([...b.subarray(16, 32)]).toEqual([...fill(16, 4)]);
    expect([...b.subarray(64, 80)]).toEqual([...fill(16, 2)]);
  });

  it("a grow uploads what is staged before copying the buffer", () => {
    const { gpu, bytes, log } = bytesGpu();
    const arena = new AttributeArena(gpu.device, new GrowBuffer(gpu.device, "a", GPUBufferUsage.STORAGE, 64 * 1024));
    const r0 = arena.alloc(1024 - 16);
    arena.write(r0, fill(1024, 7));
    // past the first 64 KB: the buffer grows
    for (let i = 0; i < 70; i++) arena.alloc(1024 - 16);
    expect(log[0]).toBe("write 0 1024");
    expect(log[1]).toMatch(/^copy /);
    expect([...bytes(arena.buffer, 1024)]).toEqual([...fill(1024, 7)]);
  });

  it("large data starting a run goes up at once; a run stays under the stage's size", () => {
    const { gpu, log } = bytesGpu();
    const arena = new AttributeArena(gpu.device, new GrowBuffer(gpu.device, "a", GPUBufferUsage.STORAGE, 4 << 20));
    arena.write(0, fill(16, 1));
    arena.write(4096, fill(512 * 1024, 2));
    expect(log).toEqual(["write 0 16", `write 4096 ${512 * 1024}`]);
    // adjacent writes fill the 1 MB stage, then a new run starts
    for (let i = 0; i < 5; i++) arena.write((1 << 20) + i * 256 * 1024, fill(256 * 1024, 3));
    arena.flush(gpu.device);
    expect(log.slice(2)).toEqual([`write ${1 << 20} ${1 << 20}`, `write ${(1 << 20) + (1 << 20)} ${256 * 1024}`]);
  });
});
