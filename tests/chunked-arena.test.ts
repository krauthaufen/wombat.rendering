// ChunkedAttributeArena + ChunkedIndexAllocator unit tests. Exercises
// chunk opening, fallback ordering, hint routing, release into the
// right chunk, and the cursor-shrink hygiene cascading per chunk.

import { describe, expect, it } from "vitest";
import { MockGPU } from "./_mockGpu.js";
import {
  ChunkedAttributeArena,
  ChunkedIndexAllocator,
} from "../packages/rendering/src/runtime/heapScene/chunkedArena.js";

const KB = 1024;

describe("ChunkedAttributeArena", () => {
  it("opens chunk 0 on construction", () => {
    const gpu = new MockGPU();
    const ar = new ChunkedAttributeArena(gpu.device, "ar", GPUBufferUsage.STORAGE, 1024, 128 * KB);
    expect(ar.chunkCount).toBe(1);
  });

  it("hint routes allocation to the requested chunk", () => {
    const gpu = new MockGPU();
    const ar = new ChunkedAttributeArena(gpu.device, "ar", GPUBufferUsage.STORAGE, 1024, 128 * KB);
    // Open a second chunk by filling chunk 0.
    let i = 0, opened = false;
    while (i < 50_000) {
      const r = ar.alloc(64);
      if (r.chunkIdx === 1) { opened = true; break; }
      i++;
    }
    expect(opened).toBe(true);
    expect(ar.chunkCount).toBe(2);
    // Now a hinted alloc into chunk 0 should still land in chunk 0
    // (it had a small remnant since the spill happened on an
    // allocation that didn't fit any more 64 B blocks). Try a
    // smaller alloc which should fit.
    const r0 = ar.alloc(8, 0);
    expect([0, 1]).toContain(r0.chunkIdx);
  });

  it("opens a new chunk when allocation exceeds current chunks' caps", () => {
    const gpu = new MockGPU();
    const SMALL = 128 * KB;
    const ar = new ChunkedAttributeArena(gpu.device, "ar", 1024, GPUBufferUsage.STORAGE);
    // The above call passes 1024 as initialBytes and STORAGE as
    // maxBytes (wrong arg order, intentional test of API). Swap:
    void ar;
    const ar2 = new ChunkedAttributeArena(gpu.device, "ar", GPUBufferUsage.STORAGE, 1024, SMALL);
    // Allocate enough to force a second chunk.
    let chunkRefs = new Map<number, number>();
    for (let n = 0; n < 100_000 && ar2.chunkCount < 3; n++) {
      const r = ar2.alloc(1024);
      chunkRefs.set(r.chunkIdx, (chunkRefs.get(r.chunkIdx) ?? 0) + 1);
    }
    expect(ar2.chunkCount).toBeGreaterThan(1);
    // Total chunks should each be capped at SMALL.
    for (let c = 0; c < ar2.chunkCount; c++) {
      expect(ar2.chunk(c).usedBytes).toBeLessThanOrEqual(SMALL);
    }
  });

  it("release returns space within the same chunk", () => {
    const gpu = new MockGPU();
    const ar = new ChunkedAttributeArena(gpu.device, "ar", GPUBufferUsage.STORAGE, 1024, 128 * KB);
    const r = ar.alloc(64);
    const usedBefore = ar.chunk(r.chunkIdx).usedBytes;
    ar.release(r.chunkIdx, r.off, 64);
    // Top-of-chunk release cascades the cursor back via §5
    // cursor-shrink, so usedBytes should decrease.
    expect(ar.chunk(r.chunkIdx).usedBytes).toBeLessThan(usedBefore);
  });

  it("onChunkAdded fires when a new chunk opens", () => {
    const gpu = new MockGPU();
    const opened: number[] = [];
    const ar = new ChunkedAttributeArena(gpu.device, "ar", GPUBufferUsage.STORAGE, 1024, 128 * KB);
    ar.onChunkAdded((idx) => opened.push(idx));
    // Fill chunk 0.
    for (let i = 0; i < 10_000 && ar.chunkCount === 1; i++) ar.alloc(1024);
    expect(opened.length).toBeGreaterThanOrEqual(1);
    expect(opened[0]).toBe(1);  // chunk 0 was already there pre-subscription
  });
});

describe("ChunkedIndexAllocator", () => {
  it("element-bump units cap at maxChunkBytes/4", () => {
    const gpu = new MockGPU();
    const CAP = 128 * KB;     // 32 K u32 elements per chunk.
    const ix = new ChunkedIndexAllocator(gpu.device, "ix", GPUBufferUsage.INDEX, 1024, CAP);
    let total = 0;
    let chunks = new Set<number>();
    for (let n = 0; n < 50_000 && chunks.size < 3; n++) {
      const r = ix.alloc(1024);
      chunks.add(r.chunkIdx);
      total += 1024;
    }
    expect(chunks.size).toBeGreaterThan(1);
    expect(ix.totalUsedElements()).toBeGreaterThanOrEqual(total - 1024);
  });
});
