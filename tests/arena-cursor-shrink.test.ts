// §5 — cursor-shrink hygiene on AttributeArena + IndexAllocator.
//
// When a release exposes a free block touching the bump cursor,
// the arena reclaims it back into the bump region instead of
// leaving a permanent high-watermark. Covers single-release,
// cascading-via-coalesce, and mid-region holes (which should NOT
// shrink).

import { describe, expect, it } from "vitest";
import { MockGPU } from "./_mockGpu.js";
import { GrowBuffer, DEFAULT_MAX_BUFFER_BYTES } from "../packages/rendering/src/runtime/heapScene/growBuffer.js";
import { AttributeArena, IndexAllocator } from "../packages/rendering/src/runtime/heapScene/pools.js";

function makeArena(gpu: MockGPU): AttributeArena {
  return new AttributeArena(new GrowBuffer(
    gpu.device, "test/attrs", GPUBufferUsage.STORAGE, 1024,
  ));
}
function makeIndices(gpu: MockGPU): IndexAllocator {
  return new IndexAllocator(new GrowBuffer(
    gpu.device, "test/idx", GPUBufferUsage.INDEX, 1024,
  ));
}

describe("AttributeArena cursor-shrink (§5)", () => {
  it("releasing the top-most block reclaims those bytes", () => {
    const gpu = new MockGPU();
    const arena = makeArena(gpu);
    arena.alloc(16);             // ref=0, allocBytes=32 (header+data → 16-aligned)
    const ref2 = arena.alloc(16);
    const before = arena.usedBytes;
    arena.release(ref2, 16);
    expect(arena.usedBytes).toBeLessThan(before);
  });

  it("cascading: coalesce-then-shrink reclaims the whole released tail", () => {
    const gpu = new MockGPU();
    const arena = makeArena(gpu);
    const r0 = arena.alloc(16);
    const r1 = arena.alloc(16);
    const r2 = arena.alloc(16);
    const r3 = arena.alloc(16);
    void r0;
    const high = arena.usedBytes;
    // Release out-of-order so coalesce runs:
    arena.release(r3, 16);   // tail block exposed
    arena.release(r2, 16);   // coalesces with r3 → bigger tail
    arena.release(r1, 16);   // coalesces all the way back to r1
    // After the cascade only r0's 32 B should remain in the bump
    // region; usedBytes is 32 (one mat-aligned slot for the
    // 16-byte payload).
    expect(arena.usedBytes).toBeLessThan(high);
    expect(arena.usedBytes).toBeGreaterThan(0);
  });

  it("releasing a middle-of-arena block does NOT shrink the cursor", () => {
    const gpu = new MockGPU();
    const arena = makeArena(gpu);
    const r0 = arena.alloc(16);
    const rMid = arena.alloc(16);
    arena.alloc(16);   // r2 keeps the tail occupied
    void r0;
    const before = arena.usedBytes;
    arena.release(rMid, 16);
    // r2 still holds the tail — cursor doesn't shrink.
    expect(arena.usedBytes).toBe(before);
  });
});

describe("IndexAllocator cursor-shrink (§5)", () => {
  it("releasing the top index range reclaims those elements", () => {
    const gpu = new MockGPU();
    const ix = makeIndices(gpu);
    ix.alloc(100);
    ix.alloc(50);   // top range
    const before = ix.usedElements;
    ix.release(100, 50);
    expect(ix.usedElements).toBe(100);
    expect(ix.usedElements).toBeLessThan(before);
  });
});

describe("GrowBuffer hardware-aware cap (§3)", () => {
  it("respects an explicit maxBytes cap", () => {
    const gpu = new MockGPU();
    const CAP = 128 * 1024;  // 128 KB cap (above the MIN_BUFFER_BYTES floor).
    const buf = new GrowBuffer(
      gpu.device, "test/cap", GPUBufferUsage.STORAGE, 1024, CAP,
    );
    expect(buf.maxCapacity).toBe(CAP);
    buf.ensureCapacity(CAP);
    expect(buf.capacity).toBe(CAP);
    expect(() => buf.ensureCapacity(CAP * 2)).toThrow(/maxBytes/);
  });

  it("default cap covers a reasonably large workload", () => {
    const gpu = new MockGPU();
    const buf = new GrowBuffer(gpu.device, "test/default", GPUBufferUsage.STORAGE, 1024);
    expect(buf.maxCapacity).toBe(DEFAULT_MAX_BUFFER_BYTES);
  });
});
