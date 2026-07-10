// Waste-triggered AttributeArena compaction.
//
// Exact-size freelist reuse can't reclaim a drifting allocation-size
// distribution: mid-arena holes never touch the bump cursor, so the
// buffer ratchets toward its high-water forever. `compact()` relocates
// the live allocations to the front, returns the OLD→NEW remap, and
// the GPU bounce does the byte move (mirror-less). These tests cover
// the bookkeeping and the decline thresholds.
//
// (The GPU-side copyBufferToBuffer is a no-op under MockGPU; full
// data-integrity through the render pipeline — drawHeaders, §7 handles,
// modes master refs — is covered by the real-GPU browser suite.)

import { describe, expect, it } from "vitest";
import { MockGPU } from "./_mockGpu.js";
import { GrowBuffer } from "../packages/rendering/src/runtime/heapScene/growBuffer.js";
import { AttributeArena } from "../packages/rendering/src/runtime/heapScene/pools.js";

function makeArena(gpu: MockGPU): AttributeArena {
  return new AttributeArena(gpu.device, new GrowBuffer(
    gpu.device, "test/attrs", GPUBufferUsage.STORAGE, 4096,
  ));
}
/** Fill the 32-byte (header+16 data → 16-aligned) block at `ref` with a
 *  recognizable sentinel so we can prove its bytes followed it. */
function stamp(arena: AttributeArena, ref: number, tag: number): void {
  const u32 = new Uint32Array(8).fill(tag);
  arena.write(ref, new Uint8Array(u32.buffer));
}

describe("AttributeArena.compact (waste-triggered)", () => {
  it("relocates live blocks to the front and returns the remap", () => {
    const gpu = new MockGPU();
    const arena = makeArena(gpu);
    // Six 32-byte blocks at 0,32,64,96,128,160.
    const refs = [0, 1, 2, 3, 4, 5].map(() => arena.alloc(16));
    refs.forEach((r, i) => stamp(arena, r, 0xa0 + i));
    expect(arena.usedBytes).toBe(192);
    expect(arena.wasteBytes).toBe(0);

    // Free the four middle blocks → live={0,160}, waste(128) ≥ live(64),
    // tail still pinned at 160 so the cursor can't shrink on its own.
    for (const i of [1, 2, 3, 4]) arena.release(refs[i]!, 16);
    expect(arena.usedBytes).toBe(192);
    expect(arena.wasteBytes).toBe(128);
    expect(arena.liveByteCount).toBe(64);

    const remap = arena.compact(gpu.device, 0);
    // Block 0 stays put; block 160 moves to 32.
    expect(remap.get(0)).toBeUndefined();
    expect(remap.get(160)).toBe(32);
    // Cursor packed down to exactly the live bytes; zero waste.
    expect(arena.usedBytes).toBe(64);
    expect(arena.wasteBytes).toBe(0);
    expect(arena.liveByteCount).toBe(64);

    // (Mirror-less arena: byte relocation happens GPU-side via the scratch
    // bounce — data integrity is proven by the real-GPU compaction golden,
    // which pixel-diffs a fragmented+compacted scene against a reference.)
  });

  it("a fresh alloc after compaction reuses the reclaimed tail (no ratchet)", () => {
    const gpu = new MockGPU();
    const arena = makeArena(gpu);
    const refs = [0, 1, 2, 3, 4, 5].map(() => arena.alloc(16));
    for (const i of [1, 2, 3, 4]) arena.release(refs[i]!, 16);
    arena.compact(gpu.device, 0);
    expect(arena.usedBytes).toBe(64);
    // Next allocation lands at the compacted cursor, not the old high-water.
    const r = arena.alloc(16);
    expect(r).toBe(64);
    expect(arena.usedBytes).toBe(96);
  });

  it("declines when waste is below the floor", () => {
    const gpu = new MockGPU();
    const arena = makeArena(gpu);
    const refs = [0, 1, 2, 3, 4, 5].map(() => arena.alloc(16));
    for (const i of [1, 2, 3, 4]) arena.release(refs[i]!, 16);
    expect(arena.wasteBytes).toBe(128);
    const remap = arena.compact(gpu.device, 4 * 1024 * 1024); // 4 MiB floor
    expect(remap.size).toBe(0);
    expect(arena.usedBytes).toBe(192); // untouched
  });

  it("declines when less than half the extent is wasted", () => {
    const gpu = new MockGPU();
    const arena = makeArena(gpu);
    const refs = [0, 1, 2, 3, 4, 5].map(() => arena.alloc(16));
    // Free only two of six → waste(64) < live(128); the 50%-live gate
    // should keep us from thrashing on light fragmentation.
    for (const i of [1, 3] as const) arena.release(refs[i]!, 16);
    expect(arena.wasteBytes).toBe(64);
    const remap = arena.compact(gpu.device, 0); // floor 0, but gate still applies
    expect(remap.size).toBe(0);
    expect(arena.usedBytes).toBe(192);
  });
});
