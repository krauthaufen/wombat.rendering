// Ref re-seating for the two arena-ref holders the real-GPU compaction
// test can't reach without pulling wombat.base into the browser bundle:
//   • §7 derived-uniform records (RecordsBuffer.remapHostHeap)
//   • derived-modes partition master refs (GpuPartitionScene.remapUniformRefs)
// Both are pure CPU bookkeeping over typed-array shadows, so we verify
// them directly. (The universal pool / drawHeader / Pos-Nor remap + the
// actual GPU byte-move are covered end-to-end by
// tests-browser/heap-compaction-real.test.ts.)

import { describe, expect, it } from "vitest";
import { MockGPU } from "./_mockGpu.js";
import {
  RecordsBuffer, makeHandle, handleTag, handlePayload,
} from "../packages/rendering/src/runtime/derivedUniforms/records.js";
import { GpuPartitionScene } from "../packages/rendering/src/runtime/derivedModes/partitionDispatcher.js";

const TAG_CONSTITUENT = 0;
const TAG_HOSTHEAP = 1;

describe("RecordsBuffer.remapHostHeap", () => {
  it("rewrites HostHeap handles, leaves Constituent handles and generation", () => {
    const rec = new RecordsBuffer();
    // One record: out = HostHeap@200, in0 = Constituent@5, in1 = HostHeap@40.
    const owner = {};
    rec.add(owner, /*ruleId*/ 1, makeHandle(TAG_HOSTHEAP, 200), [
      makeHandle(TAG_CONSTITUENT, 5),
      makeHandle(TAG_HOSTHEAP, 40),
    ]);
    // A second record whose HostHeap offset is NOT in the remap.
    rec.add({}, 2, makeHandle(TAG_HOSTHEAP, 999), [makeHandle(TAG_CONSTITUENT, 7)]);
    const genBefore = rec.generation;

    // Relocate 200→8 and 40→304; leave 999 untouched.
    const changed = rec.remapHostHeap(new Map([[200, 8], [40, 304]]));
    expect(changed).toBe(2);
    expect(rec.generation).toBe(genBefore + 1);

    const data = rec.data;
    const s = rec.strideWords;
    // record 0: out_slot (HostHeap) → 8, in1 (HostHeap) → 304, in0 (Constituent) unchanged.
    expect(handleTag(data[1]!)).toBe(TAG_HOSTHEAP);
    expect(handlePayload(data[1]!)).toBe(8);
    expect(handleTag(data[2]!)).toBe(TAG_CONSTITUENT);
    expect(handlePayload(data[2]!)).toBe(5);
    expect(handleTag(data[3]!)).toBe(TAG_HOSTHEAP);
    expect(handlePayload(data[3]!)).toBe(304);
    // record 1: out_slot HostHeap@999 not in remap → unchanged.
    expect(handlePayload(data[s + 1]!)).toBe(999);
  });

  it("no-op (no generation bump) when nothing matches", () => {
    const rec = new RecordsBuffer();
    rec.add({}, 1, makeHandle(TAG_HOSTHEAP, 16), [makeHandle(TAG_CONSTITUENT, 0)]);
    const gen = rec.generation;
    expect(rec.remapHostHeap(new Map([[64, 96]]))).toBe(0);
    expect(rec.generation).toBe(gen);
  });
});

describe("GpuPartitionScene.remapUniformRefs", () => {
  function makeScene(gpu: MockGPU, numUniforms: number): GpuPartitionScene {
    return new GpuPartitionScene(gpu.device, "test/partition", {
      totalSlots: 1,
      slotDrawBufs: [gpu.device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE })],
      kernelWGSL: "",
      initialRecords: 16,
      numUniforms,
    });
  }

  it("re-seats baked uniform refs in the master, leaving prefix fields", () => {
    const gpu = new MockGPU();
    const scene = makeScene(gpu, 2);
    // Two records, each with two baked arena refs.
    scene.appendRecord(/*drawIdx*/ 3, /*indexStart*/ 100, /*indexCount*/ 6, /*instanceCount*/ 1, /*comboId*/ 0, [128, 256]);
    scene.appendRecord(7, 200, 9, 2, 1, [256, 512]);

    const ru = scene.recordU32; // 6 + numUniforms
    const PREFIX = 6;
    // Relocate 256→16 and 512→48; 128 stays.
    const changed = scene.remapUniformRefs(new Map([[256, 16], [512, 48]]));
    expect(changed).toBe(3); // rec0[1]=256, rec1[0]=256, rec1[1]=512

    const m = scene.masterShadow;
    // record 0 uniform refs: [128 unchanged, 256→16].
    expect(m[0 * ru + PREFIX + 0]).toBe(128);
    expect(m[0 * ru + PREFIX + 1]).toBe(16);
    // record 1 uniform refs: [256→16, 512→48].
    expect(m[1 * ru + PREFIX + 0]).toBe(16);
    expect(m[1 * ru + PREFIX + 1]).toBe(48);
    // Prefix fields (drawIdx / indexStart / indexCount / instanceCount) intact.
    expect(m[1 * ru + 1]).toBe(7);   // drawIdx
    expect(m[1 * ru + 2]).toBe(200); // indexStart
    expect(m[1 * ru + 3]).toBe(9);   // indexCount
    expect(m[1 * ru + 4]).toBe(2);   // instanceCount
  });

  it("no-op when numUniforms is 0", () => {
    const gpu = new MockGPU();
    const scene = makeScene(gpu, 0);
    scene.appendRecord(0, 0, 3, 1, 0, []);
    expect(scene.remapUniformRefs(new Map([[0, 16]]))).toBe(0);
  });
});
