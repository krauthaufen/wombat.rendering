// Mock-GPU coverage of the Phase 5c.3 partition kernel + dispatcher.
// Real-GPU pixel-level validation lives in tests-browser/ when that
// infra is back online.

import { describe, expect, it } from "vitest";
import {
  GpuPartitionScene,
  PARTITION_FLIP_CULL_BY_DET_WGSL,
  PARTITION_RECORD_BYTES,
  cullModeToU32,
} from "@aardworx/wombat.rendering/runtime";
import { MockGPU } from "./_mockGpu.js";

if (typeof (globalThis as { GPUTextureUsage?: unknown }).GPUTextureUsage === "undefined") {
  (globalThis as Record<string, unknown>).GPUTextureUsage = {
    COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04,
    STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10,
  };
}
if (typeof (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage === "undefined") {
  (globalThis as Record<string, unknown>).GPUBufferUsage = {
    MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
    INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
    INDIRECT: 0x0100, QUERY_RESOLVE: 0x0200,
  };
}
if (typeof (globalThis as { GPUShaderStage?: unknown }).GPUShaderStage === "undefined") {
  (globalThis as Record<string, unknown>).GPUShaderStage = {
    VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4,
  };
}

describe("derivedModes/partition", () => {
  it("record bytes are 6 × u32 = 24", () => {
    expect(PARTITION_RECORD_BYTES).toBe(24);
  });

  it("cullModeToU32 mirrors the existing CULL_TO_U32 ordering", () => {
    expect(cullModeToU32("none")).toBe(0);
    expect(cullModeToU32("front")).toBe(1);
    expect(cullModeToU32("back")).toBe(2);
  });

  it("appendRecord lays records into the master shadow at the right stride", () => {
    const gpu = new MockGPU();
    const slot0 = gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    const slot1 = gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    const scene = new GpuPartitionScene(gpu.device, "test/partition", slot0, slot1, 8);
    scene.appendRecord(/* drawIdx */ 7, /* indexStart */ 100, /* indexCount */ 36, /* instanceCount */ 1, /* modelRef */ 4096);
    scene.appendRecord(/* drawIdx */ 8, /* indexStart */ 136, /* indexCount */ 36, /* instanceCount */ 1, /* modelRef */ 8192);
    expect(scene.numRecords).toBe(2);
    // Verify the master shadow encoding (6 u32 per record):
    //   [firstEmit=0, drawIdx, indexStart, indexCount, instanceCount, modelRef]
    expect(Array.from(scene.masterShadow.slice(0, 12))).toEqual([
      0, 7, 100, 36, 1, 4096,
      0, 8, 136, 36, 1, 8192,
    ]);
  });

  it("removeRecord swap-pops the last entry", () => {
    const gpu = new MockGPU();
    const slot0 = gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    const slot1 = gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    const scene = new GpuPartitionScene(gpu.device, "test/partition", slot0, slot1, 8);
    scene.appendRecord(1, 0, 3, 1, 100);
    scene.appendRecord(2, 3, 3, 1, 200);
    scene.appendRecord(3, 6, 3, 1, 300);
    const moved = scene.removeRecord(0);
    expect(moved).toBe(2); // record 2 moved into slot 0
    expect(scene.numRecords).toBe(2);
    expect(Array.from(scene.masterShadow.slice(0, 6))).toEqual([0, 3, 6, 3, 1, 300]);
  });

  it("dispatch creates the clear + partition pipelines on first call", () => {
    const gpu = new MockGPU();
    const slot0 = gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    const slot1 = gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    const scene = new GpuPartitionScene(gpu.device, "test/partition", slot0, slot1, 8);
    const arenaBuf = gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    expect(gpu.computePipelines.length).toBe(0);
    scene.appendRecord(0, 0, 3, 1, 0);
    scene.flush();
    const enc = gpu.device.createCommandEncoder();
    scene.dispatch(arenaBuf, "back", enc);
    enc.finish();
    expect(gpu.computePipelines.length).toBe(2);
    const entries = gpu.computePipelines.map(p => p.compute.entryPoint).sort();
    expect(entries).toEqual(["clear", "partition"]);
  });

  it("kernel module carries the partition WGSL source", () => {
    const gpu = new MockGPU();
    const slot0 = gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    const slot1 = gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    const scene = new GpuPartitionScene(gpu.device, "test/partition", slot0, slot1, 8);
    const arenaBuf = gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    scene.appendRecord(0, 0, 3, 1, 0);
    scene.flush();
    const enc = gpu.device.createCommandEncoder();
    scene.dispatch(arenaBuf, "back", enc);
    enc.finish();
    expect(gpu.shaderModules.some(m => m.code === PARTITION_FLIP_CULL_BY_DET_WGSL)).toBe(true);
  });

  it("dispatch with zero records still encodes the clear pass", () => {
    const gpu = new MockGPU();
    const slot0 = gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    const slot1 = gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    const scene = new GpuPartitionScene(gpu.device, "test/partition", slot0, slot1, 8);
    const arenaBuf = gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    const enc = gpu.device.createCommandEncoder();
    scene.dispatch(arenaBuf, "back", enc);
    enc.finish();
    // Clear still set up the pipelines.
    expect(gpu.computePipelines.length).toBe(2);
  });
});
