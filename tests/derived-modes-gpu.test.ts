// Mock-GPU coverage of the GPU-eval kernel + dispatcher. Real-GPU
// pixel-level validation lives in tests-browser/ when the browser
// fixture infra is back online.

import { describe, expect, it } from "vitest";
import {
  GpuDerivedModesScene,
  gpuFlipCullByDeterminant,
  isDerivedModeRule,
  GPU_FLIP_CULL_BY_DET_WGSL,
  CULL_TO_U32,
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
if (typeof (globalThis as { GPUMapMode?: unknown }).GPUMapMode === "undefined") {
  (globalThis as Record<string, unknown>).GPUMapMode = {
    READ: 0x1, WRITE: 0x2,
  };
}

describe("derivedModes/gpuFlipCullByDeterminant", () => {
  it("returns a DerivedModeRule with both CPU fallback and gpu marker", () => {
    const rule = gpuFlipCullByDeterminant("ModelTrafo", "back");
    expect(isDerivedModeRule(rule)).toBe(true);
    expect(rule.axis).toBe("cull");
    expect(rule.gpu).toBeDefined();
    expect(rule.gpu!.kernel).toBe("flipCullByDeterminant");
    expect(rule.gpu!.inputUniform).toBe("ModelTrafo");
    expect(rule.domain).toEqual(["back", "front", "none"]);
  });

  it("CPU fallback correctly flips by determinant sign", () => {
    const rule = gpuFlipCullByDeterminant("ModelTrafo", "back");
    // Row-major identity → det = 1 → no flip.
    const identity = { forward: { _data: new Float64Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]) } };
    expect(rule.evaluate({ ModelTrafo: identity }, "back")).toBe("back");
    // -1 scale on x → det = -1 → flip.
    const mirroredX = { forward: { _data: new Float64Array([
      -1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1,
    ]) } };
    expect(rule.evaluate({ ModelTrafo: mirroredX }, "back")).toBe("front");
    expect(rule.evaluate({ ModelTrafo: mirroredX }, "front")).toBe("back");
    // 'none' is preserved.
    expect(rule.evaluate({ ModelTrafo: mirroredX }, "none")).toBe("none");
  });
});

describe("derivedModes/GpuDerivedModesScene", () => {
  it("creates compute pipeline lazily on first dispatch", () => {
    const gpu = new MockGPU();
    const scene = new GpuDerivedModesScene(gpu.device);
    const arenaBuf = gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    expect(gpu.computePipelines.length).toBe(0);
    scene.registerRo(0, 16, "back");
    const enc = gpu.device.createCommandEncoder();
    scene.dispatch(arenaBuf, 1, enc);
    enc.finish();
    expect(gpu.computePipelines.length).toBe(1);
    expect(gpu.computePipelines[0]!.compute.entryPoint).toBe("evaluate");
  });

  it("emits the correct WGSL source", () => {
    const gpu = new MockGPU();
    const scene = new GpuDerivedModesScene(gpu.device);
    const arenaBuf = gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    scene.registerRo(0, 16, "back");
    const enc = gpu.device.createCommandEncoder();
    scene.dispatch(arenaBuf, 1, enc);
    enc.finish();
    // The shader module created should carry the kernel source.
    const modules = gpu.shaderModules;
    expect(modules.some(m => m.code === GPU_FLIP_CULL_BY_DET_WGSL)).toBe(true);
  });

  it("registerRo/deregisterRo updates liveCount", () => {
    const gpu = new MockGPU();
    const scene = new GpuDerivedModesScene(gpu.device);
    expect(scene.registered).toBe(0);
    scene.registerRo(0, 16, "back");
    scene.registerRo(1, 80, "back");
    expect(scene.registered).toBe(2);
    scene.deregisterRo(0);
    expect(scene.registered).toBe(1);
  });

  it("CULL_TO_U32 mirrors bitfield.ts enum order", () => {
    expect(CULL_TO_U32.none).toBe(0);
    expect(CULL_TO_U32.front).toBe(1);
    expect(CULL_TO_U32.back).toBe(2);
  });
});
