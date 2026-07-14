// The device pipeline cache must key on the emitted SOURCE, not just the effect
// id. wombat.shader's `Effect.id` hashes the effect template — closure-hole
// values are specialised into the source as literals and do NOT move the id
// when a getter starts returning something else. Keying on the id alone handed
// back a pipeline built from the previous hole value: a stale shader that keeps
// running silently (this is how an `u32(0xffffffff)` constant survived a
// changed capture).

import { describe, test, expect } from "vitest";
import { compileRenderPipeline } from "../packages/rendering/src/resources/renderPipeline.js";
import type { CompileRenderPipelineDescription } from "../packages/rendering/src/resources/renderPipeline.js";

function mockDevice(): GPUDevice & { pipelines: number; modules: number } {
  let pipelines = 0;
  let modules = 0;
  const dev = {
    get pipelines() { return pipelines; },
    get modules() { return modules; },
    createShaderModule() { modules++; return { getCompilationInfo: async () => ({ messages: [] }) } as unknown as GPUShaderModule; },
    createPipelineLayout() { return {} as GPUPipelineLayout; },
    createRenderPipeline() { pipelines++; return { label: `p${pipelines}` } as unknown as GPURenderPipeline; },
  };
  return dev as unknown as GPUDevice & { pipelines: number; modules: number };
}

const descFor = (fs: string): CompileRenderPipelineDescription => ({
  effectId: "effect-42", // SAME id — only the baked constant differs
  vertexShaderSource: "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0); }",
  fragmentShaderSource: fs,
  vertexEntryPoint: "vs",
  fragmentEntryPoint: "fs",
  vertexBufferLayouts: [],
  bindGroupLayouts: [],
  colorTargets: [{ format: "rgba8unorm" }],
  primitive: { topology: "triangle-list" },
});

const fsWith = (constant: string): string =>
  `@fragment fn fs() -> @location(0) vec4f { return vec4f(${constant}); }`;

describe("pipeline cache keys on the emitted source, not just effect.id", () => {
  test("same id + same source ⇒ cached (one pipeline)", () => {
    const device = mockDevice();
    const a = compileRenderPipeline(device, descFor(fsWith("1.0")));
    const b = compileRenderPipeline(device, descFor(fsWith("1.0")));
    expect(a).toBe(b);
    expect(device.pipelines).toBe(1);
  });

  test("same id + DIFFERENT source (a changed closure hole) ⇒ fresh pipeline", () => {
    const device = mockDevice();
    const before = compileRenderPipeline(device, descFor(fsWith("1.0")));
    const after = compileRenderPipeline(device, descFor(fsWith("0.5")));
    expect(after).not.toBe(before);
    expect(device.pipelines).toBe(2);
    // and going back to the original source re-hits the original entry
    expect(compileRenderPipeline(device, descFor(fsWith("1.0")))).toBe(before);
    expect(device.pipelines).toBe(2);
  });
});
