// Runtime lifecycle: disposeAll() tears down outstanding tasks +
// blocks future compile/renderTo. device.lost handling: when the
// mock fires its lost-promise, the runtime auto-disposes.

import { describe, expect, it } from "vitest";
import { AList } from "@aardworx/wombat.adaptive";
import type { Command } from "@aardworx/wombat.rendering-core";
import { Runtime } from "@aardworx/wombat.rendering-runtime";
import { MockGPU } from "./_mockGpu.js";

describe("Runtime lifecycle", () => {
  it("disposeAll() disposes outstanding tasks and forbids further compiles", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device });
    const t1 = runtime.compile(AList.empty<Command>());
    const t2 = runtime.compile(AList.empty<Command>());
    expect(() => runtime.disposeAll()).not.toThrow();
    // Subsequent dispose calls on already-disposed tasks should be silent.
    expect(() => t1.dispose()).not.toThrow();
    expect(() => t2.dispose()).not.toThrow();
    expect(() => runtime.compile(AList.empty<Command>())).toThrow(/disposeAll/);
  });

  it("device.lost handler fires disposeAll on lost-promise resolution", async () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device });
    runtime.compile(AList.empty<Command>());
    expect(runtime.isDeviceLost).toBe(false);
    gpu.simulateLost({ reason: "destroyed", message: "test" } as GPUDeviceLostInfo);
    await runtime.deviceLost;
    expect(runtime.isDeviceLost).toBe(true);
    expect(() => runtime.compile(AList.empty<Command>())).toThrow();
  });
});
