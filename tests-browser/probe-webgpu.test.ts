// Smoke check: WebGPU is reachable in this browser, with a real
// adapter that we can read identifying info from. If this fails the
// rest of the browser test suite is moot.

import { describe, expect, it } from "vitest";
import { requestRealDevice } from "./_realGpu.js";

describe("WebGPU probe", () => {
  it("navigator.gpu is defined", () => {
    expect(typeof navigator).toBe("object");
    expect("gpu" in navigator).toBe(true);
  });

  it("can request an adapter + device", async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no adapter");
    // Surface what the test environment actually picked — useful
    // for confirming whether vitest-browser used the real GPU
    // (e.g. NVIDIA / Blackwell) or fell back to SwiftShader.
    console.log("adapter info:", JSON.stringify({
      vendor: adapter.info?.vendor,
      architecture: adapter.info?.architecture,
      device: adapter.info?.device,
      description: adapter.info?.description,
    }));
    const device = await adapter.requestDevice();
    expect(device).toBeDefined();
    expect(typeof device.createBuffer).toBe("function");
    expect(typeof device.createTexture).toBe("function");
    device.destroy();
  });
});
