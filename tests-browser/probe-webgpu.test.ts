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
    const device = await requestRealDevice();
    expect(device).toBeDefined();
    expect(typeof device.createBuffer).toBe("function");
    expect(typeof device.createTexture).toBe("function");
    device.destroy();
  });
});
