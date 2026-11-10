// Device errors: logged once each, the first with the device (adapter and key
// limits), observable; uncaptured ones from the device's event.

import { describe, expect, it, vi } from "vitest";
import { describeDevice, deviceErrors, onDeviceError, reportDeviceError, watchDeviceErrors } from "../packages/rendering/src/runtime/deviceErrors.js";

function fakeDevice(): GPUDevice & EventTarget {
  const d = new EventTarget() as unknown as Record<string, unknown>;
  d.limits = { maxTextureDimension2D: 16384, maxTextureArrayLayers: 2048, maxBufferSize: 4294967296, maxStorageBufferBindingSize: 2147483648, maxStorageBuffersPerShaderStage: 31 };
  d.adapterInfo = { vendor: "apple", architecture: "metal-3", device: "", description: "" };
  return d as unknown as GPUDevice & EventTarget;
}

describe("device errors", () => {
  it("describes the device: adapter and key limits", () => {
    expect(describeDevice(fakeDevice()).startsWith(
      "apple / metal-3; maxTextureDimension2D 16384, maxTextureArrayLayers 2048, maxBufferSize 4096 MB, " +
      "maxStorageBufferBindingSize 2048 MB, maxStorageBuffersPerShaderStage 31")).toBe(true);
  });

  it("logs each error, the first with the device; subscribers and the record get them", () => {
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      const device = fakeDevice();
      watchDeviceErrors(device);
      watchDeviceErrors(device); // once per device
      const seen: string[] = [];
      onDeviceError(device, (e) => seen.push(e.kind));
      const ev = new Event("uncapturederror");
      (ev as unknown as { error: unknown }).error = { message: "Invalid Buffer" };
      device.dispatchEvent(ev);
      reportDeviceError(device, { kind: "frame", message: "Error: heap chunk spill", recovered: true }, false);
      expect(seen).toEqual(["unknown", "frame"]);
      expect(deviceErrors(device).map((e) => e.message)).toEqual(["Invalid Buffer", "Error: heap chunk spill"]);
      expect(log.mock.calls.map((c) => String(c[0]))).toEqual([
        expect.stringMatching(/^\[wombat\] GPU device: apple \/ metal-3; maxTextureDimension2D 16384/),
        "[wombat] GPU device error (unknown): Invalid Buffer",
      ]);
    } finally { log.mockRestore(); }
  });
});
