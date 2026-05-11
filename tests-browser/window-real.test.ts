// Window package on real GPU. Verifies attachCanvas + runFrame
// wire up correctly without crashing or producing GPU validation
// errors. Pixel-readback assertions live in render-real / clear-real
// where we control the texture lifetime explicitly.

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { AList, AdaptiveToken, HashMap, cval, transact } from "@aardworx/wombat.adaptive";
import { V4f } from "@aardworx/wombat.base";
import type { ClearValues, Command } from "@aardworx/wombat.rendering/core";
import { Runtime } from "@aardworx/wombat.rendering/runtime";
import { attachCanvas, runFrame } from "@aardworx/wombat.rendering/window";
import { requestRealDevice } from "./_realGpu.js";

let device!: GPUDevice;
const errors: GPUError[] = [];

beforeAll(async () => {
  device = await requestRealDevice();
  device.onuncapturederror = (e) => errors.push(e.error);
});
afterAll(() => { device.destroy(); });

describe("window — canvas attach", () => {
  it("attachCanvas: signature mirrors the chosen formats", () => {
    const canvas = document.createElement("canvas");
    canvas.width = 8; canvas.height = 8;
    document.body.appendChild(canvas);
    const window_ = attachCanvas(device, canvas, {
      colorAttachmentName: "color", format: "rgba8unorm",
      depthFormat: "depth24plus",
    });
    expect(window_.signature.colors.tryFind("color")).toBe("rgba8unorm");
    expect(window_.signature.depthStencil?.format).toBe("depth24plus");
    expect(window_.signature.depthStencil?.hasDepth).toBe(true);
    window_.dispose();
    canvas.remove();
  });

  it("attachCanvas + manual frame loop: clears land without errors", () => {
    errors.length = 0;
    const canvas = document.createElement("canvas");
    canvas.width = 8; canvas.height = 8;
    document.body.appendChild(canvas);
    const window_ = attachCanvas(device, canvas, {
      colorAttachmentName: "color", format: "rgba8unorm",
    });
    const runtime = new Runtime({ device });
    for (let i = 0; i < 3; i++) {
      window_.markFrame();
      runtime.compile(AList.ofArray<Command>([
        { kind: "Clear", output: window_.framebuffer, values: {
          colors: HashMap.empty<string, V4f>().add("color", new V4f(i / 3, 0, 0, 1)),
        } as ClearValues },
      ])).run(AdaptiveToken.top);
    }
    expect(errors).toEqual([]);
    window_.dispose();
    canvas.remove();
  });

  it("runFrame: rAF loop fires N callbacks and stops", async () => {
    errors.length = 0;
    const canvas = document.createElement("canvas");
    canvas.width = 4; canvas.height = 4;
    document.body.appendChild(canvas);
    const window_ = attachCanvas(device, canvas, {
      colorAttachmentName: "color", format: "rgba8unorm",
    });
    const runtime = new Runtime({ device });
    const tickC = cval(0);
    const clearC = cval(new V4f(0, 0, 0, 1));
    let count = 0;
    const done = new Promise<void>((resolve) => {
      runFrame(window_, (token) => {
        // Read tickC through the token so the wrapping renderAval
        // gains a dependency on it — otherwise the marking callback
        // never fires and the rAF loop stalls after one frame.
        const t = tickC.getValue(token);
        transact(() => { clearC.value = new V4f(t / 5, 0, 0, 1); });
        runtime.compile(AList.ofArray<Command>([
          { kind: "Clear", output: window_.framebuffer, values: {
            colors: HashMap.empty<string, V4f>().add("color", clearC.value),
          } as ClearValues },
        ])).run(token);
        count++;
        if (count >= 5) resolve();
      }, {
        maxFrames: 5,
        onAfterFrame: () => { tickC.value = tickC.value + 1; },
      });
    });
    await done;
    expect(count).toBe(5);
    expect(errors).toEqual([]);
    window_.dispose();
    canvas.remove();
  }, 30000);
});
