// Clear-pass on a real GPU. Allocates a 2×2 framebuffer, runs a
// `Clear` command with a known color, reads back pixels, asserts.
//
// This is the first real-GPU validation of the rendering layer:
// proves that allocateFramebuffer, the clear() helper, and
// Runtime.compile(alist<Command>) actually drive a WebGPU device.

import { describe, expect, it } from "vitest";
import { AList, AdaptiveToken, HashMap, cval } from "@aardworx/wombat.adaptive";
import { V4f } from "@aardworx/wombat.base";
import type { ClearValues, Command } from "@aardworx/wombat.rendering-core";
import {
  allocateFramebuffer,
  createFramebufferSignature,
  TextureUsage,
} from "@aardworx/wombat.rendering-resources";
import { Runtime } from "@aardworx/wombat.rendering-runtime";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";

describe("clear pass — real GPU", () => {
  it("clears a 2×2 rgba8unorm framebuffer to a known color", async () => {
    const device = await requestRealDevice();
    try {
      const sig = createFramebufferSignature({ colors: { color: "rgba8unorm" } });
      const fbo = allocateFramebuffer(device, sig, cval({ width: 2, height: 2 }), {
        // We need COPY_SRC on the color texture so we can read it back.
        extraUsage: TextureUsage.COPY_SRC,
      });
      fbo.acquire();

      const runtime = new Runtime({ device });
      const clearValues: ClearValues = {
        colors: HashMap.empty<string, V4f>().add("color", new V4f(1, 0.5, 0.25, 1)),
      };
      const task = runtime.compile(AList.ofArray<Command>([
        { kind: "Clear", output: fbo, values: clearValues },
      ]));
      task.run(AdaptiveToken.top);
      await device.queue.onSubmittedWorkDone();

      const ifb = fbo.getValue(AdaptiveToken.top);
      const tex = ifb.colorTextures!.tryFind("color")!;
      const pixels = await readTexturePixels(device, tex);
      // 2×2 RGBA = 16 bytes. Each pixel must be roughly (255, 128, 64, 255).
      expect(pixels.length).toBe(16);
      for (let i = 0; i < 4; i++) {
        const r = pixels[i * 4 + 0]!;
        const g = pixels[i * 4 + 1]!;
        const b = pixels[i * 4 + 2]!;
        const a = pixels[i * 4 + 3]!;
        expect(r).toBe(255);
        expect(g).toBeGreaterThan(120);
        expect(g).toBeLessThan(135);
        expect(b).toBeGreaterThan(60);
        expect(b).toBeLessThan(70);
        expect(a).toBe(255);
      }

      task.dispose();
      fbo.release();
    } finally {
      device.destroy();
    }
  });
});
