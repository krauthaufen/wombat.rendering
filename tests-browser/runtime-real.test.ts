// Custom + Copy commands on real GPU. Validates the runtime walker's
// encoder threading for the simpler command kinds.

import { describe, expect, it } from "vitest";
import { AList, AdaptiveToken } from "@aardworx/wombat.adaptive";
import type { Command } from "@aardworx/wombat.rendering-core";
import { Runtime } from "@aardworx/wombat.rendering-runtime";
import { requestRealDevice } from "./_realGpu.js";

describe("runtime — Custom and Copy on real GPU", () => {
  it("Custom command receives the active encoder", async () => {
    const device = await requestRealDevice();
    try {
      const runtime = new Runtime({ device });
      let encSeen: GPUCommandEncoder | null = null;
      const cmds = AList.ofArray<Command>([
        { kind: "Custom", encode: (enc) => { encSeen = enc; } },
      ]);
      runtime.compile(cmds).run(AdaptiveToken.top);
      await device.queue.onSubmittedWorkDone();
      expect(encSeen).not.toBeNull();
      expect(typeof encSeen!.beginRenderPass).toBe("function");
    } finally {
      device.destroy();
    }
  });

  it("Copy moves bytes between two GPUBuffers", async () => {
    const device = await requestRealDevice();
    try {
      // Source: write known bytes via writeBuffer.
      const data = new Uint8Array([10, 20, 30, 40, 50, 60, 70, 80]);
      const src = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(src, 0, data);

      const dst = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });

      const runtime = new Runtime({ device });
      const cmds = AList.ofArray<Command>([
        { kind: "Copy", copy: { kind: "buffer", src, dst, range: { srcOffset: 0, dstOffset: 0, size: data.byteLength } } },
      ]);
      runtime.compile(cmds).run(AdaptiveToken.top);
      await device.queue.onSubmittedWorkDone();

      // WebGPU forbids MAP_READ together with COPY_SRC on the same
      // buffer; copy `dst` once more into a dedicated readback buffer.
      const readback = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(dst, 0, readback, 0, data.byteLength);
      device.queue.submit([enc.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const read = new Uint8Array(readback.getMappedRange().slice(0));
      readback.unmap();
      expect(Array.from(read)).toEqual(Array.from(data));

      src.destroy(); dst.destroy(); readback.destroy();
    } finally {
      device.destroy();
    }
  });
});
