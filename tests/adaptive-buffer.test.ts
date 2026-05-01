// prepareAdaptiveBuffer — verify lifecycle and upload behaviour
// against a mock GPUDevice.

import { describe, expect, it } from "vitest";
import { AdaptiveToken, cval, transact } from "@aardworx/wombat.adaptive";
import { V2fArray } from "@aardworx/wombat.base";
import { IBuffer } from "@aardworx/wombat.rendering/core";
import { prepareAdaptiveBuffer } from "@aardworx/wombat.rendering/resources";
import { MockGPU, type MockBuffer } from "./_mockGpu.js";

const VERTEX_USAGE = 0x20; // GPUBufferUsage.VERTEX

function asMock(b: GPUBuffer | undefined): MockBuffer {
  if (!b) throw new Error("buffer undefined");
  return b as unknown as MockBuffer;
}

describe("prepareAdaptiveBuffer", () => {
  it("host source: allocates, uploads, reuses on size-fit", () => {
    const gpu = new MockGPU();
    const arr = new V2fArray(4); // 32 bytes
    const src = cval(IBuffer.fromHost(arr.buffer));
    const r = prepareAdaptiveBuffer(gpu.device, src, { usage: VERTEX_USAGE, label: "positions" });
    r.acquire();
    const tok = AdaptiveToken.top;
    const handle1 = r.getValue(tok);
    expect(asMock(handle1).id).toBe(1);
    expect(asMock(handle1).size).toBe(32);
    expect(asMock(handle1).label).toBe("positions");
    expect(gpu.buffers).toHaveLength(1);
    expect(gpu.writeBufferCalls).toHaveLength(1);
    expect(gpu.writeBufferCalls[0]!.size).toBe(32);

    // Same-size update — must reuse.
    transact(() => { src.value = IBuffer.fromHost(new V2fArray(4).buffer); });
    const handle2 = r.getValue(tok);
    expect(asMock(handle2).id).toBe(1);
    expect(gpu.buffers).toHaveLength(1);
    expect(gpu.writeBufferCalls).toHaveLength(2);

    r.release();
    expect(gpu.buffers[0]!.destroyed).toBe(true);
  });

  it("host source: reallocates on growth", () => {
    const gpu = new MockGPU();
    const src = cval(IBuffer.fromHost(new V2fArray(4).buffer));
    const r = prepareAdaptiveBuffer(gpu.device, src, { usage: VERTEX_USAGE });
    r.acquire();
    const a = asMock(r.getValue(AdaptiveToken.top));
    expect(a.size).toBe(32);

    transact(() => { src.value = IBuffer.fromHost(new V2fArray(64).buffer); });
    const b = asMock(r.getValue(AdaptiveToken.top));
    expect(b.id).not.toBe(a.id);
    expect(b.size).toBe(64 * 8);
    expect(gpu.buffers).toHaveLength(2);
    expect(gpu.buffers[0]!.destroyed).toBe(true);   // old freed

    r.release();
  });

  it("gpu source: passes through, no allocation", () => {
    const gpu = new MockGPU();
    const userBuf = { size: 1024 } as GPUBuffer;
    const src = cval(IBuffer.fromGPU(userBuf));
    const r = prepareAdaptiveBuffer(gpu.device, src, { usage: VERTEX_USAGE });
    r.acquire();
    expect(r.getValue(AdaptiveToken.top)).toBe(userBuf);
    expect(gpu.buffers).toHaveLength(0);
    expect(gpu.writeBufferCalls).toHaveLength(0);
    r.release();
  });

  it("gpu→host transition allocates; host→gpu transition frees", () => {
    const gpu = new MockGPU();
    const userBuf = { size: 64 } as GPUBuffer;
    const src = cval<ReturnType<typeof IBuffer.fromGPU> | ReturnType<typeof IBuffer.fromHost>>(
      IBuffer.fromGPU(userBuf),
    );
    const r = prepareAdaptiveBuffer(gpu.device, src, { usage: VERTEX_USAGE });
    r.acquire();
    expect(r.getValue(AdaptiveToken.top)).toBe(userBuf);

    transact(() => { src.value = IBuffer.fromHost(new V2fArray(8).buffer); });
    const ownedHandle = asMock(r.getValue(AdaptiveToken.top));
    expect(ownedHandle.size).toBe(64);
    expect(gpu.buffers).toHaveLength(1);
    expect(ownedHandle.destroyed).toBe(false);

    transact(() => { src.value = IBuffer.fromGPU(userBuf); });
    expect(r.getValue(AdaptiveToken.top)).toBe(userBuf);
    expect(gpu.buffers[0]!.destroyed).toBe(true);   // owned freed on switch

    r.release();
  });

  it("host source via TypedArray view (with byteOffset) uploads correct slice", () => {
    const gpu = new MockGPU();
    const big = new Float32Array(16);
    for (let i = 0; i < 16; i++) big[i] = i;
    // Slice viewing elements 4..12 (32 bytes at byteOffset 16).
    const view = new Float32Array(big.buffer, 16, 8);
    const src = cval(IBuffer.fromHost(view));
    const r = prepareAdaptiveBuffer(gpu.device, src, { usage: VERTEX_USAGE });
    r.acquire();
    r.getValue(AdaptiveToken.top);
    expect(gpu.writeBufferCalls).toHaveLength(1);
    const c = gpu.writeBufferCalls[0]!;
    expect(c.dataOffset).toBe(16);
    expect(c.size).toBe(32);
    r.release();
  });

  it("acquire/release ref-counting", () => {
    const gpu = new MockGPU();
    const src = cval(IBuffer.fromHost(new V2fArray(2).buffer));
    const r = prepareAdaptiveBuffer(gpu.device, src, { usage: VERTEX_USAGE });
    r.acquire();
    r.acquire();
    r.getValue(AdaptiveToken.top);
    expect(gpu.buffers[0]!.destroyed).toBe(false);
    r.release();
    expect(gpu.buffers[0]!.destroyed).toBe(false);
    r.release();
    expect(gpu.buffers[0]!.destroyed).toBe(true);
  });
});
