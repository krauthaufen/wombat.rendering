// In-place GPU resources on a real GPU: region writes into a 3D
// texture, buffer range writes + growth, readback helpers, and a
// ComputeBatch whose dispatches see their own uniform values while
// sharing one submit.

import { describe, expect, it } from "vitest";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { computeShader, type ComputeShader } from "@aardworx/wombat.shader";
import { Tu32, type Module, type Type, type ValueDef } from "@aardworx/wombat.shader/ir";
import {
  ComputeBatch, MutableBuffer, MutableTexture,
  prepareComputeShader, readBuffer, readTextureRegion, writeTextureRegion,
} from "@aardworx/wombat.rendering/resources";
import { requestRealDevice } from "./_realGpu.js";

const N = 256;

// data[i] = data[i] + offset * (i + 1)
function accumulateShader(): ComputeShader {
  const source = `
    declare const data: number[];
    declare const offset: u32;

    /** @workgroupSize 64 1 1 */
    export function csMain(b: ComputeBuiltins): void {
      const i = b.globalInvocationId.x as i32;
      data[i] = data[i] + offset * ((i + 1) as u32);
    }
  `;
  const sb: ValueDef = {
    kind: "StorageBuffer",
    binding: { group: 0, slot: 0 },
    name: "data",
    layout: { kind: "Array", element: { kind: "Int", signed: false, width: 32 }, length: N },
    access: "read_write",
  };
  const ub: ValueDef = {
    kind: "Uniform",
    uniforms: [{ name: "offset", type: Tu32, group: 0, slot: 1, buffer: "params" }],
  };
  const externalTypes = new Map<string, Type>([["data", sb.layout], ["offset", Tu32]]);
  const entries: EntryRequest[] = [{ name: "csMain", stage: "compute" }];
  const parsed = parseShader({ source, entries, externalTypes });
  const merged: Module = { ...parsed, values: [sb, ub, ...parsed.values] };
  return computeShader(merged);
}

async function withDevice(f: (device: GPUDevice, errors: GPUError[]) => Promise<void>): Promise<void> {
  const device = await requestRealDevice();
  const errors: GPUError[] = [];
  device.onuncapturederror = (e) => errors.push(e.error);
  try {
    await f(device, errors);
    await device.queue.onSubmittedWorkDone();
    expect(errors.map(e => e.message)).toEqual([]);
  } finally {
    device.destroy();
  }
}

describe("in-place GPU resources — real GPU", () => {
  it("MutableTexture: 3D region writes land in place, the rest stays zero", () => withDevice(async (device) => {
    const tex = new MutableTexture(device, { size: [12, 8, 16], format: "rg8unorm", dimension: "3d" });
    expect([tex.width, tex.height, tex.depthOrArrayLayers]).toEqual([12, 8, 16]);
    const brick = new Uint8Array(4 * 4 * 4 * 2);
    for (let i = 0; i < brick.length; i++) brick[i] = (i % 251) + 1;
    tex.write(brick, { origin: [4, 2, 8], size: [4, 4, 4] });

    const back = new Uint8Array(await tex.read({ origin: [4, 2, 8], size: [4, 4, 4] }));
    expect(Array.from(back)).toEqual(Array.from(brick));

    const all = new Uint8Array(await tex.read());
    expect(all.byteLength).toBe(12 * 8 * 16 * 2);
    let nonZero = 0;
    for (let z = 0; z < 16; z++) for (let y = 0; y < 8; y++) for (let x = 0; x < 12; x++) {
      const inside = x >= 4 && x < 8 && y >= 2 && y < 6 && z >= 8 && z < 12;
      const o = ((z * 8 + y) * 12 + x) * 2;
      if (all[o] !== 0 || all[o + 1] !== 0) {
        nonZero++;
        expect(inside).toBe(true);
      }
    }
    expect(nonZero).toBe(64);
    tex.dispose();
  }), 30000);

  it("MutableTexture is an aval: writes mark it (coalesced), notify:false does not", () => withDevice(async (device) => {
    const tex = new MutableTexture(device, { size: [4, 4], format: "rgba8unorm" });
    // a dependent goes out of date when the texture is marked (its value,
    // the same gpu ITexture, does not change — map skips re-running f)
    const dependent = tex.map((t) => t);
    dependent.force();
    expect(dependent.outOfDate).toBe(false);
    const px = new Uint8Array(4 * 4 * 4).fill(7);
    tex.write(px);
    tex.write(px);
    expect(dependent.outOfDate).toBe(false);   // the mark is deferred to a microtask
    await Promise.resolve();
    expect(dependent.outOfDate).toBe(true);
    dependent.force();
    tex.write(px, {}, { notify: false });
    await new Promise(r => setTimeout(r, 0));
    expect(dependent.outOfDate).toBe(false);
    expect(tex.force().kind).toBe("gpu");
    tex.dispose();
  }), 30000);

  it("readTextureRegion strips the 256-byte row padding", () => withDevice(async (device) => {
    const w = 13, h = 5;
    const tex = device.createTexture({ size: [w, h], format: "rgba8unorm", usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING });
    const px = new Uint8Array(w * h * 4);
    for (let i = 0; i < px.length; i++) px[i] = i & 0xff;
    writeTextureRegion(device, tex, px);
    expect(Array.from(new Uint8Array(await readTextureRegion(device, tex)))).toEqual(Array.from(px));
    const sub = new Uint8Array(await readTextureRegion(device, tex, { origin: [2, 1], size: [3, 2] }));
    for (let y = 0; y < 2; y++) for (let x = 0; x < 3; x++) for (let c = 0; c < 4; c++) {
      expect(sub[(y * 3 + x) * 4 + c]).toBe(px[((y + 1) * w + x + 2) * 4 + c]);
    }
    tex.destroy();
  }), 30000);

  it("MutableBuffer: range writes, growth keeps contents and changes the aval", () => withDevice(async (device) => {
    const buf = new MutableBuffer(device, { size: 64 });
    const first = buf.force();
    buf.write(8, new Float32Array([1, 2, 3, 4]));
    buf.write(0, new Uint32Array([10, 11, 12, 13]), 8, 8);   // bytes 8..15 of the source: 12, 13
    expect(Array.from(new Float32Array(await buf.read(8, 16)))).toEqual([1, 2, 3, 4]);
    expect(Array.from(new Uint32Array(await buf.read(0, 8)))).toEqual([12, 13]);
    expect(Array.from(new Uint8Array(await readBuffer(device, buf.buffer, 9, 3)))).toEqual(
      Array.from(new Uint8Array(new Float32Array([1]).buffer)).slice(1));

    expect(buf.ensureCapacity(32)).toBe(false);
    expect(buf.ensureCapacity(1000)).toBe(true);
    expect(buf.size).toBeGreaterThanOrEqual(1000);
    await Promise.resolve();
    const second = buf.force();
    expect(second).not.toBe(first);
    expect(second.kind === "gpu" && second.buffer).toBe(buf.buffer);
    expect(Array.from(new Float32Array(await buf.read(8, 16)))).toEqual([1, 2, 3, 4]);
    buf.dispose();
  }), 30000);

  it("ComputeBatch: per-dispatch uniforms in one submit, interleaved writes, readback", () => withDevice(async (device) => {
    const shader = prepareComputeShader(device, accumulateShader(), { label: "accumulate" });
    const data = new MutableBuffer(device, { size: N * 4 });
    const binding = shader.createInputBinding().setBuffer("data", data.buffer);
    const batch = new ComputeBatch(device, { label: "test-batch" });

    // 1..10 in ONE submit: every dispatch must see its own offset
    for (let k = 1; k <= 10; k++) {
      binding.setUniform("offset", k);
      batch.dispatch(shader, binding, [N / 64, 1, 1]);
    }
    expect(batch.pendingCount).toBe(10);
    let out = new Uint32Array(await batch.read(data.buffer));
    expect(batch.submitCount).toBe(1);
    for (let i = 0; i < N; i++) expect(out[i]).toBe(55 * (i + 1));

    // a host write between dispatches is ordered after the first and before the second
    binding.setUniform("offset", 1);
    batch.dispatch(shader, binding, [N / 64, 1, 1]);
    batch.writeBuffer(data.buffer, 0, new Uint32Array([1000]));
    batch.dispatch(shader, binding, [N / 64, 1, 1]);
    out = new Uint32Array(await batch.read(data.buffer, 0, 8));
    expect(out[0]).toBe(1000 + 1);
    expect(out[1]).toBe(55 * 2 + 2 * 2);

    // more dispatches than one uniform chunk holds (64 KB / 256 B)
    data.write(0, new Uint32Array(N));
    for (let k = 0; k < 600; k++) {
      binding.setUniform("offset", k % 3);
      batch.dispatch(shader, binding, [N / 64, 1, 1]);
      if (k % 100 === 99) batch.submit();
    }
    out = new Uint32Array(await batch.read(data.buffer));
    const sum = 600 / 3 * (0 + 1 + 2);
    for (let i = 0; i < N; i++) expect(out[i]).toBe(sum * (i + 1));
    expect(batch.dispatchCount).toBe(612);

    batch.dispose();
    binding.dispose();
    data.dispose();
  }), 30000);
});
