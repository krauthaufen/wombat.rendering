// Compute primitive on real GPU. Builds `ComputeShader`s (single
// compute stage, peer of `Effect`) and dispatches them via the
// imperative `ComputeInputBinding`. Three scenarios:
//
//   1. Storage buffer write: `out[i] = i + something`.
//   2. Same as 1, but `something` arrives via a uniform block.
//   3. Storage texture write: image-fill kernel writing
//      `(x, y, 0, 1)` per texel into an `rgba8unorm` storage texture.

import { describe, expect, it } from "vitest";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { computeShader, type ComputeShader } from "@aardworx/wombat.shader";
import {
  Tu32, type Module, type Type, type ValueDef,
} from "@aardworx/wombat.shader/ir";
import {
  prepareComputeShader,
  type DispatchSize,
} from "@aardworx/wombat.rendering-resources";
import { requestRealDevice } from "./_realGpu.js";

const SIZE = 64;

function buildBufferShader(): ComputeShader {
  const source = `
    declare const out_buffer: number[];

    /** @workgroupSize 8 1 1 */
    export function csMain(b: ComputeBuiltins): void {
      const i = b.globalInvocationId.x as i32;
      out_buffer[i] = ((i + 1000) as u32);
    }
  `;
  const storageBuffer: ValueDef = {
    kind: "StorageBuffer",
    binding: { group: 0, slot: 0 },
    name: "out_buffer",
    layout: {
      kind: "Array",
      element: { kind: "Int", signed: false, width: 32 },
      length: SIZE,
    },
    access: "read",
  };
  const externalTypes = new Map<string, Type>();
  externalTypes.set("out_buffer", storageBuffer.layout);
  const entries: EntryRequest[] = [{ name: "csMain", stage: "compute" }];
  const parsed = parseShader({ source, entries, externalTypes });
  const merged: Module = { ...parsed, values: [storageBuffer, ...parsed.values] };
  return computeShader(merged);
}

function buildBufferShaderWithUniform(): ComputeShader {
  const source = `
    declare const out_buffer: number[];
    declare const offset: u32;

    /** @workgroupSize 8 1 1 */
    export function csMain(b: ComputeBuiltins): void {
      const i = b.globalInvocationId.x as i32;
      out_buffer[i] = ((i as u32) + offset);
    }
  `;
  const storageBuffer: ValueDef = {
    kind: "StorageBuffer",
    binding: { group: 0, slot: 0 },
    name: "out_buffer",
    layout: {
      kind: "Array",
      element: { kind: "Int", signed: false, width: 32 },
      length: SIZE,
    },
    access: "read",
  };
  // UniformDecl carries its own group/slot/buffer; same `buffer`
  // name groups members into one UBO at that binding.
  const uniform: ValueDef = {
    kind: "Uniform",
    uniforms: [
      { name: "offset", type: Tu32, group: 0, slot: 1, buffer: "params" },
    ],
  };
  const externalTypes = new Map<string, Type>();
  externalTypes.set("out_buffer", storageBuffer.layout);
  externalTypes.set("offset", Tu32);
  const entries: EntryRequest[] = [{ name: "csMain", stage: "compute" }];
  const parsed = parseShader({ source, entries, externalTypes });
  const merged: Module = {
    ...parsed,
    values: [storageBuffer, uniform, ...parsed.values],
  };
  return computeShader(merged);
}

const TEX_DIM = 8;

function buildStorageTextureShader(): ComputeShader {
  // Image-fill kernel: every workgroup invocation writes one texel
  // with `(x, y, 0, 1)` (normalised 0..1 in x/y, 0 in b, 1 in a).
  const source = `
    declare const out: StorageTexture2D<"rgba8unorm", "write">;

    /** @workgroupSize 8 8 1 */
    export function csMain(b: ComputeBuiltins): void {
      const x = b.globalInvocationId.x as i32;
      const y = b.globalInvocationId.y as i32;
      const r = (x as f32) / ${TEX_DIM - 1}.0;
      const g = (y as f32) / ${TEX_DIM - 1}.0;
      textureStore(out, new V2i(x, y), new V4f(r, g, 0.0, 1.0));
    }
  `;
  const stex: ValueDef = {
    kind: "Sampler",
    binding: { group: 0, slot: 0 },
    name: "out",
    type: {
      kind: "StorageTexture",
      target: "2d",
      format: "rgba8unorm",
      access: "write",
      arrayed: false,
    },
  };
  const externalTypes = new Map<string, Type>();
  externalTypes.set("out", stex.type);

  const entries: EntryRequest[] = [{ name: "csMain", stage: "compute" }];
  const parsed = parseShader({ source, entries, externalTypes });
  const merged: Module = { ...parsed, values: [stex, ...parsed.values] };
  return computeShader(merged);
}

describe("compute primitive — real GPU", () => {
  it("storage buffer dispatch fills out_buffer with gid+1000", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const prepared = prepareComputeShader(device, buildBufferShader(), { label: "compute-buf" });
      const storage = device.createBuffer({
        size: SIZE * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const binding = prepared.createInputBinding();
      binding.setBuffer("out_buffer", storage);
      const groups: DispatchSize = { x: SIZE / 8, y: 1, z: 1 };
      await prepared.dispatch(binding, groups);

      const data = await readbackU32(device, storage, SIZE);
      expect(errors).toEqual([]);
      for (let i = 0; i < SIZE; i++) expect(data[i]).toBe(i + 1000);

      binding.dispose();
      storage.destroy();
    } finally {
      device.destroy();
    }
  }, 30000);

  it("uniform-block input is routed through setUniform", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const prepared = prepareComputeShader(device, buildBufferShaderWithUniform(), { label: "compute-ubo" });
      const storage = device.createBuffer({
        size: SIZE * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const binding = prepared.createInputBinding();
      binding.setBuffer("out_buffer", storage);
      binding.setUniform("offset", 4242);
      await prepared.dispatch(binding, { x: SIZE / 8, y: 1, z: 1 });

      const data = await readbackU32(device, storage, SIZE);
      expect(errors).toEqual([]);
      for (let i = 0; i < SIZE; i++) expect(data[i]).toBe(i + 4242);

      // Re-dispatch with a different offset; same buffer, no re-binding.
      binding.setUniform("offset", 1);
      await prepared.dispatch(binding, { x: SIZE / 8, y: 1, z: 1 });
      const data2 = await readbackU32(device, storage, SIZE);
      for (let i = 0; i < SIZE; i++) expect(data2[i]).toBe(i + 1);

      binding.dispose();
      storage.destroy();
    } finally {
      device.destroy();
    }
  }, 30000);

  it("storage-texture write emits the expected per-texel pattern", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const prepared = prepareComputeShader(device, buildStorageTextureShader(), { label: "compute-stex" });
      const tex = device.createTexture({
        size: { width: TEX_DIM, height: TEX_DIM, depthOrArrayLayers: 1 },
        format: "rgba8unorm",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
        label: "stex.out",
      });
      const binding = prepared.createInputBinding();
      binding.setStorageTexture("out", tex);
      await prepared.dispatch(binding, { x: 1, y: 1, z: 1 });

      const pixels = await readTexture(device, tex);
      expect(errors).toEqual([]);
      for (let y = 0; y < TEX_DIM; y++) {
        for (let x = 0; x < TEX_DIM; x++) {
          const i = (y * TEX_DIM + x) * 4;
          const expR = Math.round((x / (TEX_DIM - 1)) * 255);
          const expG = Math.round((y / (TEX_DIM - 1)) * 255);
          expect(Math.abs(pixels[i + 0]! - expR)).toBeLessThanOrEqual(1);
          expect(Math.abs(pixels[i + 1]! - expG)).toBeLessThanOrEqual(1);
          expect(pixels[i + 2]).toBe(0);
          expect(pixels[i + 3]).toBe(255);
        }
      }

      binding.dispose();
      tex.destroy();
    } finally {
      device.destroy();
    }
  }, 30000);
});

async function readbackU32(device: GPUDevice, storage: GPUBuffer, count: number): Promise<Uint32Array> {
  const bytes = count * 4;
  const readback = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(storage, 0, readback, 0, bytes);
  device.queue.submit([enc.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const data = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return data;
}

async function readTexture(device: GPUDevice, tex: GPUTexture): Promise<Uint8Array> {
  const w = tex.width, h = tex.height;
  const bpr = Math.ceil((w * 4) / 256) * 256;
  const staging = device.createBuffer({
    size: bpr * h,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer(
    { texture: tex },
    { buffer: staging, bytesPerRow: bpr, rowsPerImage: h },
    { width: w, height: h, depthOrArrayLayers: 1 },
  );
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    out.set(padded.subarray(y * bpr, y * bpr + w * 4), y * w * 4);
  }
  return out;
}
