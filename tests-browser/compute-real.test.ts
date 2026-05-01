// Compute primitive on real GPU. Build a `ComputeShader` (single
// compute stage, peer of `Effect`) with one storage buffer + one
// uniform block, dispatch via the imperative `ComputeInputBinding`,
// copy the buffer to a readback buffer, verify each slot is
// `gid + offset`.

import { describe, expect, it } from "vitest";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { computeShader, type ComputeShader } from "@aardworx/wombat.shader";
import {
  Tu32, Vec, type Module, type Type, type ValueDef,
} from "@aardworx/wombat.shader/ir";
import {
  prepareComputeShader,
  type DispatchSize,
} from "@aardworx/wombat.rendering-resources";
import { requestRealDevice } from "./_realGpu.js";

const SIZE = 64; // 64-element storage buffer; dispatch 8 workgroups × 8 lanes.

function buildShader(): ComputeShader {
  // Same idiom as wombat.shader's compute-frontend tests: bare
  // `declare const out_buffer: number[]` + `b: ComputeBuiltins`,
  // workgroup size via JSDoc tag.
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
    access: "read", // flipped to read_write by inferStorageAccess.
  };
  void Vec;
  void Tu32;

  const externalTypes = new Map<string, Type>();
  externalTypes.set("out_buffer", storageBuffer.layout);

  const entries: EntryRequest[] = [
    { name: "csMain", stage: "compute" },
  ];
  const parsed = parseShader({ source, entries, externalTypes });
  const merged: Module = {
    ...parsed,
    values: [storageBuffer, ...parsed.values],
  };
  return computeShader(merged);
}

describe("compute primitive — real GPU", () => {
  it("imperative input binding + dispatch fills storage buffer", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const shader = buildShader();
      const prepared = prepareComputeShader(device, shader, { label: "compute-test" });

      const storage = device.createBuffer({
        size: SIZE * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        label: "out_buffer",
      });

      const binding = prepared.createInputBinding();
      binding.setBuffer("out_buffer", storage);

      const groups: DispatchSize = { x: SIZE / 8, y: 1, z: 1 };
      await prepared.dispatch(binding, groups);

      const readback = device.createBuffer({
        size: SIZE * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(storage, 0, readback, 0, SIZE * 4);
      device.queue.submit([enc.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const data = new Uint32Array(readback.getMappedRange().slice(0));
      readback.unmap();

      expect(errors).toEqual([]);
      for (let i = 0; i < SIZE; i++) {
        expect(data[i]).toBe(i + 1000);
      }

      binding.dispose();
      storage.destroy();
      readback.destroy();
    } finally {
      device.destroy();
    }
  }, 30000);
});
