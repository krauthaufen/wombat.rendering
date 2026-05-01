// prepareUniformBuffer — pack a name-keyed bag of `aval<unknown>`
// uniforms into a single UBO described by a `UniformBlockInfo`
// (from wombat.shader's ProgramInterface).
//
// The layout dictates byte offsets + sizes per field; this packer
// trusts those numbers and just encodes values into a Float32-
// addressable scratch buffer, then uploads via `queue.writeBuffer`
// on each change.
//
// Value encoding by type:
//   - `number`             → 1 × f32 at offset
//   - `Float32Array`       → memcpy floats
//   - `Int32Array`/`Uint32Array` → memcpy as-is (also valid for f32 storage)
//   - any object with `_data: Float32Array | Int32Array | Uint32Array`
//     (V2f/V3f/V4f/M44f/M33f/M22f, etc. from wombat.base) → memcpy `_data`
//
// Missing inputs (a layout field with no corresponding entry in
// the bag) are left zero-initialised. Extras in the bag are
// silently ignored — same "pull, don't push" semantics as the
// rest of the renderer.

import {
  AdaptiveResource,
  tryAcquire,
  tryRelease,
  type UniformBlockInfo,
} from "@aardworx/wombat.rendering-core";
import {
  type AdaptiveToken,
  type aval,
  type HashMap,
} from "@aardworx/wombat.adaptive";
import { BufferUsage } from "./webgpuFlags.js";

export interface PackedView {
  readonly bytes: ArrayBuffer;
  readonly f32: Float32Array;
  readonly i32: Int32Array;
  readonly u32: Uint32Array;
}

export function makePackedView(byteSize: number): PackedView {
  const bytes = new ArrayBuffer(roundUp4(byteSize));
  return {
    bytes,
    f32: new Float32Array(bytes),
    i32: new Int32Array(bytes),
    u32: new Uint32Array(bytes),
  };
}

class UniformBufferResource extends AdaptiveResource<GPUBuffer> {
  private _gpu: GPUBuffer | undefined;
  private readonly _scratch: PackedView;

  constructor(
    private readonly device: GPUDevice,
    private readonly layout: UniformBlockInfo,
    private readonly inputs: HashMap<string, aval<unknown>>,
    private readonly label: string | undefined,
  ) {
    super();
    const bytes = new ArrayBuffer(roundUp4(layout.size));
    this._scratch = {
      bytes,
      f32: new Float32Array(bytes),
      i32: new Int32Array(bytes),
      u32: new Uint32Array(bytes),
    };
  }

  protected create(): void {
    for (const f of this.layout.fields) {
      const av = this.inputs.tryFind(f.name);
      if (av !== undefined) tryAcquire(av);
    }
  }

  protected destroy(): void {
    if (this._gpu !== undefined) {
      this._gpu.destroy();
      this._gpu = undefined;
    }
    for (const f of this.layout.fields) {
      const av = this.inputs.tryFind(f.name);
      if (av !== undefined) tryRelease(av);
    }
  }

  override compute(token: AdaptiveToken): GPUBuffer {
    // Pack scratch.
    new Uint8Array(this._scratch.bytes).fill(0);
    for (const field of this.layout.fields) {
      const av = this.inputs.tryFind(field.name);
      if (av === undefined) continue;
      writeField(this._scratch, field.offset, av.getValue(token));
    }
    if (this._gpu === undefined) {
      const desc: GPUBufferDescriptor = {
        size: this._scratch.bytes.byteLength,
        usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
        ...(this.label !== undefined ? { label: this.label } : {}),
      };
      this._gpu = this.device.createBuffer(desc);
    }
    this.device.queue.writeBuffer(this._gpu, 0, this._scratch.bytes);
    return this._gpu;
  }
}

/**
 * Write a value into a UBO scratch buffer at the given byte offset.
 * `fieldType`, when supplied, disambiguates a bare `number`:
 *   - Int (signed)    → i32
 *   - Int (unsigned)  → u32
 *   - Float (default) → f32
 * Without it, numbers are assumed to be f32.
 */
export function writeField(
  view: PackedView,
  offset: number,
  value: unknown,
  fieldType?: import("@aardworx/wombat.rendering-core").UniformFieldInfo["type"],
): void {
  if (typeof value === "number") {
    if (fieldType?.kind === "Int") {
      if (fieldType.signed) view.i32[offset >> 2] = value;
      else view.u32[offset >> 2] = value;
    } else {
      view.f32[offset >> 2] = value;
    }
    return;
  }
  if (value instanceof Float32Array) {
    view.f32.set(value, offset >> 2);
    return;
  }
  if (value instanceof Int32Array) {
    view.i32.set(value, offset >> 2);
    return;
  }
  if (value instanceof Uint32Array) {
    view.u32.set(value, offset >> 2);
    return;
  }
  if (value !== null && typeof value === "object" && "_data" in (value as object)) {
    const data = (value as { _data: unknown })._data;
    if (data instanceof Float32Array) { view.f32.set(data, offset >> 2); return; }
    if (data instanceof Int32Array)   { view.i32.set(data, offset >> 2); return; }
    if (data instanceof Uint32Array)  { view.u32.set(data, offset >> 2); return; }
  }
  throw new Error(`prepareUniformBuffer: unsupported uniform value at offset ${offset}: ${String(value)}`);
}

function roundUp4(n: number): number {
  return (n + 3) & ~3;
}

export interface PrepareUniformBufferOptions {
  readonly label?: string;
}

export function prepareUniformBuffer(
  device: GPUDevice,
  layout: UniformBlockInfo,
  inputs: HashMap<string, aval<unknown>>,
  opts: PrepareUniformBufferOptions = {},
): AdaptiveResource<GPUBuffer> {
  return new UniformBufferResource(device, layout, inputs, opts.label);
}
