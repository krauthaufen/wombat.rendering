// ComputeBatch — records many compute dispatches (plus buffer copies,
// writes and readbacks) into few command buffers.
//
// `PreparedComputeShader.dispatch` submits and waits per call, and a
// `ComputeInputBinding` owns ONE uniform buffer that `setUniform` +
// `queue.writeBuffer` overwrite — so encoding several dispatches with
// different uniform values into one submit would make all of them see
// the last values. A batch instead snapshots every dispatch's uniform
// blocks into its own 256-byte aligned slot of a per-batch uniform
// arena (uploaded with one `writeBuffer` per arena chunk at submit),
// so the usual mutable-binding pattern works unchanged:
//
//   const b = shader.createInputBinding().setBuffer("data", buf);
//   const batch = new ComputeBatch(device);
//   for (const step of steps) {
//     b.setUniform("step", step);
//     batch.dispatch(shader, b, [nx, ny, nz]);
//     if (step % 8 === 7) batch.submit();      // keep GPU jobs short
//   }
//   const bytes = await batch.read(buf);       // copy + submit + map
//
// Storage buffers / textures are resolved at `dispatch` time (later
// `setBuffer` calls don't affect recorded dispatches). Consecutive
// dispatches share one compute pass; WebGPU orders their storage
// accesses. Arena chunks are recycled once the GPU has finished the
// submit that used them.

import type { ComputeInputBinding, DispatchSize, PreparedComputeShader } from "./preparedComputeShader.js";
import { writeBufferRange, type HostData } from "./gpuResources.js";
import { BufferUsage } from "./webgpuFlags.js";

const CHUNK_BYTES = 64 * 1024;

interface ArenaChunk {
  readonly buffer: GPUBuffer;
  readonly shadow: Uint8Array<ArrayBuffer>;
  used: number;
}

export interface ComputeBatchOptions {
  readonly label?: string;
}

export class ComputeBatch {
  private _encoder: GPUCommandEncoder | undefined;
  private _pass: GPUComputePassEncoder | undefined;
  private _chunks: ArenaChunk[] = [];     // chunks of the open command buffer
  private readonly _free: ArenaChunk[] = [];
  private readonly _align: number;
  private _recorded = 0;
  private _dispatches = 0;
  private _submits = 0;
  private _disposed = false;

  constructor(readonly device: GPUDevice, private readonly opts: ComputeBatchOptions = {}) {
    this._align = device.limits.minUniformBufferOffsetAlignment ?? 256;
  }

  /** Dispatches recorded so far (all submits). */
  get dispatchCount(): number { return this._dispatches; }
  /** Command buffers submitted so far. */
  get submitCount(): number { return this._submits; }
  /** Commands recorded since the last submit. */
  get pendingCount(): number { return this._recorded; }

  private encoder(): GPUCommandEncoder {
    if (this._disposed) throw new Error("ComputeBatch: used after dispose");
    this._encoder ??= this.device.createCommandEncoder(
      this.opts.label !== undefined ? { label: this.opts.label } : {},
    );
    return this._encoder;
  }

  private endPass(): void {
    this._pass?.end();
    this._pass = undefined;
  }

  private uniformSlot(bytes: ArrayBuffer): GPUBufferBinding {
    const size = bytes.byteLength;
    if (size > CHUNK_BYTES) throw new Error(`ComputeBatch: uniform block of ${size} bytes exceeds ${CHUNK_BYTES}`);
    let chunk = this._chunks[this._chunks.length - 1];
    if (chunk === undefined || chunk.used + size > CHUNK_BYTES) {
      chunk = this._free.pop() ?? {
        buffer: this.device.createBuffer({
          size: CHUNK_BYTES,
          usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
          ...(this.opts.label !== undefined ? { label: `${this.opts.label}.uniforms` } : {}),
        }),
        shadow: new Uint8Array(CHUNK_BYTES),
        used: 0,
      };
      chunk.used = 0;
      this._chunks.push(chunk);
    }
    const offset = chunk.used;
    chunk.shadow.set(new Uint8Array(bytes), offset);
    chunk.used = Math.ceil((offset + size) / this._align) * this._align;
    return { buffer: chunk.buffer, offset, size };
  }

  /**
   * Record one dispatch of `shader` with the binding's CURRENT uniform
   * values and resources. `groups` is the workgroup count.
   */
  dispatch(shader: PreparedComputeShader, binding: ComputeInputBinding, groups: DispatchSize | readonly number[]): this {
    const g = Array.isArray(groups)
      ? { x: groups[0] ?? 1, y: groups[1] ?? 1, z: groups[2] ?? 1 }
      : groups as DispatchSize;
    const enc = this.encoder();
    const bindGroups = binding.buildBindGroups((ub) => this.uniformSlot(ub.bytes));
    if (this._pass === undefined) {
      this._pass = enc.beginComputePass(this.opts.label !== undefined ? { label: this.opts.label } : {});
    }
    this._pass.setPipeline(shader.pipeline);
    for (let i = 0; i < bindGroups.length; i++) this._pass.setBindGroup(i, bindGroups[i]!);
    this._pass.dispatchWorkgroups(g.x, g.y, g.z);
    this._recorded++;
    this._dispatches++;
    return this;
  }

  /** Record a buffer-to-buffer copy (sizes/offsets in bytes, multiples of 4). */
  copyBuffer(src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, size: number): this {
    const enc = this.encoder();
    this.endPass();
    enc.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size);
    this._recorded++;
    return this;
  }

  /** Record a clear of `size` bytes of `buffer` from `offset` to zero. */
  clearBuffer(buffer: GPUBuffer, offset = 0, size?: number): this {
    const enc = this.encoder();
    this.endPass();
    enc.clearBuffer(buffer, offset, size);
    this._recorded++;
    return this;
  }

  /**
   * Write host data to `buffer`, ordered after everything recorded so
   * far (submits the pending work first — `queue.writeBuffer` would
   * otherwise overtake it).
   */
  writeBuffer(buffer: GPUBuffer, byteOffset: number, data: HostData, srcByteOffset = 0, byteLength?: number): this {
    this.submit();
    writeBufferRange(this.device, buffer, byteOffset, data, srcByteOffset, byteLength);
    return this;
  }

  /** Submit everything recorded since the last submit (no-op when nothing is pending). */
  submit(): void {
    if (this._encoder === undefined) return;
    this.endPass();
    const chunks = this._chunks;
    this._chunks = [];
    for (const c of chunks) {
      if (c.used > 0) this.device.queue.writeBuffer(c.buffer, 0, c.shadow, 0, Math.min(c.used, CHUNK_BYTES));
    }
    this.device.queue.submit([this._encoder.finish()]);
    this._encoder = undefined;
    this._recorded = 0;
    this._submits++;
    if (chunks.length > 0) {
      this.device.queue.onSubmittedWorkDone().then(() => {
        if (this._disposed) for (const c of chunks) c.buffer.destroy();
        else this._free.push(...chunks);
      }, () => { /* device lost: buffers die with it */ });
    }
  }

  /**
   * Copy `byteLength` bytes of `buffer` (default: all from `byteOffset`)
   * into a staging buffer after the recorded work, submit, and resolve
   * to the bytes. `buffer` needs `COPY_SRC` usage.
   */
  async read(buffer: GPUBuffer, byteOffset = 0, byteLength?: number): Promise<ArrayBuffer> {
    const length = byteLength ?? buffer.size - byteOffset;
    const start = byteOffset & ~3;
    const end = Math.min(buffer.size, (byteOffset + length + 3) & ~3);
    const staging = this.device.createBuffer({
      size: Math.max(4, end - start),
      usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST,
      ...(this.opts.label !== undefined ? { label: `${this.opts.label}.read` } : {}),
    });
    try {
      this.copyBuffer(buffer, start, staging, 0, end - start);
      this.submit();
      await staging.mapAsync(GPUMapMode.READ);
      return staging.getMappedRange().slice(byteOffset - start, byteOffset - start + length);
    } finally {
      staging.destroy();
    }
  }

  /** Submit pending work and resolve once the GPU has finished everything submitted. */
  async flush(): Promise<void> {
    this.submit();
    await this.device.queue.onSubmittedWorkDone();
  }

  /** Free the uniform arena. Pending (unsubmitted) work is dropped. */
  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    this._pass = undefined;
    this._encoder = undefined;
    for (const c of this._free) c.buffer.destroy();
    for (const c of this._chunks) c.buffer.destroy();
    this._free.length = 0;
    this._chunks = [];
  }
}
