// §7 single-dispatch leaf dispatcher.
//
// Each bucket holds a static-ish records buffer (only edited on RO
// add/remove). Per frame the caller:
//   1. Drains dirty constituent avals and uploads the changed byte
//      range to the shared constituents buffer.
//   2. Builds a dirty bitmask (one bit per constituent slot) and
//      uploads it to the shared dirty buffer.
//   3. For each bucket, calls `dispatcher.encode(enc)` — one
//      compute pass with one dispatch per bucket. The kernel
//      early-returns on records whose inputs are all clean.
//
// No worklist building, no atomicAdd compaction, no per-frame record
// encoding. The records buffer changes only when ROs come and go.

import {
  ConstituentSlots, DF32_MAT4_BYTES,
  type DerivationRecord, type SlotIndex,
} from "./slots.js";
import {
  RECORD_STRIDE_BYTES, RECORD_STRIDE_U32, RECORD_FIELD,
  DERIVED_UBER_KERNEL_WGSL,
} from "./uberKernel.wgsl.js";

/** Live getters for the GPU resources the dispatcher needs. The scene
 *  may replace these buffers in-place when capacity grows; the
 *  dispatcher rebinds each frame. */
export interface DerivedUniformsResources {
  readonly constituentsBuf: () => GPUBuffer;
  readonly mainHeapBuf:     () => GPUBuffer;
}

/** Per-bucket records buffer. Records are added/removed when ROs
 *  enter/leave the bucket. Uses swap-remove to keep the buffer dense.
 *  CPU mirror is kept up-to-date; the dirty range is uploaded once
 *  per `flush()`. */
export class RecordsBuffer {
  private readonly device: GPUDevice;
  private cpu: Uint32Array;
  private cap = 0;             // record slots (capacity, in records)
  private count = 0;           // live records
  private nextId = 0;
  private readonly slotById = new Map<number, number>();
  private readonly idBySlot: number[] = [];

  /** Dirty CPU-mirror byte range that must be uploaded next flush. */
  private dirtyMinByte = Infinity;
  private dirtyMaxByte = -1;

  private gpu: GPUBuffer | undefined;
  private gpuCap = 0;          // record slots in `gpu`

  constructor(device: GPUDevice, initialCapacity = 64) {
    this.device = device;
    this.cap = initialCapacity;
    this.cpu = new Uint32Array(this.cap * RECORD_STRIDE_U32);
  }

  /** Add a record. Returns a stable handle for later removal. */
  add(rec: DerivationRecord): number {
    if (this.count === this.cap) this.growCpu();
    const slot = this.count++;
    const id = this.nextId++;
    this.slotById.set(id, slot);
    this.idBySlot[slot] = id;
    this.writeAt(slot, rec);
    return id;
  }

  /** Swap-remove. The last record moves into the freed slot. */
  remove(id: number): void {
    const slot = this.slotById.get(id);
    if (slot === undefined) {
      throw new Error("RecordsBuffer.remove: unknown id");
    }
    const last = this.count - 1;
    if (slot !== last) {
      // Move last → slot.
      const srcOff = last * RECORD_STRIDE_U32;
      const dstOff = slot * RECORD_STRIDE_U32;
      for (let i = 0; i < RECORD_STRIDE_U32; i++) {
        this.cpu[dstOff + i] = this.cpu[srcOff + i]!;
      }
      const movedId = this.idBySlot[last]!;
      this.slotById.set(movedId, slot);
      this.idBySlot[slot] = movedId;
      this.markDirty(slot, slot + 1);
    }
    this.slotById.delete(id);
    this.idBySlot.length = last;
    this.count = last;
  }

  /** Live record count. */
  get recordCount(): number { return this.count; }

  /** Ensure GPU buffer matches cap and upload dirty CPU range. Call
   *  before `encode`. */
  flush(): void {
    if (this.gpu === undefined || this.gpuCap < this.cap) {
      if (this.gpu !== undefined) this.gpu.destroy();
      this.gpu = this.device.createBuffer({
        label: "derivedUniforms.records",
        size:  this.cap * RECORD_STRIDE_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.gpuCap = this.cap;
      // Whole CPU mirror needs re-upload.
      this.dirtyMinByte = 0;
      this.dirtyMaxByte = this.count * RECORD_STRIDE_BYTES;
    }
    if (this.dirtyMaxByte > this.dirtyMinByte) {
      const start = this.dirtyMinByte;
      const end   = Math.min(this.dirtyMaxByte, this.count * RECORD_STRIDE_BYTES);
      if (end > start) {
        this.device.queue.writeBuffer(
          this.gpu!, start,
          this.cpu.buffer,
          this.cpu.byteOffset + start,
          end - start,
        );
      }
      this.dirtyMinByte = Infinity;
      this.dirtyMaxByte = -1;
    }
  }

  buffer(): GPUBuffer {
    if (this.gpu === undefined) {
      // Empty bucket — make a tiny stub so binding always has something.
      this.gpu = this.device.createBuffer({
        label: "derivedUniforms.records.stub",
        size:  RECORD_STRIDE_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.gpuCap = 1;
    }
    return this.gpu;
  }

  dispose(): void {
    if (this.gpu !== undefined) {
      this.gpu.destroy();
      this.gpu = undefined;
    }
  }

  // ── Internals ─────────────────────────────────────────────────────

  private writeAt(slot: number, rec: DerivationRecord): void {
    const off = slot * RECORD_STRIDE_U32;
    this.cpu[off + RECORD_FIELD.recipe_id] = rec.recipe;
    this.cpu[off + RECORD_FIELD.in0]       = rec.in0;
    this.cpu[off + RECORD_FIELD.in1]       = rec.in1;
    this.cpu[off + RECORD_FIELD.in2]       = rec.in2;
    this.cpu[off + RECORD_FIELD.out_byte]  = rec.outByte;
    this.markDirty(slot, slot + 1);
  }

  private markDirty(startSlot: number, endSlot: number): void {
    const startByte = startSlot * RECORD_STRIDE_BYTES;
    const endByte   = endSlot   * RECORD_STRIDE_BYTES;
    if (startByte < this.dirtyMinByte) this.dirtyMinByte = startByte;
    if (endByte   > this.dirtyMaxByte) this.dirtyMaxByte = endByte;
  }

  private growCpu(): void {
    const newCap = this.cap * 2;
    const grown = new Uint32Array(newCap * RECORD_STRIDE_U32);
    grown.set(this.cpu);
    this.cpu = grown;
    this.cap = newCap;
  }
}

/** One compute pipeline shared across buckets — created once per scene. */
export class DerivedUniformsPipeline {
  readonly pipeline: GPUComputePipeline;
  readonly bgl:      GPUBindGroupLayout;

  constructor(device: GPUDevice) {
    const module = device.createShaderModule({ code: DERIVED_UBER_KERNEL_WGSL });
    this.bgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    this.pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bgl] }),
      compute: { module, entryPoint: "main" },
    });
  }
}

/** Per-bucket dispatcher. One bind group cached, rebuilt only when a
 *  bound buffer reference changes (constituents or main heap grew, or
 *  records buffer was re-created). */
export class DerivedUniformsDispatcher {
  private readonly device: GPUDevice;
  private readonly pipe:   DerivedUniformsPipeline;
  private readonly resources: DerivedUniformsResources;
  readonly records: RecordsBuffer;

  private cachedBg: GPUBindGroup | undefined;
  private bgKeyConsts: GPUBuffer | undefined;
  private bgKeyHeap:   GPUBuffer | undefined;
  private bgKeyRecs:   GPUBuffer | undefined;

  /** Tiny uniform holding the live record count — gates threads past
   *  the live range from reading stale swap-removed records. */
  private readonly countBuf: GPUBuffer;
  private readonly countCpu = new Uint32Array(1);
  private lastCountUploaded = -1;

  constructor(
    device: GPUDevice,
    pipe:   DerivedUniformsPipeline,
    resources: DerivedUniformsResources,
    initialRecordCapacity = 64,
  ) {
    this.device    = device;
    this.pipe      = pipe;
    this.resources = resources;
    this.records   = new RecordsBuffer(device, initialRecordCapacity);
    this.countBuf = device.createBuffer({
      label: "derivedUniforms.recordCount",
      size:  16,                             // uniform min size
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /** Dispatch into an open compute pass. Caller is responsible for
   *  beginning/ending the pass and setting the pipeline once across
   *  all dispatchers (the pipeline is shared). */
  dispatchInto(pass: GPUComputePassEncoder): boolean {
    const count = this.records.recordCount;
    if (count === 0) return false;
    this.records.flush();
    if (count !== this.lastCountUploaded) {
      this.countCpu[0] = count;
      this.device.queue.writeBuffer(
        this.countBuf, 0, this.countCpu.buffer, this.countCpu.byteOffset, 4,
      );
      this.lastCountUploaded = count;
    }
    pass.setBindGroup(0, this.bindGroup());
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    return true;
  }

  /** Convenience: open/close a pass and dispatch. Use only when this
   *  is the sole dispatcher in the pass; otherwise use `dispatchInto`
   *  inside a caller-managed pass. */
  encode(enc: GPUCommandEncoder): boolean {
    if (this.records.recordCount === 0) return false;
    const pass = enc.beginComputePass({ label: "derivedUniforms.uber" });
    pass.setPipeline(this.pipe.pipeline);
    const ran = this.dispatchInto(pass);
    pass.end();
    return ran;
  }

  dispose(): void {
    this.records.dispose();
    this.countBuf.destroy();
  }

  private bindGroup(): GPUBindGroup {
    const consts = this.resources.constituentsBuf();
    const heap   = this.resources.mainHeapBuf();
    const recs   = this.records.buffer();
    if (
      this.cachedBg !== undefined &&
      this.bgKeyConsts === consts && this.bgKeyHeap === heap &&
      this.bgKeyRecs === recs
    ) {
      return this.cachedBg;
    }
    this.cachedBg = this.device.createBindGroup({
      layout: this.pipe.bgl,
      entries: [
        { binding: 0, resource: { buffer: consts } },
        { binding: 1, resource: { buffer: heap } },
        { binding: 2, resource: { buffer: recs } },
        { binding: 3, resource: { buffer: this.countBuf } },
      ],
    });
    this.bgKeyConsts = consts; this.bgKeyHeap = heap;
    this.bgKeyRecs = recs;
    return this.cachedBg;
  }
}

/** Scene-side helper: O(changed) maintenance of a GPU dirty bitmask.
 *
 *  Steady state (clean frame): zero work. The CPU mirror reflects the
 *  GPU buffer, so we only ever touch bits that actually flip.
 *
 *  Each `upload(dirty)` call:
 *    1. Clears the bits set last frame (O(prev_dirty)).
 *    2. Sets the bits dirty this frame (O(dirty)).
 *    3. Writes only the affected u32 range to the GPU (one writeBuffer).
 *  If both sets are empty, returns immediately. */
export class DirtyMaskUploader {
  private readonly device: GPUDevice;
  private cpu: Uint32Array;
  private gpu: GPUBuffer | undefined;
  private gpuCapU32 = 0;
  /** Bits set on the previous upload — must be cleared next time. */
  private prevDirty: SlotIndex[] = [];

  constructor(device: GPUDevice, initialU32 = 8) {
    this.device = device;
    this.cpu = new Uint32Array(Math.max(1, initialU32));
  }

  /** Resize CPU + GPU mirrors so they cover `slotCount` bits. */
  ensureCapacity(slotCount: number): void {
    const need = Math.max(1, Math.ceil(slotCount / 32));
    if (need > this.cpu.length) {
      let cap = Math.max(this.cpu.length, 1);
      while (cap < need) cap *= 2;
      const grown = new Uint32Array(cap);
      grown.set(this.cpu);
      this.cpu = grown;
    }
    if (this.gpu === undefined || this.gpuCapU32 < need) {
      if (this.gpu !== undefined) this.gpu.destroy();
      const cap = this.cpu.length;
      this.gpu = this.device.createBuffer({
        label: "derivedUniforms.dirtyMask",
        size:  cap * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.gpuCapU32 = cap;
      // GPU buffer is fresh-zero; CPU mirror should match. Wipe any
      // prevDirty (those bits live in the destroyed buffer).
      this.prevDirty.length = 0;
    }
  }

  /** Apply the new dirty set. O(prev_dirty + dirty), no O(slotCount). */
  upload(dirty: ReadonlySet<SlotIndex>): void {
    let minU32 =  Infinity;
    let maxU32 = -Infinity;
    for (const s of this.prevDirty) {
      const i = s >>> 5; const b = s & 31;
      this.cpu[i] = (this.cpu[i]! & ~(1 << b)) >>> 0;
      if (i < minU32) minU32 = i;
      if (i > maxU32) maxU32 = i;
    }
    this.prevDirty.length = 0;
    for (const s of dirty) {
      const i = s >>> 5; const b = s & 31;
      this.cpu[i] = (this.cpu[i]! | (1 << b)) >>> 0;
      if (i < minU32) minU32 = i;
      if (i > maxU32) maxU32 = i;
      this.prevDirty.push(s);
    }
    if (maxU32 < 0 || maxU32 === -Infinity) return;
    const startByte = minU32 * 4;
    const endByte   = (maxU32 + 1) * 4;
    this.device.queue.writeBuffer(
      this.gpu!, startByte,
      this.cpu.buffer, this.cpu.byteOffset + startByte,
      endByte - startByte,
    );
  }

  buffer(): GPUBuffer {
    if (this.gpu === undefined) this.ensureCapacity(1);
    return this.gpu!;
  }

  dispose(): void {
    if (this.gpu !== undefined) {
      this.gpu.destroy();
      this.gpu = undefined;
    }
  }
}

/** Upload the dirty constituent value range to the constituents GPU
 *  buffer. Helper used by the scene each frame after draining. */
export function uploadConstituentsRange(
  device: GPUDevice,
  buf:    GPUBuffer,
  mirror: Float32Array,
  dirty:  ReadonlySet<SlotIndex>,
): void {
  let minSlot = Infinity, maxSlot = -1;
  for (const s of dirty) {
    if (s < minSlot) minSlot = s;
    if (s > maxSlot) maxSlot = s;
  }
  if (maxSlot < 0) return;
  const startByte = minSlot * DF32_MAT4_BYTES;
  const endByte   = (maxSlot + 1) * DF32_MAT4_BYTES;
  device.queue.writeBuffer(
    buf, startByte,
    mirror.buffer, mirror.byteOffset + startByte,
    endByte - startByte,
  );
}
