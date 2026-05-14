// Phase 5c.3 — per-bucket partition dispatcher.
//
// Owns the master record buffer + per-slot atomic counters + the
// compute pipelines. One instance per GPU-routed bucket. The WGSL
// kernel is codegen'd by `emitPartitionKernel(...)` upstream (in
// heapScene) and handed to the dispatcher as a string — declared
// values are baked in as Consts, so a `declared`-aval mark
// triggers `rebuildKernel(newWgsl)` to swap to a freshly-codegen'd
// kernel for the new declared.
//
// `dispatch(enc)` encodes:
//   1. clear pass (zero every slot's atomic counter)
//   2. partition pass (one thread per record; evaluates the per-
//      axis rule, scatters into the matching slot via the kernel's
//      embedded SLOT_LOOKUP table)
//
// After dispatch the existing scan kernel runs per slot, reading the
// slot's drawTable that the partition just populated.

import { partitionRecordU32, partitionRecordBytes, PARTITION_RECORD_PREFIX_U32 } from "./partitionKernelLayout.js";

const WG_SIZE = 64;
const POW2 = (n: number): number => { let p = 1; while (p < n) p <<= 1; return Math.max(64, p); };

export interface PartitionSceneSpec {
  /** Total slot count = `resolved.length` (for the single-axis v1). */
  readonly totalSlots: number;
  /** Externally-owned slot draw buffers (one per slot). */
  readonly slotDrawBufs: ReadonlyArray<GPUBuffer>;
  /** Initial record capacity. */
  readonly initialRecords?: number;
  /** Codegen'd kernel WGSL. Embeds `declared` as a Const + the
   *  SLOT_LOOKUP table. Swap via `rebuildKernel(...)` on declared mark. */
  readonly kernelWGSL: string;
  /** Number of per-record uniform refs (= bucket.uniformOrder.length).
   *  The master record's width is `6 + numUniforms` u32s. Grow with
   *  `growUniforms(...)` when a new rule reads a uniform not yet in
   *  the bucket's order. */
  readonly numUniforms: number;
}

export class GpuPartitionScene {
  readonly device: GPUDevice;
  readonly label: string;
  get totalSlots(): number { return this._totalSlots; }

  masterBuf: GPUBuffer;
  masterShadow: Uint32Array;
  numRecords = 0;
  capacity = 0;

  slotCountBufs: GPUBuffer[];
  slotDrawBufs: GPUBuffer[];
  // `totalSlots` is mutable — `growSlots(...)` extends both the
  // dispatcher and the kernel layout to accommodate new pipeline
  // slots that appear when a new rule is registered on the bucket.
  private _totalSlots: number;
  // Per-bucket master record width = 6 + numUniforms. Mutable —
  // `growUniforms(...)` widens existing records in-place so a rule
  // reading a new uniform can land without rebuilding the master.
  private _numUniforms: number;
  get numUniforms(): number { return this._numUniforms; }
  get recordU32(): number { return partitionRecordU32(this._numUniforms); }
  get recordBytes(): number { return partitionRecordBytes(this._numUniforms); }

  /** params: [numRecords] — single u32 (padded to 16 bytes for WGSL
   *  uniform alignment). `declared` is no longer carried here; the
   *  kernel WGSL bakes it as a Const at codegen time. */
  paramsBuf: GPUBuffer;

  private kernelWGSL: string;

  private bindGroup: GPUBindGroup | null = null;
  private layout: GPUBindGroupLayout | null = null;
  private clearPipeline: GPUComputePipeline | null = null;
  private partitionPipeline: GPUComputePipeline | null = null;

  constructor(device: GPUDevice, label: string, spec: PartitionSceneSpec) {
    if (spec.slotDrawBufs.length !== spec.totalSlots) {
      throw new Error(
        `GpuPartitionScene: slotDrawBufs.length (${spec.slotDrawBufs.length}) != totalSlots (${spec.totalSlots})`,
      );
    }
    this.device = device;
    this.label = label;
    this._totalSlots = spec.totalSlots;
    this._numUniforms = spec.numUniforms;
    this.capacity = POW2(spec.initialRecords ?? 16);
    const bytes = this.capacity * partitionRecordBytes(this._numUniforms);
    this.masterBuf = device.createBuffer({
      label: `${label}/master`, size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.masterShadow = new Uint32Array(bytes / 4);
    this.slotCountBufs = new Array(spec.totalSlots).fill(0).map((_, i) =>
      device.createBuffer({
        label: `${label}/slot${i}Count`, size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }),
    );
    this.slotDrawBufs = [...spec.slotDrawBufs];
    this.paramsBuf = device.createBuffer({
      label: `${label}/params`, size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.kernelWGSL = spec.kernelWGSL;
  }

  /** Swap to a freshly-codegen'd kernel WGSL string. Triggered when
   *  `declared` (baked as a Const in the kernel) changes — the
   *  pipelines + bind group are reset so the next dispatch
   *  recompiles. The master records, slot count buffers and slot
   *  draw buffers are kept. */
  rebuildKernel(kernelWGSL: string): void {
    this.kernelWGSL = kernelWGSL;
    this.clearPipeline = null;
    this.partitionPipeline = null;
    // layout shape is invariant across declared values (slot count
    // is fixed for a given bucket); keep it.
  }

  rebindSlotDrawBufs(slotDraws: ReadonlyArray<GPUBuffer>): void {
    if (slotDraws.length !== this._totalSlots) {
      throw new Error(`rebindSlotDrawBufs: expected ${this._totalSlots} buffers, got ${slotDraws.length}`);
    }
    this.slotDrawBufs = [...slotDraws];
    this.bindGroup = null;
  }

  /** Grow the bucket's slot count when a new rule registers and the
   *  union of its outputs introduces fresh slots. Allocates the
   *  additional atomic-count buffers and discards the bind-group
   *  layout (different N = different binding shape). The caller
   *  must follow up with `rebuildKernel(...)` so the new kernel
   *  knows about the new slot count. */
  growSlots(newCount: number, newSlotDraws: ReadonlyArray<GPUBuffer>): void {
    if (newCount < this._totalSlots) {
      throw new Error(`growSlots: cannot shrink (have ${this._totalSlots}, asked for ${newCount})`);
    }
    if (newSlotDraws.length !== newCount) {
      throw new Error(`growSlots: slotDraws.length ${newSlotDraws.length} != newCount ${newCount}`);
    }
    for (let i = this._totalSlots; i < newCount; i++) {
      this.slotCountBufs.push(this.device.createBuffer({
        label: `${this.label}/slot${i}Count`, size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }));
    }
    this._totalSlots = newCount;
    this.slotDrawBufs = [...newSlotDraws];
    this.layout = null;
    this.clearPipeline = null;
    this.partitionPipeline = null;
    this.bindGroup = null;
  }

  private ensurePipelines(): void {
    if (this.layout === null) {
      const entries: GPUBindGroupLayoutEntry[] = [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // arena
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // master
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },            // params
      ];
      let nb = 3;
      for (let i = 0; i < this.totalSlots; i++) {
        entries.push({ binding: nb + i, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } });
      }
      nb += this.totalSlots;
      for (let i = 0; i < this.totalSlots; i++) {
        entries.push({ binding: nb + i, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } });
      }
      this.layout = this.device.createBindGroupLayout({ label: `${this.label}/bgl`, entries });
    }
    if (this.clearPipeline === null || this.partitionPipeline === null) {
      const module = this.device.createShaderModule({
        code: this.kernelWGSL,
        label: `${this.label}/module`,
      });
      const plLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.layout!] });
      this.clearPipeline = this.device.createComputePipeline({
        label: `${this.label}/clear`, layout: plLayout,
        compute: { module, entryPoint: "clear" },
      });
      this.partitionPipeline = this.device.createComputePipeline({
        label: `${this.label}/partition`, layout: plLayout,
        compute: { module, entryPoint: "partitionRecords" },
      });
    }
  }

  private ensureBindGroup(arenaBuf: GPUBuffer): void {
    this.ensurePipelines();
    if (this.bindGroup === null) {
      const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: arenaBuf       } },
        { binding: 1, resource: { buffer: this.masterBuf } },
        { binding: 2, resource: { buffer: this.paramsBuf } },
      ];
      let nb = 3;
      for (let i = 0; i < this.totalSlots; i++) {
        entries.push({ binding: nb + i, resource: { buffer: this.slotCountBufs[i]! } });
      }
      nb += this.totalSlots;
      for (let i = 0; i < this.totalSlots; i++) {
        entries.push({ binding: nb + i, resource: { buffer: this.slotDrawBufs[i]! } });
      }
      this.bindGroup = this.device.createBindGroup({
        label: `${this.label}/bg`,
        layout: this.layout!,
        entries,
      });
    }
  }

  appendRecord(
    drawIdx: number,
    indexStart: number,
    indexCount: number,
    instanceCount: number,
    comboId: number,
    uniformRefs: ReadonlyArray<number>,
  ): number {
    if (uniformRefs.length !== this._numUniforms) {
      throw new Error(`GpuPartitionScene/appendRecord: expected ${this._numUniforms} uniform refs, got ${uniformRefs.length}`);
    }
    if (this.numRecords >= this.capacity) this.grow();
    const i = this.numRecords;
    const ru32 = this.recordU32;
    const o = i * ru32;
    this.masterShadow[o + 0] = 0;
    this.masterShadow[o + 1] = drawIdx;
    this.masterShadow[o + 2] = indexStart;
    this.masterShadow[o + 3] = indexCount;
    this.masterShadow[o + 4] = instanceCount;
    this.masterShadow[o + 5] = comboId >>> 0;
    for (let k = 0; k < this._numUniforms; k++) {
      this.masterShadow[o + PARTITION_RECORD_PREFIX_U32 + k] = (uniformRefs[k] ?? 0) >>> 0;
    }
    this.numRecords = i + 1;
    return i;
  }

  /** Update an existing record's comboId. Used when an RO already
   *  in the master is re-classified (rare; only when rule rebucket
   *  fires on aval marks). */
  setRecordComboId(recordIdx: number, comboId: number): void {
    if (recordIdx < 0 || recordIdx >= this.numRecords) return;
    this.masterShadow[recordIdx * this.recordU32 + 5] = comboId >>> 0;
  }

  /** Update an existing record's uniform ref by uniform index. */
  setRecordUniformRef(recordIdx: number, uniformIdx: number, ref: number): void {
    if (recordIdx < 0 || recordIdx >= this.numRecords) return;
    if (uniformIdx < 0 || uniformIdx >= this._numUniforms) return;
    this.masterShadow[recordIdx * this.recordU32 + PARTITION_RECORD_PREFIX_U32 + uniformIdx] = ref >>> 0;
  }

  removeRecord(recordIdx: number): number {
    if (recordIdx < 0 || recordIdx >= this.numRecords) return -1;
    const last = this.numRecords - 1;
    const ru32 = this.recordU32;
    if (recordIdx !== last) {
      const dst = recordIdx * ru32;
      const src = last * ru32;
      for (let k = 0; k < ru32; k++) {
        this.masterShadow[dst + k] = this.masterShadow[src + k]!;
      }
    }
    this.numRecords = last;
    return recordIdx !== last ? last : -1;
  }

  /** Widen the per-record uniform tail from `_numUniforms` to
   *  `newNumUniforms`. Existing records are restrided in-place; the
   *  new tail slots are zero-filled (rules added later that read
   *  the new uniform will only do so under their own combo, where
   *  records carry real refs). The kernel WGSL must be rebuilt
   *  separately (`rebuildKernel`). */
  growUniforms(newNumUniforms: number): void {
    if (newNumUniforms <= this._numUniforms) return;
    const oldU32 = partitionRecordU32(this._numUniforms);
    const newU32 = partitionRecordU32(newNumUniforms);
    const newBytes = this.capacity * newU32 * 4;
    const grown = new Uint32Array(newBytes / 4);
    // Restride existing records into the wider layout.
    for (let i = 0; i < this.numRecords; i++) {
      const oldOff = i * oldU32;
      const newOff = i * newU32;
      for (let k = 0; k < oldU32; k++) grown[newOff + k] = this.masterShadow[oldOff + k]!;
      // Trailing newU32-oldU32 slots stay 0.
    }
    this.masterShadow = grown;
    this.masterBuf.destroy();
    this.masterBuf = this.device.createBuffer({
      label: `${this.label}/master`, size: newBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._numUniforms = newNumUniforms;
    this.bindGroup = null;
  }

  private grow(): void {
    const newCap = POW2(this.numRecords + 1);
    const bytes = newCap * this.recordBytes;
    const grown = new Uint32Array(bytes / 4);
    grown.set(this.masterShadow);
    this.masterShadow = grown;
    this.masterBuf.destroy();
    this.masterBuf = this.device.createBuffer({
      label: `${this.label}/master`, size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.capacity = newCap;
    this.bindGroup = null;
  }

  flush(): void {
    if (this.numRecords === 0) return;
    const bytes = this.numRecords * this.recordBytes;
    this.device.queue.writeBuffer(
      this.masterBuf, 0,
      this.masterShadow.buffer, this.masterShadow.byteOffset, bytes,
    );
  }

  /** Encode clear + partition into `enc`. */
  dispatch(arenaBuf: GPUBuffer, enc: GPUCommandEncoder): void {
    this.ensureBindGroup(arenaBuf);
    this.device.queue.writeBuffer(
      this.paramsBuf, 0,
      new Uint32Array([this.numRecords, 0, 0, 0]).buffer, 0, 16,
    );
    const pass = enc.beginComputePass({ label: `${this.label}/pass` });
    pass.setBindGroup(0, this.bindGroup!);
    pass.setPipeline(this.clearPipeline!);
    pass.dispatchWorkgroups(1);
    if (this.numRecords > 0) {
      pass.setPipeline(this.partitionPipeline!);
      pass.dispatchWorkgroups(Math.ceil(this.numRecords / WG_SIZE));
    }
    pass.end();
  }

  dispose(): void {
    this.masterBuf.destroy();
    for (const b of this.slotCountBufs) b.destroy();
    this.paramsBuf.destroy();
  }
}
