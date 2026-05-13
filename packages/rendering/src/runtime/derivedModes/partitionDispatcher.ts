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

import { PARTITION_RECORD_BYTES, PARTITION_RECORD_U32 } from "./partitionKernelLayout.js";

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
}

export class GpuPartitionScene {
  readonly device: GPUDevice;
  readonly label: string;
  readonly totalSlots: number;

  masterBuf: GPUBuffer;
  masterShadow: Uint32Array;
  numRecords = 0;
  capacity = 0;

  readonly slotCountBufs: GPUBuffer[];
  slotDrawBufs: GPUBuffer[];

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
    this.totalSlots = spec.totalSlots;
    this.capacity = POW2(spec.initialRecords ?? 16);
    const bytes = this.capacity * PARTITION_RECORD_BYTES;
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
    if (slotDraws.length !== this.totalSlots) {
      throw new Error(`rebindSlotDrawBufs: expected ${this.totalSlots} buffers, got ${slotDraws.length}`);
    }
    this.slotDrawBufs = [...slotDraws];
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
    modelRef: number,
  ): number {
    if (this.numRecords >= this.capacity) this.grow();
    const i = this.numRecords;
    const o = i * PARTITION_RECORD_U32;
    this.masterShadow[o + 0] = 0;
    this.masterShadow[o + 1] = drawIdx;
    this.masterShadow[o + 2] = indexStart;
    this.masterShadow[o + 3] = indexCount;
    this.masterShadow[o + 4] = instanceCount;
    this.masterShadow[o + 5] = modelRef;
    this.numRecords = i + 1;
    return i;
  }

  removeRecord(recordIdx: number): number {
    if (recordIdx < 0 || recordIdx >= this.numRecords) return -1;
    const last = this.numRecords - 1;
    if (recordIdx !== last) {
      const dst = recordIdx * PARTITION_RECORD_U32;
      const src = last * PARTITION_RECORD_U32;
      for (let k = 0; k < PARTITION_RECORD_U32; k++) {
        this.masterShadow[dst + k] = this.masterShadow[src + k]!;
      }
    }
    this.numRecords = last;
    return recordIdx !== last ? last : -1;
  }

  private grow(): void {
    const newCap = POW2(this.numRecords + 1);
    const bytes = newCap * PARTITION_RECORD_BYTES;
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
    const bytes = this.numRecords * PARTITION_RECORD_BYTES;
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
