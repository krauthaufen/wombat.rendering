// Phase 5c.3 — per-bucket partition dispatcher.
//
// Owns the master record buffer + per-slot output drawTables + the
// compute pipelines. One instance per GPU-routed bucket. The
// `dispatch(enc)` method encodes:
//   1. clear pass (zero the per-slot atomic counters)
//   2. partition pass (one thread per record, scatter into slot
//      drawTables based on rule output)
//
// After dispatch the existing scan kernel runs per slot, reading the
// slot's drawTable that the partition just populated.

import {
  PARTITION_FLIP_CULL_BY_DET_WGSL,
  PARTITION_RECORD_BYTES,
  cullModeToU32,
} from "./partitionKernel.js";
import type { CullMode } from "../pipelineCache/index.js";

const WG_SIZE = 64;
const POW2 = (n: number): number => { let p = 1; while (p < n) p <<= 1; return Math.max(64, p); };

export class GpuPartitionScene {
  readonly device: GPUDevice;
  readonly label: string;

  /** Master record buffer (CPU populates via writeBuffer at addRO).
   *  Holds (firstEmit=0, drawIdx, indexStart, indexCount,
   *  instanceCount, modelRef) per record. */
  masterBuf: GPUBuffer;
  /** CPU shadow of master. */
  masterShadow: Uint32Array;
  /** Live record count in master. */
  numRecords = 0;
  /** Capacity in records. */
  capacity = 0;

  /** Per-slot atomic counters (u32) — cleared each frame by the
   *  clear pass, atomic-added by partition. Caller copies these
   *  into the slot's scan paramsBuf after dispatch. */
  slot0CountBuf: GPUBuffer;
  slot1CountBuf: GPUBuffer;
  /** Per-slot output drawTables — partition writes 5-u32 records
   *  here. Owned by the caller (typically the bucket's existing
   *  `slot.drawTableBuf.buffer`) so the scan + render bind groups
   *  read from the same buffer the partition writes. */
  slot0DrawBuf: GPUBuffer;
  slot1DrawBuf: GPUBuffer;

  /** Partition kernel uniform: numRecords + declaredCull. */
  paramsBuf: GPUBuffer;

  private bindGroup: GPUBindGroup | null = null;
  private layout: GPUBindGroupLayout | null = null;
  private clearPipeline: GPUComputePipeline | null = null;
  private partitionPipeline: GPUComputePipeline | null = null;

  constructor(
    device: GPUDevice,
    label: string,
    slot0Draw: GPUBuffer,
    slot1Draw: GPUBuffer,
    initialRecords = 16,
  ) {
    this.device = device;
    this.label = label;
    this.capacity = POW2(initialRecords);
    const bytes = this.capacity * PARTITION_RECORD_BYTES;
    this.masterBuf = device.createBuffer({
      label: `${label}/master`, size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.masterShadow = new Uint32Array(bytes / 4);
    this.slot0CountBuf = device.createBuffer({
      label: `${label}/slot0Count`, size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.slot1CountBuf = device.createBuffer({
      label: `${label}/slot1Count`, size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.slot0DrawBuf = slot0Draw;
    this.slot1DrawBuf = slot1Draw;
    this.paramsBuf = device.createBuffer({
      label: `${label}/params`, size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /** Caller-driven repoint of the external slot draw buffers.
   *  Used when the bucket's `slot.drawTableBuf` GrowBuffer
   *  reallocates and produces a fresh underlying GPUBuffer. */
  rebindSlotDrawBufs(slot0Draw: GPUBuffer, slot1Draw: GPUBuffer): void {
    this.slot0DrawBuf = slot0Draw;
    this.slot1DrawBuf = slot1Draw;
    this.bindGroup = null;
  }

  private ensurePipelines(): void {
    if (this.layout === null) {
      this.layout = this.device.createBindGroupLayout({
        label: `${this.label}/bgl`,
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // arena
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // master
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },            // params
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // slot0Count
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // slot1Count
          { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // slot0Draw
          { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // slot1Draw
        ],
      });
    }
    if (this.clearPipeline === null || this.partitionPipeline === null) {
      const module = this.device.createShaderModule({
        code: PARTITION_FLIP_CULL_BY_DET_WGSL,
        label: `${this.label}/module`,
      });
      const plLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.layout!] });
      this.clearPipeline = this.device.createComputePipeline({
        label: `${this.label}/clear`, layout: plLayout,
        compute: { module, entryPoint: "clear" },
      });
      this.partitionPipeline = this.device.createComputePipeline({
        label: `${this.label}/partition`, layout: plLayout,
        compute: { module, entryPoint: "partition" },
      });
    }
  }

  private ensureBindGroup(arenaBuf: GPUBuffer): void {
    this.ensurePipelines();
    if (this.bindGroup === null) {
      this.bindGroup = this.device.createBindGroup({
        label: `${this.label}/bg`,
        layout: this.layout!,
        entries: [
          { binding: 0, resource: { buffer: arenaBuf       } },
          { binding: 1, resource: { buffer: this.masterBuf } },
          { binding: 2, resource: { buffer: this.paramsBuf } },
          { binding: 3, resource: { buffer: this.slot0CountBuf } },
          { binding: 4, resource: { buffer: this.slot1CountBuf } },
          { binding: 5, resource: { buffer: this.slot0DrawBuf  } },
          { binding: 6, resource: { buffer: this.slot1DrawBuf  } },
        ],
      });
    }
  }

  /** Append a record into the master buffer. Called by CPU on addRO. */
  appendRecord(
    drawIdx: number,
    indexStart: number,
    indexCount: number,
    instanceCount: number,
    modelRef: number,
  ): number {
    if (this.numRecords >= this.capacity) {
      this.grow();
    }
    const i = this.numRecords;
    const o = i * PARTITION_RECORD_BYTES / 4;
    this.masterShadow[o + 0] = 0;             // firstEmit (GPU writes via scan)
    this.masterShadow[o + 1] = drawIdx;
    this.masterShadow[o + 2] = indexStart;
    this.masterShadow[o + 3] = indexCount;
    this.masterShadow[o + 4] = instanceCount;
    this.masterShadow[o + 5] = modelRef;
    this.numRecords = i + 1;
    return i;
  }

  /** Remove a record by swap-pop. Returns the localSlot of the
   *  moved tail record (or -1 if no move). */
  removeRecord(recordIdx: number): number {
    if (recordIdx < 0 || recordIdx >= this.numRecords) return -1;
    const last = this.numRecords - 1;
    if (recordIdx !== last) {
      const dst = recordIdx * PARTITION_RECORD_BYTES / 4;
      const src = last * PARTITION_RECORD_BYTES / 4;
      for (let k = 0; k < PARTITION_RECORD_U32_LOCAL; k++) {
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
    // Replace master buffer only — slot draw bufs are externally
    // owned (the bucket's GrowBuffer manages its own grow path
    // and calls rebindSlotDrawBufs when the underlying GPUBuffer
    // is replaced).
    this.masterBuf.destroy();
    this.masterBuf = this.device.createBuffer({
      label: `${this.label}/master`, size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.capacity = newCap;
    this.bindGroup = null; // force rebuild
  }

  /** Flush the CPU shadow to GPU master. */
  flush(): void {
    if (this.numRecords === 0) return;
    const bytes = this.numRecords * PARTITION_RECORD_BYTES;
    this.device.queue.writeBuffer(this.masterBuf, 0, this.masterShadow.buffer, this.masterShadow.byteOffset, bytes);
  }

  /** Encode clear + partition into `enc`. Caller submits the encoder. */
  dispatch(arenaBuf: GPUBuffer, declared: CullMode, enc: GPUCommandEncoder): void {
    this.ensureBindGroup(arenaBuf);
    this.device.queue.writeBuffer(
      this.paramsBuf, 0,
      new Uint32Array([this.numRecords, cullModeToU32(declared), 0, 0]).buffer, 0, 16,
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
    // slot{0,1}DrawBuf are externally owned — caller disposes.
    this.masterBuf.destroy();
    this.slot0CountBuf.destroy();
    this.slot1CountBuf.destroy();
    this.paramsBuf.destroy();
  }
}

const PARTITION_RECORD_U32_LOCAL = 6; // mirror of the export, avoids cyclic re-import
