// §7 v2 — uber-kernel dispatcher (GPU plumbing).
//
// Owns: the GPU mirror of the CPU records buffer, a tiny `Count` uniform, and a
// recompilable compute pipeline. The kernel WGSL is regenerated whenever the rule
// registry gains a new rule (`registry.version`) or the record stride grows
// (`records.layoutVersion`). The records GPU buffer is re-uploaded whenever the CPU
// records buffer mutates (`records.generation`) — records change only on RO add/remove,
// never per frame. One `dispatchWorkgroups` per frame, no levels, no barriers.
// See docs/derived-uniforms-extensible.md § "Dispatcher".

import type { DerivedUniformRegistry } from "./registry.js";
import type { RecordsBuffer } from "./records.js";
import { buildUberKernel } from "./codegen.js";
import { DF32_MAT4_BYTES, type SlotIndex } from "./slots.js";

/** Upload the changed-constituent value range to the constituents GPU buffer. */
export function uploadConstituentsRange(
  device: GPUDevice,
  buf: GPUBuffer,
  mirror: Float32Array,
  dirty: ReadonlySet<SlotIndex>,
): void {
  let minSlot = Infinity;
  let maxSlot = -1;
  for (const s of dirty) {
    if (s < minSlot) minSlot = s;
    if (s > maxSlot) maxSlot = s;
  }
  if (maxSlot < 0) return;
  const startByte = minSlot * DF32_MAT4_BYTES;
  const endByte = (maxSlot + 1) * DF32_MAT4_BYTES;
  device.queue.writeBuffer(buf, startByte, mirror.buffer, mirror.byteOffset + startByte, endByte - startByte);
}

/** Live getters for the GPU buffers the kernel binds (scene may grow/replace them). */
export interface DerivedUniformsResources {
  /** `array<vec2<f32>>` — df32 trafo halves (the constituents heap). */
  readonly constituentsBuf: () => GPUBuffer;
  /** `array<f32>` — the main heap (drawHeaders); written by the kernel. */
  readonly mainHeapBuf: () => GPUBuffer;
}

function bglEntries(): GPUBindGroupLayoutEntry[] {
  return [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
  ];
}

/** GPU mirror of a CPU `RecordsBuffer` + the `Count` uniform. */
class RecordsGpu {
  private readonly device: GPUDevice;
  private buf: GPUBuffer | undefined;
  private bufCapBytes = 0;
  private uploadedGen = -1;
  readonly countBuf: GPUBuffer;
  private readonly countCpu = new Uint32Array(1);
  private uploadedCount = -1;

  constructor(device: GPUDevice) {
    this.device = device;
    this.countBuf = device.createBuffer({
      label: "derivedUniforms.recordCount",
      size: 16, // uniform min binding size
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /** Re-upload the packed record data if it changed since last sync, and refresh `Count`. */
  sync(records: RecordsBuffer): void {
    const data = records.data;
    const needBytes = Math.max(16, data.byteLength);
    if (this.buf === undefined || this.bufCapBytes < needBytes) {
      if (this.buf !== undefined) this.buf.destroy();
      let cap = Math.max(16, this.bufCapBytes || 16);
      while (cap < needBytes) cap *= 2;
      this.buf = this.device.createBuffer({
        label: "derivedUniforms.records",
        size: cap,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.bufCapBytes = cap;
      this.uploadedGen = -1; // force re-upload into the fresh buffer
    }
    if (records.generation !== this.uploadedGen && data.byteLength > 0) {
      this.device.queue.writeBuffer(this.buf!, 0, data.buffer, data.byteOffset, data.byteLength);
      this.uploadedGen = records.generation;
    }
    if (records.recordCount !== this.uploadedCount) {
      this.countCpu[0] = records.recordCount;
      this.device.queue.writeBuffer(this.countBuf, 0, this.countCpu.buffer, this.countCpu.byteOffset, 4);
      this.uploadedCount = records.recordCount;
    }
  }

  buffer(): GPUBuffer {
    if (this.buf === undefined) {
      this.buf = this.device.createBuffer({
        label: "derivedUniforms.records.stub",
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.bufCapBytes = 16;
    }
    return this.buf;
  }

  dispose(): void {
    this.buf?.destroy();
    this.buf = undefined;
    this.countBuf.destroy();
  }
}

/** Recompilable uber-kernel pipeline. Rebuilt on `(registry.version, strideU32)` change. */
class UberPipeline {
  private readonly device: GPUDevice;
  readonly bgl: GPUBindGroupLayout;
  private pipe: GPUComputePipeline | undefined;
  private builtVersion = -1;
  private builtStride = -1;

  constructor(device: GPUDevice) {
    this.device = device;
    this.bgl = device.createBindGroupLayout({ entries: bglEntries() });
  }

  pipeline(registry: DerivedUniformRegistry, strideU32: number): GPUComputePipeline | undefined {
    if (registry.size === 0) return undefined;
    if (this.pipe !== undefined && this.builtVersion === registry.version && this.builtStride === strideU32) {
      return this.pipe;
    }
    const { wgsl } = buildUberKernel(registry, strideU32);
    const module = this.device.createShaderModule({ code: wgsl, label: "derivedUniforms.uber" });
    this.pipe = this.device.createComputePipeline({
      label: "derivedUniforms.uber",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bgl] }),
      compute: { module, entryPoint: "main" },
    });
    this.builtVersion = registry.version;
    this.builtStride = strideU32;
    return this.pipe;
  }
}

export class DerivedUniformsDispatcher {
  private readonly device: GPUDevice;
  private readonly resources: DerivedUniformsResources;
  private readonly gpu: RecordsGpu;
  private readonly pipe: UberPipeline;
  private cachedBg: GPUBindGroup | undefined;
  private bgKey: [GPUBuffer | undefined, GPUBuffer | undefined, GPUBuffer | undefined] = [undefined, undefined, undefined];

  constructor(device: GPUDevice, resources: DerivedUniformsResources) {
    this.device = device;
    this.resources = resources;
    this.gpu = new RecordsGpu(device);
    this.pipe = new UberPipeline(device);
  }

  /** Open a compute pass and run the uber-kernel. No-ops if there are no records / no rules. */
  encode(enc: GPUCommandEncoder, registry: DerivedUniformRegistry, records: RecordsBuffer): boolean {
    if (records.recordCount === 0) return false;
    const pipeline = this.pipe.pipeline(registry, records.strideWords);
    if (pipeline === undefined) return false;
    this.gpu.sync(records);
    const pass = enc.beginComputePass({ label: "derivedUniforms.uber" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.bindGroup());
    pass.dispatchWorkgroups(Math.ceil(records.recordCount / 64));
    pass.end();
    return true;
  }

  dispose(): void {
    this.gpu.dispose();
  }

  private bindGroup(): GPUBindGroup {
    const consts = this.resources.constituentsBuf();
    const heap = this.resources.mainHeapBuf();
    const recs = this.gpu.buffer();
    if (this.cachedBg !== undefined && this.bgKey[0] === consts && this.bgKey[1] === heap && this.bgKey[2] === recs) {
      return this.cachedBg;
    }
    this.cachedBg = this.device.createBindGroup({
      label: "derivedUniforms.uber",
      layout: this.pipe.bgl,
      entries: [
        { binding: 0, resource: { buffer: consts } },
        { binding: 1, resource: { buffer: heap } },
        { binding: 2, resource: { buffer: recs } },
        { binding: 3, resource: { buffer: this.gpu.countBuf } },
      ],
    });
    this.bgKey = [consts, heap, recs];
    return this.cachedBg;
  }
}
