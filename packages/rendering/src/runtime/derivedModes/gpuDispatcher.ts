// Scene-level dispatcher for the v1 GPU-evaluated derived-mode rule
// (flip-cull-by-determinant; see ./gpuKernel.ts).
//
// Lifecycle
// =========
//
//   - Construct once per heap scene with the device.
//   - registerRo(drawId, modelRef, declaredCull): register an RO that
//     has the GPU flip-cull rule. modelRef = byte offset into the
//     arena where the RO's ModelTrafo mat4 lives.
//   - deregisterRo(drawId): unregister on removeDraw.
//   - markDirty(): caller hints that an input changed; the next
//     `dispatch()` will run the kernel.
//   - dispatch(arenaBuf, enc): if dirty, encode the compute pass into
//     `enc`. Always copies the output buffer into a staging buffer for
//     CPU readback.
//   - finish(): mapAsync the staging buffer and resolve with the
//     per-RO output array. The scene calls this AFTER submit and
//     before deciding rebuckets. The promise resolves on the next
//     microtask after the GPU completes the copy.
//
// The buffers grow lazily (pow2) as ROs are added.

import { GPU_FLIP_CULL_BY_DET_WGSL, CULL_TO_U32 } from "./gpuKernel.js";
import type { CullMode } from "../pipelineCache/index.js";

const NO_RULE = 0xFFFFFFFF;

interface RoEntry {
  /** Byte offset of the RO's ModelTrafo mat4 in the arena. */
  modelRef: number;
  /** SG-declared cull mode, encoded as u32 (0=none/1=front/2=back). */
  declaredU32: number;
}

const WG_SIZE = 64;
const POW2 = (n: number): number => { let p = 1; while (p < n) p <<= 1; return Math.max(64, p); };

export class GpuDerivedModesScene {
  readonly device: GPUDevice;

  private readonly entries: (RoEntry | undefined)[] = [];
  private liveCount = 0;
  private dirty = false;

  // GPU resources — lazily allocated on first registerRo.
  private inputBuf:   GPUBuffer | null = null;
  private outputBuf:  GPUBuffer | null = null;
  private stagingBuf: GPUBuffer | null = null;
  private paramsBuf:  GPUBuffer | null = null;
  private pipeline:   GPUComputePipeline | null = null;
  private bindGroup:  GPUBindGroup | null = null;
  private layout:     GPUBindGroupLayout | null = null;
  /** Capacity in number of u32 entries (= numROs the buffers can hold). */
  private capacity = 0;
  /** Last-known arena buffer, used to detect bind-group invalidation. */
  private arenaBuf: GPUBuffer | null = null;
  /** Last-known declared (for kernel uniform). Common case: all RO
   *  rules share the same declared (= the SG scope's value). */
  private declared: number = CULL_TO_U32.back;
  /** True while the staging buffer's mapAsync is in flight. We must
   *  not enqueue a new copyBufferToBuffer into it until it's been
   *  unmapped (mapAsync resolved + unmap called). */
  private readbackInFlight = false;

  /** Per-RO last-known output. CPU mirror, populated by finish(). */
  readonly lastOutput: Uint32Array;

  constructor(device: GPUDevice) {
    this.device = device;
    this.lastOutput = new Uint32Array(0);
  }

  get registered(): number { return this.liveCount; }

  registerRo(drawId: number, modelRef: number, declared: CullMode): void {
    this.entries[drawId] = { modelRef, declaredU32: CULL_TO_U32[declared] };
    this.liveCount++;
    this.dirty = true;
    this.declared = CULL_TO_U32[declared];
  }

  deregisterRo(drawId: number): void {
    if (this.entries[drawId] === undefined) return;
    this.entries[drawId] = undefined;
    this.liveCount--;
    this.dirty = true;
  }

  markDirty(): void { this.dirty = true; }
  consumeDirty(): boolean { const was = this.dirty; this.dirty = false; return was; }

  private ensureResources(arenaBuf: GPUBuffer, numROs: number): void {
    const needCap = POW2(Math.max(1, numROs));
    if (this.outputBuf === null || needCap > this.capacity) {
      this.inputBuf  ?.destroy();
      this.outputBuf ?.destroy();
      this.stagingBuf?.destroy();
      const bytes = needCap * 4;
      this.inputBuf   = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: "gpuDerivedModes/in" });
      this.outputBuf  = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label: "gpuDerivedModes/out" });
      this.stagingBuf = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: "gpuDerivedModes/staging" });
      (this as { lastOutput: Uint32Array }).lastOutput = new Uint32Array(needCap);
      this.capacity = needCap;
      this.bindGroup = null;
    }
    if (this.paramsBuf === null) {
      this.paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "gpuDerivedModes/params" });
    }
    if (this.layout === null) {
      this.layout = this.device.createBindGroupLayout({
        label: "gpuDerivedModes/bgl",
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
    }
    if (this.pipeline === null) {
      const module = this.device.createShaderModule({ code: GPU_FLIP_CULL_BY_DET_WGSL, label: "gpuDerivedModes/module" });
      this.pipeline = this.device.createComputePipeline({
        label: "gpuDerivedModes/pipeline",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
        compute: { module, entryPoint: "evaluate" },
      });
    }
    if (this.bindGroup === null || this.arenaBuf !== arenaBuf) {
      this.bindGroup = this.device.createBindGroup({
        label: "gpuDerivedModes/bindGroup",
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: arenaBuf } },
          { binding: 1, resource: { buffer: this.inputBuf!  } },
          { binding: 2, resource: { buffer: this.outputBuf! } },
          { binding: 3, resource: { buffer: this.paramsBuf! } },
        ],
      });
      this.arenaBuf = arenaBuf;
    }
  }

  /** Upload current per-RO input refs + params; called when dirty. */
  private uploadInputs(numROs: number): void {
    const refs = new Uint32Array(this.capacity).fill(NO_RULE);
    for (let i = 0; i < numROs; i++) {
      const e = this.entries[i];
      if (e !== undefined) refs[i] = e.modelRef;
    }
    this.device.queue.writeBuffer(this.inputBuf!, 0, refs.buffer, refs.byteOffset, refs.byteLength);
    const params = new Uint32Array([numROs, this.declared, 0, 0]);
    this.device.queue.writeBuffer(this.paramsBuf!, 0, params.buffer, params.byteOffset, params.byteLength);
  }

  /**
   * Encode the compute dispatch + the staging copy into `enc`. Caller
   * is responsible for `device.queue.submit([enc.finish()])`. Returns
   * the number of ROs the kernel was dispatched against.
   */
  dispatch(arenaBuf: GPUBuffer, numROs: number, enc: GPUCommandEncoder): number {
    this.ensureResources(arenaBuf, numROs);
    // If the previous frame's readback hasn't resolved yet, we can
    // still dispatch the compute (writes to outputBuf), but we must
    // skip the staging copy — otherwise WebGPU rejects the submit
    // with "buffer used in submit while mapped". Cheap to skip: the
    // next dispatch will re-copy when staging is unmapped.
    this.uploadInputs(numROs);
    const pass = enc.beginComputePass({ label: "gpuDerivedModes/pass" });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, this.bindGroup!);
    pass.dispatchWorkgroups(Math.ceil(numROs / WG_SIZE));
    pass.end();
    if (!this.readbackInFlight) {
      const bytes = numROs * 4;
      enc.copyBufferToBuffer(this.outputBuf!, 0, this.stagingBuf!, 0, bytes);
      this.readbackInFlight = true;
    }
    return numROs;
  }

  /**
   * mapAsync the staging buffer + copy values into `lastOutput`.
   * Resolves on the GPU finishing the copy. Caller diffs lastOutput
   * against its own cached values to find changed ROs.
   */
  async finish(numROs: number): Promise<void> {
    if (this.stagingBuf === null || numROs === 0) return;
    if (!this.readbackInFlight) return; // dispatch skipped the copy; nothing to read
    const bytes = numROs * 4;
    try {
      await this.stagingBuf.mapAsync(GPUMapMode.READ, 0, bytes);
      const view = new Uint32Array(this.stagingBuf.getMappedRange(0, bytes).slice(0));
      this.stagingBuf.unmap();
      (this as { lastOutput: Uint32Array }).lastOutput.set(view, 0);
    } finally {
      this.readbackInFlight = false;
    }
  }

  dispose(): void {
    this.inputBuf  ?.destroy();
    this.outputBuf ?.destroy();
    this.stagingBuf?.destroy();
    this.paramsBuf ?.destroy();
  }
}
