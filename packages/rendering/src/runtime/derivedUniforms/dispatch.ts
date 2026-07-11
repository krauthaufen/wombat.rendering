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

/** Upload the changed-constituent values to the constituents GPU buffer —
 *  one writeBuffer per RUN of nearby dirty slots, not one min..max span.
 *  The span version re-uploaded nearly the whole buffer for a handful of
 *  scattered edits (68k-object CAD bench: ~28 MB/frame for k = 10 random
 *  trafo edits — 10 GB/s of writeBuffer traffic, ~20 ms frames). */
export function uploadConstituentsRange(
  device: GPUDevice,
  buf: GPUBuffer,
  mirror: Float32Array,
  dirty: ReadonlySet<SlotIndex>,
): void {
  if (dirty.size === 0) return;
  const slots = [...dirty].sort((a, b) => a - b);
  // merge runs separated by ≤32 clean slots (≤4 KB of clean bytes —
  // cheaper to upload than to issue another writeBuffer)
  const MERGE_GAP_SLOTS = 32;
  const emit = (s: number, e: number): void => {
    const startByte = s * DF32_MAT4_BYTES;
    const endByte = (e + 1) * DF32_MAT4_BYTES;
    device.queue.writeBuffer(
      buf, startByte,
      mirror.buffer, mirror.byteOffset + startByte,
      endByte - startByte,
    );
  };
  let runStart = slots[0]!;
  let runEnd = slots[0]!;
  for (let i = 1; i < slots.length; i++) {
    const s = slots[i]!;
    if (s - runEnd <= MERGE_GAP_SLOTS) runEnd = s;
    else { emit(runStart, runEnd); runStart = s; runEnd = s; }
  }
  emit(runStart, runEnd);
}

/** Live getters for the scene-wide GPU buffers the kernel binds. The
 *  main-heap binding is per-dispatch and supplied to `encodeChunk`,
 *  since one DerivedUniformsScene now serves multiple arena chunks
 *  (§3) and each chunk has its own main-heap GPUBuffer. */
export interface DerivedUniformsResources {
  /** `array<vec2<f32>>` — df32 trafo halves (the constituents heap). */
  readonly constituentsBuf: () => GPUBuffer;
}

function bglEntries(): GPUBindGroupLayoutEntry[] {
  return [
    // Constituents: read_write — the chain pass writes per-RO Model constituents
    // here (a separate, earlier dispatch); §7 reads them. read_write permits both.
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
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
    // Fields 1..3 are DF32_LIB's fast-math guards (see codegen.ts): zeroBits,
    // one, negOne — runtime values the shader compiler cannot constant-fold.
    const init = new ArrayBuffer(16);
    new Uint32Array(init, 0, 2).set([0, 0]); // count, zeroBits
    new Float32Array(init, 8, 2).set([1.0, -1.0]); // one, negOne
    device.queue.writeBuffer(this.countBuf, 0, init);
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
  // Cache one pipeline per record stride (the kernel bakes RECORD_STRIDE). The
  // chain pass and the §7 pass usually have different strides, and both are
  // dispatched every frame, so a single-slot cache would thrash → recompile
  // per frame. Cleared on a registry-version bump (new rule shape).
  private readonly pipes = new Map<number, GPUComputePipeline>();
  private builtVersion = -1;

  constructor(device: GPUDevice) {
    this.device = device;
    this.bgl = device.createBindGroupLayout({ entries: bglEntries() });
  }

  pipeline(registry: DerivedUniformRegistry, strideU32: number): GPUComputePipeline | undefined {
    // Always buildable: the kernel carries the built-in CHAIN arm even with an
    // empty registry, so a scene using only transform-propagation chains (no
    // §7 rules) still gets a pipeline. encodeChunk's recordCount===0 guard is
    // what skips the dispatch when there's genuinely nothing to do.
    if (registry.version !== this.builtVersion) {
      this.pipes.clear();
      this.builtVersion = registry.version;
    }
    const cached = this.pipes.get(strideU32);
    if (cached !== undefined) return cached;
    const { wgsl } = buildUberKernel(registry, strideU32);
    const module = this.device.createShaderModule({ code: wgsl, label: "derivedUniforms.uber" });
    const pipe = this.device.createComputePipeline({
      label: "derivedUniforms.uber",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bgl] }),
      compute: { module, entryPoint: "main" },
    });
    this.pipes.set(strideU32, pipe);
    return pipe;
  }
}

/** Per-chunk dispatch state: GPU mirror of that chunk's records +
 *  a cached bind group keyed on (consts, heap, recs). */
interface ChunkState {
  readonly gpu: RecordsGpu;
  cachedBg: GPUBindGroup | undefined;
  bgKey: [GPUBuffer | undefined, GPUBuffer | undefined, GPUBuffer | undefined];
}

export class DerivedUniformsDispatcher {
  private readonly device: GPUDevice;
  private readonly resources: DerivedUniformsResources;
  private readonly pipe: UberPipeline;
  private readonly byChunk = new Map<RecordsBuffer, ChunkState>();

  constructor(device: GPUDevice, resources: DerivedUniformsResources) {
    this.device = device;
    this.resources = resources;
    this.pipe = new UberPipeline(device);
  }

  /** Run the uber-kernel for ONE chunk's records, binding
   *  `mainHeapGetter()` as the heap target. No-op if the chunk's
   *  records buffer is empty or the registry has no rules. */
  encodeChunk(
    enc: GPUCommandEncoder,
    registry: DerivedUniformRegistry,
    records: RecordsBuffer,
    mainHeapGetter: () => GPUBuffer,
  ): boolean {
    if (records.recordCount === 0) return false;
    const pipeline = this.pipe.pipeline(registry, records.strideWords);
    if (pipeline === undefined) return false;
    const state = this.stateFor(records);
    state.gpu.sync(records);
    const pass = enc.beginComputePass({ label: "derivedUniforms.uber" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.bindGroup(state, mainHeapGetter()));
    pass.dispatchWorkgroups(Math.ceil(records.recordCount / 64));
    pass.end();
    return true;
  }

  dispose(): void {
    for (const s of this.byChunk.values()) s.gpu.dispose();
    this.byChunk.clear();
  }

  private stateFor(records: RecordsBuffer): ChunkState {
    let s = this.byChunk.get(records);
    if (s === undefined) {
      s = {
        gpu: new RecordsGpu(this.device),
        cachedBg: undefined,
        bgKey: [undefined, undefined, undefined],
      };
      this.byChunk.set(records, s);
    }
    return s;
  }

  private bindGroup(state: ChunkState, heap: GPUBuffer): GPUBindGroup {
    const consts = this.resources.constituentsBuf();
    const recs = state.gpu.buffer();
    if (state.cachedBg !== undefined && state.bgKey[0] === consts && state.bgKey[1] === heap && state.bgKey[2] === recs) {
      return state.cachedBg;
    }
    state.cachedBg = this.device.createBindGroup({
      label: "derivedUniforms.uber",
      layout: this.pipe.bgl,
      entries: [
        { binding: 0, resource: { buffer: consts } },
        { binding: 1, resource: { buffer: heap } },
        { binding: 2, resource: { buffer: recs } },
        { binding: 3, resource: { buffer: state.gpu.countBuf } },
      ],
    });
    state.bgKey = [consts, heap, recs];
    return state.cachedBg;
  }
}
