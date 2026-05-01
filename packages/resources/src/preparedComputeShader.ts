// prepareComputeShader — turn a `ComputeShader` (single compute
// stage, peer of `Effect`) into an imperative dispatch surface.
//
// Mirrors Aardvark.GPGPU's `IComputeShader` + mutable
// `IComputeInputBinding` pattern: callers build a binding by name,
// `set` values/buffers/textures/samplers into it imperatively, then
// `dispatch` to encode + submit (or hand an encoder for batching).
//
// Unlike `prepareRenderObject`, none of the inputs are `aval<T>`. The
// binding owns its own mutable state; uniform writes are byte-poked
// into a staging `ArrayBuffer` and uploaded on dispatch.
//
// Supported binding kinds (from the program interface):
//   - One uniform block per group: `setUniform(name, value)`.
//   - Storage buffers: `setBuffer(name, GPUBuffer)`.
//   - Sampled textures: `setTexture(name, GPUTextureView | GPUTexture)`.
//   - Samplers: `setSampler(name, GPUSampler)`.
//
// Storage textures aren't covered here yet — separate gap.

import {
  type ComputeShader,
  type ProgramInterface,
} from "@aardworx/wombat.rendering-core";
import type { StorageTextureFormat, Type } from "@aardworx/wombat.shader/ir";
import { ShaderStage } from "./webgpuFlags.js";
import { installShaderDiagnostics } from "./shaderDiagnostics.js";
import { makePackedView, writeField, type PackedView } from "./uniformBuffer.js";

// ---------------------------------------------------------------------------
// Bind-group-entry descriptions (mirrors PreparedRenderObject's shape, but
// over plain GPU handles instead of AdaptiveResource<...>).
// ---------------------------------------------------------------------------

type EntryDesc =
  | { kind: "ubuf";   binding: number; uniformBlock: import("@aardworx/wombat.rendering-core").UniformBlockInfo }
  | { kind: "sbuf";   binding: number; name: string; access: "read" | "read_write" }
  | { kind: "tex";    binding: number; name: string; sampleType: GPUTextureSampleType }
  | { kind: "sampler";binding: number; name: string }
  | { kind: "stex";   binding: number; name: string; format: GPUTextureFormat; access: GPUStorageTextureAccess };

interface GroupDesc {
  readonly group: number;
  readonly layout: GPUBindGroupLayout;
  readonly entries: readonly EntryDesc[];
}

function storageAccessToGPU(a: "read" | "write" | "read_write"): GPUStorageTextureAccess {
  switch (a) {
    case "read":       return "read-only";
    case "write":      return "write-only";
    case "read_write": return "read-write";
  }
}

function sampleTypeFor(type: Type): GPUTextureSampleType {
  if (type.kind === "Texture") {
    if (type.comparison === true) return "depth";
    const s = type.sampled;
    if (s.kind === "Int") return s.signed ? "sint" : "uint";
    return "float";
  }
  return "float";
}

function buildGroups(device: GPUDevice, descs: readonly EntryDesc[][]): GroupDesc[] {
  const out: GroupDesc[] = [];
  for (let g = 0; g < descs.length; g++) {
    const entries = (descs[g] ?? []).slice().sort((a, b) => a.binding - b.binding);
    const layoutEntries: GPUBindGroupLayoutEntry[] = entries.map(e => {
      const visibility = ShaderStage.COMPUTE;
      switch (e.kind) {
        case "ubuf":    return { binding: e.binding, visibility, buffer: { type: "uniform" } };
        case "sbuf":    return {
          binding: e.binding, visibility,
          buffer: { type: e.access === "read_write" ? "storage" : "read-only-storage" },
        };
        case "tex":     return { binding: e.binding, visibility, texture: { sampleType: e.sampleType } };
        case "sampler": return { binding: e.binding, visibility, sampler: { type: "filtering" } };
        case "stex":    return { binding: e.binding, visibility, storageTexture: { access: e.access, format: e.format } };
      }
    });
    const layout = device.createBindGroupLayout({ entries: layoutEntries });
    out.push({ group: g, layout, entries });
  }
  return out;
}

// ---------------------------------------------------------------------------
// ComputeInputBinding — mutable, name-keyed state for one shader.
// ---------------------------------------------------------------------------

export interface DispatchSize {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

interface UboField {
  readonly offset: number;
  readonly type: import("@aardworx/wombat.rendering-core").UniformFieldInfo["type"];
}

interface UboState {
  readonly group: number;
  readonly binding: number;
  readonly view: PackedView;
  readonly fields: ReadonlyMap<string, UboField>;
  buffer: GPUBuffer | undefined;
  dirty: boolean;
}

export class ComputeInputBinding {
  private readonly _ubos = new Map<string, UboState>();              // ubo name → state
  private readonly _ubosByBinding = new Map<string, UboState>();     // "g:b" → state
  private readonly _buffers = new Map<string, GPUBuffer>();          // name → buffer
  private readonly _textures = new Map<string, GPUTextureView>();    // name → view
  private readonly _samplers = new Map<string, GPUSampler>();        // name → sampler
  private readonly _storageTextures = new Map<string, GPUTextureView>(); // name → view (storage)
  /** Names declared by the shader as storage textures (vs sampled). */
  private readonly _storageTextureNames: ReadonlySet<string>;

  /** @internal — built by `PreparedComputeShader`. */
  constructor(
    private readonly device: GPUDevice,
    private readonly groups: readonly GroupDesc[],
    private readonly iface: ProgramInterface,
    private readonly label: string | undefined,
  ) {
    this._storageTextureNames = new Set(
      iface.samplers.filter(s => s.type.kind === "StorageTexture").map(s => s.name),
    );
    for (const ub of iface.uniformBlocks) {
      const view = makePackedView(ub.size);
      const fields = new Map<string, UboField>(
        ub.fields.map(f => [f.name, { offset: f.offset, type: f.type }]),
      );
      const state: UboState = {
        group: ub.group, binding: ub.slot, view, fields,
        buffer: undefined, dirty: true,
      };
      this._ubos.set(ub.name, state);
      this._ubosByBinding.set(`${ub.group}:${ub.slot}`, state);
    }
  }

  /**
   * Write a uniform value by name. Searches all uniform blocks; the
   * first hit wins. Throws if the name isn't a uniform field.
   * Accepted types: `number`, `Float32Array`/`Int32Array`/`Uint32Array`,
   * any wombat.base packed value (`{ _data: TypedArray }`).
   */
  setUniform(name: string, value: unknown): this {
    for (const ub of this._ubos.values()) {
      const f = ub.fields.get(name);
      if (f === undefined) continue;
      writeField(ub.view, f.offset, value, f.type);
      ub.dirty = true;
      return this;
    }
    throw new Error(`ComputeInputBinding: no uniform field "${name}"`);
  }

  setBuffer(name: string, buffer: GPUBuffer): this {
    if (!this.iface.storageBuffers.some(s => s.name === name)) {
      throw new Error(`ComputeInputBinding: no storage buffer "${name}"`);
    }
    this._buffers.set(name, buffer);
    return this;
  }

  setTexture(name: string, source: GPUTexture | GPUTextureView): this {
    if (!this.iface.textures.some(t => t.name === name)) {
      throw new Error(`ComputeInputBinding: no texture "${name}"`);
    }
    const view = "createView" in source ? source.createView() : source;
    this._textures.set(name, view);
    return this;
  }

  /**
   * Bind a storage-texture view (write/read/read_write per the
   * shader declaration). `view` should reference a `GPUTexture`
   * created with `STORAGE_BINDING` usage in the format the shader
   * expects.
   */
  setStorageTexture(name: string, source: GPUTexture | GPUTextureView): this {
    if (!this._storageTextureNames.has(name)) {
      throw new Error(`ComputeInputBinding: no storage texture "${name}"`);
    }
    const view = "createView" in source ? source.createView() : source;
    this._storageTextures.set(name, view);
    return this;
  }

  setSampler(name: string, sampler: GPUSampler): this {
    if (!this.iface.samplers.some(s => s.name === name)) {
      throw new Error(`ComputeInputBinding: no sampler "${name}"`);
    }
    this._samplers.set(name, sampler);
    return this;
  }

  /** @internal — flush dirty UBOs and build current bind groups. */
  flushAndBuildBindGroups(): GPUBindGroup[] {
    for (const ub of this._ubos.values()) {
      if (ub.buffer === undefined) {
        ub.buffer = this.device.createBuffer({
          size: ub.view.bytes.byteLength,
          usage: 0x40 | 0x8, // UNIFORM | COPY_DST
          ...(this.label !== undefined ? { label: `${this.label}.ubo` } : {}),
        });
      }
      if (ub.dirty) {
        this.device.queue.writeBuffer(ub.buffer, 0, ub.view.bytes);
        ub.dirty = false;
      }
    }

    const out: GPUBindGroup[] = [];
    for (const g of this.groups) {
      const entries: GPUBindGroupEntry[] = g.entries.map(e => {
        switch (e.kind) {
          case "ubuf": {
            const ub = this._ubosByBinding.get(`${g.group}:${e.binding}`);
            if (!ub || ub.buffer === undefined) {
              throw new Error(`ComputeInputBinding.dispatch: UBO at group=${g.group} binding=${e.binding} not initialised`);
            }
            return { binding: e.binding, resource: { buffer: ub.buffer } };
          }
          case "sbuf": {
            const buf = this._buffers.get(e.name);
            if (buf === undefined) throw new Error(`ComputeInputBinding.dispatch: missing storage buffer "${e.name}"`);
            return { binding: e.binding, resource: { buffer: buf } };
          }
          case "tex": {
            const view = this._textures.get(e.name);
            if (view === undefined) throw new Error(`ComputeInputBinding.dispatch: missing texture "${e.name}"`);
            return { binding: e.binding, resource: view };
          }
          case "sampler": {
            const samp = this._samplers.get(e.name);
            if (samp === undefined) throw new Error(`ComputeInputBinding.dispatch: missing sampler "${e.name}"`);
            return { binding: e.binding, resource: samp };
          }
          case "stex": {
            const view = this._storageTextures.get(e.name);
            if (view === undefined) throw new Error(`ComputeInputBinding.dispatch: missing storage texture "${e.name}"`);
            return { binding: e.binding, resource: view };
          }
        }
      });
      out.push(this.device.createBindGroup({ layout: g.layout, entries }));
    }
    return out;
  }

  /** Free the binding's owned UBO buffers. Storage buffers / textures / samplers are caller-owned. */
  dispose(): void {
    for (const ub of this._ubos.values()) {
      if (ub.buffer !== undefined) {
        ub.buffer.destroy();
        ub.buffer = undefined;
      }
      ub.dirty = true;
    }
  }
}

// ---------------------------------------------------------------------------
// PreparedComputeShader
// ---------------------------------------------------------------------------

export interface PrepareComputeShaderOptions {
  readonly label?: string;
}

export class PreparedComputeShader {
  /** @internal */
  constructor(
    private readonly device: GPUDevice,
    public readonly pipeline: GPUComputePipeline,
    public readonly interface_: ProgramInterface,
    private readonly groups: readonly GroupDesc[],
    private readonly label: string | undefined,
  ) {}

  /** Same shape as Aardvark's `IComputeShader.Interface` — useful for tooling. */
  get programInterface(): ProgramInterface { return this.interface_; }

  /** Allocate a fresh, empty binding. */
  createInputBinding(): ComputeInputBinding {
    return new ComputeInputBinding(this.device, this.groups, this.interface_, this.label);
  }

  /**
   * Encode `Bind + SetInput + Dispatch` into the supplied encoder.
   * Use this when batching compute work alongside other GPU commands
   * — caller submits.
   */
  encode(encoder: GPUCommandEncoder, binding: ComputeInputBinding, groups: DispatchSize): void {
    const bindGroups = binding.flushAndBuildBindGroups();
    const pass = encoder.beginComputePass({
      ...(this.label !== undefined ? { label: this.label } : {}),
    });
    pass.setPipeline(this.pipeline);
    for (let i = 0; i < bindGroups.length; i++) pass.setBindGroup(i, bindGroups[i]!);
    pass.dispatchWorkgroups(groups.x, groups.y, groups.z);
    pass.end();
  }

  /**
   * One-shot dispatch — builds an encoder, encodes, submits, awaits
   * `queue.onSubmittedWorkDone()`. Aardvark's `IComputeShader.Invoke`
   * shape.
   */
  async dispatch(binding: ComputeInputBinding, groups: DispatchSize): Promise<void> {
    const encoder = this.device.createCommandEncoder({
      ...(this.label !== undefined ? { label: `${this.label}.dispatch` } : {}),
    });
    this.encode(encoder, binding, groups);
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }
}

// ---------------------------------------------------------------------------
// Compute pipeline cache
// ---------------------------------------------------------------------------

interface ComputePipelineCache {
  pipelines: Map<string, GPUComputePipeline>;
  modules: Map<string, GPUShaderModule>;
}
const computeCaches = new WeakMap<GPUDevice, ComputePipelineCache>();
function cacheFor(device: GPUDevice): ComputePipelineCache {
  let c = computeCaches.get(device);
  if (c === undefined) {
    c = { pipelines: new Map(), modules: new Map() };
    computeCaches.set(device, c);
  }
  return c;
}

// ---------------------------------------------------------------------------
// prepareComputeShader
// ---------------------------------------------------------------------------

export function prepareComputeShader(
  device: GPUDevice,
  shader: ComputeShader,
  opts: PrepareComputeShaderOptions = {},
): PreparedComputeShader {
  const compiled = shader.compile({ target: "wgsl" });
  const csStage = compiled.stages.find(s => s.stage === "compute");
  if (csStage === undefined) {
    throw new Error("prepareComputeShader: ComputeShader compiled without a compute stage");
  }
  const iface = compiled.interface;

  const maxGroup = Math.max(
    -1,
    ...iface.uniformBlocks.map(b => b.group),
    ...iface.samplers.map(b => b.group),
    ...iface.textures.map(b => b.group),
    ...iface.storageBuffers.map(b => b.group),
  );
  const perGroup: EntryDesc[][] = [];
  for (let g = 0; g <= maxGroup; g++) perGroup.push([]);

  for (const ub of iface.uniformBlocks) {
    perGroup[ub.group]!.push({ kind: "ubuf", binding: ub.slot, uniformBlock: ub });
  }
  for (const t of iface.textures) {
    perGroup[t.group]!.push({ kind: "tex", binding: t.slot, name: t.name, sampleType: sampleTypeFor(t.type) });
  }
  for (const s of iface.samplers) {
    if (s.type.kind === "StorageTexture") {
      perGroup[s.group]!.push({
        kind: "stex", binding: s.slot, name: s.name,
        format: s.type.format as GPUTextureFormat,
        access: storageAccessToGPU(s.type.access),
      });
    } else {
      perGroup[s.group]!.push({ kind: "sampler", binding: s.slot, name: s.name });
    }
  }
  for (const sb of iface.storageBuffers) {
    perGroup[sb.group]!.push({ kind: "sbuf", binding: sb.slot, name: sb.name, access: sb.access });
  }

  const groups = buildGroups(device, perGroup);

  // Compile module + pipeline (cached per device by effect id).
  const cache = cacheFor(device);
  const moduleKey = csStage.source;
  let module = cache.modules.get(moduleKey);
  if (module === undefined) {
    module = device.createShaderModule({
      code: csStage.source,
      ...(opts.label !== undefined ? { label: `${opts.label}.cs` } : {}),
    });
    cache.modules.set(moduleKey, module);
    installShaderDiagnostics(module, csStage.source, {
      ...(opts.label !== undefined ? { label: opts.label } : {}),
      ...(csStage.sourceMap ? { sourceMap: csStage.sourceMap } : {}),
    });
  }

  const pipelineKey = `${shader.id}|${csStage.entryName}|${groups.length}`;
  let pipeline = cache.pipelines.get(pipelineKey);
  if (pipeline === undefined) {
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: groups.map(g => g.layout),
      ...(opts.label !== undefined ? { label: `${opts.label}.layout` } : {}),
    });
    pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: csStage.entryName },
      ...(opts.label !== undefined ? { label: opts.label } : {}),
    });
    cache.pipelines.set(pipelineKey, pipeline);
  }

  return new PreparedComputeShader(device, pipeline, iface, groups, opts.label);
}
