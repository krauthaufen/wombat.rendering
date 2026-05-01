// prepareRenderObject — turn a `RenderObject` + `CompiledEffect`
// + `FramebufferSignature` into a runnable `PreparedRenderObject`.
//
// Reads the wombat.shader `ProgramInterface` for everything needed
// to lay out vertex buffers, bind groups, color targets, and draw
// state.

import {
  AdaptiveResource,
  type CompiledEffect,
  type FramebufferSignature,
  type ProgramInterface,
  type RenderObject,
} from "@aardworx/wombat.rendering-core";
import {
  type AdaptiveToken,
  type aval,
  type HashMap,
  cval,
} from "@aardworx/wombat.adaptive";
import type { Type } from "@aardworx/wombat.shader/ir";
import { prepareAdaptiveBuffer } from "./adaptiveBuffer.js";
import { prepareAdaptiveTexture } from "./adaptiveTexture.js";
import { prepareAdaptiveSampler } from "./adaptiveSampler.js";
import { prepareUniformBuffer } from "./uniformBuffer.js";
import { compileRenderPipeline, type CompileRenderPipelineDescription } from "./renderPipeline.js";
import { BufferUsage, ShaderStage } from "./webgpuFlags.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function vertexFormatStride(fmt: GPUVertexFormat): number {
  switch (fmt) {
    case "float32": return 4;
    case "float32x2": return 8;
    case "float32x3": return 12;
    case "float32x4": return 16;
    case "uint32": case "sint32": return 4;
    case "uint32x2": case "sint32x2": return 8;
    case "uint32x3": case "sint32x3": return 12;
    case "uint32x4": case "sint32x4": return 16;
    case "uint16x2": case "sint16x2": case "float16x2": return 4;
    case "uint16x4": case "sint16x4": case "float16x4": return 8;
    case "uint8x2": case "sint8x2": case "unorm8x2": case "snorm8x2": return 2;
    case "uint8x4": case "sint8x4": case "unorm8x4": case "snorm8x4": return 4;
    default: throw new Error(`vertexFormatStride: unsupported format ${fmt}`);
  }
}

interface VertexBindingInfo {
  readonly name: string;
  readonly slot: number;
  readonly format: GPUVertexFormat;
  readonly byteSize: number;
}

function vertexBindingsFor(iface: ProgramInterface): VertexBindingInfo[] {
  // `iface.attributes` is already filtered to non-builtin vertex inputs
  // by wombat.shader's interface builder.
  return iface.attributes.map(a => ({
    name: a.name,
    slot: a.location,
    format: a.format as GPUVertexFormat,
    byteSize: a.byteSize,
  }));
}

function colorTargetsFor(
  iface: ProgramInterface,
  sig: FramebufferSignature,
  blends: import("@aardworx/wombat.adaptive").HashMap<string, import("@aardworx/wombat.rendering-core").BlendState> | undefined,
): GPUColorTargetState[] {
  const out: GPUColorTargetState[] = [];
  for (const o of iface.fragmentOutputs) {
    const fmt = sig.colors.tryFind(o.name);
    if (fmt === undefined) {
      throw new Error(`prepareRenderObject: fragment output "${o.name}" has no matching signature attachment`);
    }
    const target: GPUColorTargetState = { format: fmt };
    const blend = blends?.tryFind(o.name);
    if (blend !== undefined) {
      target.blend = {
        color: { operation: blend.color.operation, srcFactor: blend.color.srcFactor, dstFactor: blend.color.dstFactor },
        alpha: { operation: blend.alpha.operation, srcFactor: blend.alpha.srcFactor, dstFactor: blend.alpha.dstFactor },
      };
      target.writeMask = blend.writeMask;
    }
    out[o.location] = target;
  }
  return out;
}

function depthStencilStateFor(
  sig: FramebufferSignature,
  ps: import("@aardworx/wombat.rendering-core").PipelineState,
): GPUDepthStencilState | undefined {
  if (sig.depthStencil === undefined) return undefined;
  if (ps.depth === undefined && ps.stencil === undefined) return undefined;
  const out: GPUDepthStencilState = {
    format: sig.depthStencil.format,
    depthWriteEnabled: ps.depth?.write ?? false,
    depthCompare: ps.depth?.compare ?? "always",
  };
  if (ps.stencil !== undefined) {
    out.stencilFront = ps.stencil.front;
    out.stencilBack = ps.stencil.back;
    out.stencilReadMask = ps.stencil.readMask;
    out.stencilWriteMask = ps.stencil.writeMask;
  }
  if (ps.rasterizer.depthBias !== undefined) {
    out.depthBias = ps.rasterizer.depthBias.constant;
    out.depthBiasSlopeScale = ps.rasterizer.depthBias.slopeScale;
    out.depthBiasClamp = ps.rasterizer.depthBias.clamp;
  }
  return out;
}

/** Map a wombat.shader IR Type for a sampled texture to WebGPU's GPUTextureSampleType. */
function sampleTypeFor(type: Type): GPUTextureSampleType {
  if (type.kind === "Texture") {
    if (type.comparison === true) return "depth";
    const s = type.sampled;
    if (s.kind === "Int") return s.signed ? "sint" : "uint";
    return "float";
  }
  // Storage textures don't have a sampleType in the binding-layout
  // sense; this code path is only for sampled textures. If we see
  // anything else default to filterable float.
  return "float";
}

// ---------------------------------------------------------------------------
// Bind-group entry descriptions
// ---------------------------------------------------------------------------

type EntryDesc =
  | { kind: "ubuf";    binding: number; resource: AdaptiveResource<GPUBuffer> }
  | { kind: "sbuf";    binding: number; resource: AdaptiveResource<GPUBuffer>; access: "read" | "read_write" }
  | { kind: "tex";     binding: number; resource: AdaptiveResource<GPUTexture>; sampleType: GPUTextureSampleType }
  | { kind: "sampler"; binding: number; resource: AdaptiveResource<GPUSampler> };

interface GroupDesc {
  readonly group: number;
  readonly layout: GPUBindGroupLayout;
  readonly entries: readonly EntryDesc[];
}

function buildGroups(device: GPUDevice, descs: readonly EntryDesc[][]): GroupDesc[] {
  const out: GroupDesc[] = [];
  for (let g = 0; g < descs.length; g++) {
    const entries = (descs[g] ?? []).slice().sort((a, b) => a.binding - b.binding);
    const layoutEntries: GPUBindGroupLayoutEntry[] = entries.map(e => {
      const visibility = ShaderStage.VERTEX | ShaderStage.FRAGMENT;
      switch (e.kind) {
        case "ubuf":   return { binding: e.binding, visibility, buffer: { type: "uniform" } };
        case "sbuf":   return {
          binding: e.binding, visibility,
          buffer: { type: e.access === "read_write" ? "storage" : "read-only-storage" },
        };
        case "tex":    return { binding: e.binding, visibility, texture: { sampleType: e.sampleType } };
        case "sampler": return { binding: e.binding, visibility, sampler: { type: "filtering" } };
      }
    });
    const layout = device.createBindGroupLayout({ entries: layoutEntries });
    out.push({ group: g, layout, entries });
  }
  return out;
}

// Per-handle identity counter for the bind-group cache key. WebGPU
// handles have no built-in id; we attach one via a WeakMap.
const handleIds = new WeakMap<object, number>();
let nextHandleId = 1;
function handleId(h: object): number {
  let id = handleIds.get(h);
  if (id === undefined) { id = nextHandleId++; handleIds.set(h, id); }
  return id;
}

// ---------------------------------------------------------------------------
// PreparedRenderObject
// ---------------------------------------------------------------------------

export class PreparedRenderObject {
  readonly pipeline: GPURenderPipeline;
  readonly groups: readonly GroupDesc[];

  // Per-group bind group cache. Keyed on the concatenated identity
  // of the underlying GPU handles (`buffer:N`, `tex:N`, etc.); a
  // cache hit means none of the resources reallocated since the
  // last record(), so the GPUBindGroup is reusable.
  private readonly _bindGroupCache: { key: string | null; group: GPUBindGroup | null }[];

  constructor(
    private readonly device: GPUDevice,
    private readonly vertexBindings: readonly VertexBindingInfo[],
    private readonly vertexBuffers: ReadonlyMap<string, AdaptiveResource<GPUBuffer>>,
    private readonly indexBuffer: AdaptiveResource<GPUBuffer> | undefined,
    private readonly indexFormat: GPUIndexFormat | undefined,
    private readonly indices: aval<import("@aardworx/wombat.rendering-core").BufferView> | undefined,
    groups: readonly GroupDesc[],
    private readonly drawCall: aval<import("@aardworx/wombat.rendering-core").DrawCall>,
    pipeline: GPURenderPipeline,
  ) {
    this.pipeline = pipeline;
    this.groups = groups;
    this._bindGroupCache = groups.map(() => ({ key: null, group: null }));
  }

  acquire(): void {
    for (const r of this.vertexBuffers.values()) r.acquire();
    if (this.indexBuffer !== undefined) this.indexBuffer.acquire();
    for (const g of this.groups) for (const e of g.entries) e.resource.acquire();
  }

  release(): void {
    for (const r of this.vertexBuffers.values()) r.release();
    if (this.indexBuffer !== undefined) this.indexBuffer.release();
    for (const g of this.groups) for (const e of g.entries) e.resource.release();
  }

  record(pass: GPURenderPassEncoder, token: AdaptiveToken): void {
    pass.setPipeline(this.pipeline);

    for (let gi = 0; gi < this.groups.length; gi++) {
      const g = this.groups[gi]!;
      // Build a cache key from the resolved GPU handle identities;
      // resources may be re-allocated by their AdaptiveResource on
      // capacity growth or shape change, so we need to key on the
      // CURRENT handles, not on the AdaptiveResource references.
      let cacheKey = "";
      const resolved: { binding: number; resource: GPUBindingResource }[] = [];
      for (const e of g.entries) {
        switch (e.kind) {
          case "ubuf":
          case "sbuf": {
            const buf = e.resource.getValue(token);
            cacheKey += `${e.binding}:b${handleId(buf)};`;
            resolved.push({ binding: e.binding, resource: { buffer: buf } });
            break;
          }
          case "tex": {
            const tex = e.resource.getValue(token);
            cacheKey += `${e.binding}:t${handleId(tex)};`;
            // We create a new view per record because views are cheap
            // and texture identity already keys the cache. (A
            // real-perf-tuned implementation would also cache the
            // view per texture.)
            resolved.push({ binding: e.binding, resource: tex.createView() });
            break;
          }
          case "sampler": {
            const samp = e.resource.getValue(token);
            cacheKey += `${e.binding}:s${handleId(samp)};`;
            resolved.push({ binding: e.binding, resource: samp });
            break;
          }
        }
      }
      const slot = this._bindGroupCache[gi]!;
      let bg: GPUBindGroup;
      if (slot.key === cacheKey && slot.group !== null) {
        bg = slot.group;
      } else {
        bg = this.device.createBindGroup({ layout: g.layout, entries: resolved });
        slot.key = cacheKey;
        slot.group = bg;
      }
      pass.setBindGroup(g.group, bg);
    }

    for (const vb of this.vertexBindings) {
      const res = this.vertexBuffers.get(vb.name);
      if (res === undefined) throw new Error(`PreparedRenderObject.record: missing vertex buffer for "${vb.name}"`);
      pass.setVertexBuffer(vb.slot, res.getValue(token));
    }

    if (this.indexBuffer !== undefined && this.indices !== undefined && this.indexFormat !== undefined) {
      const buf = this.indexBuffer.getValue(token);
      const view = this.indices.getValue(token);
      pass.setIndexBuffer(buf, this.indexFormat, view.offset, view.count * (this.indexFormat === "uint16" ? 2 : 4));
    }

    const dc = this.drawCall.getValue(token);
    if (dc.kind === "indexed") {
      pass.drawIndexed(dc.indexCount, dc.instanceCount, dc.firstIndex, dc.baseVertex, dc.firstInstance);
    } else {
      pass.draw(dc.vertexCount, dc.instanceCount, dc.firstVertex, dc.firstInstance);
    }
  }
}

// ---------------------------------------------------------------------------
// prepareRenderObject
// ---------------------------------------------------------------------------

export interface PrepareRenderObjectOptions {
  readonly label?: string;
  /**
   * Stable identity of the upstream effect — wombat.shader's
   * `Effect.id`. Threaded into the pipeline cache key so that
   * pipelines compiled from the same effect against the same
   * signature dedupe correctly.
   */
  readonly effectId?: string;
}

export function prepareRenderObject(
  device: GPUDevice,
  obj: RenderObject,
  effect: CompiledEffect,
  signature: FramebufferSignature,
  opts: PrepareRenderObjectOptions = {},
): PreparedRenderObject {
  const iface = effect.interface;
  const vertexBindings = vertexBindingsFor(iface);

  // Vertex + instance buffers — same shape; only stepMode differs.
  // Each shader-required attribute is looked up in obj.vertexAttributes
  // first (per-vertex) then obj.instanceAttributes (per-instance).
  const vertexBuffers = new Map<string, AdaptiveResource<GPUBuffer>>();
  const vertexLayouts: GPUVertexBufferLayout[] = [];
  for (const vb of vertexBindings) {
    const perVertex = obj.vertexAttributes.tryFind(vb.name);
    const perInstance = obj.instanceAttributes?.tryFind(vb.name);
    const av = perVertex ?? perInstance;
    if (av === undefined) throw new Error(`prepareRenderObject: missing vertex attribute "${vb.name}"`);
    const stepMode: GPUVertexStepMode = perInstance !== undefined ? "instance" : "vertex";
    const stride = vb.byteSize !== undefined && vb.byteSize > 0
      ? vb.byteSize
      : vertexFormatStride(vb.format);
    const bufAval = av.map(view => view.buffer);
    const res = prepareAdaptiveBuffer(device, bufAval, {
      usage: BufferUsage.VERTEX,
      ...(opts.label !== undefined ? { label: `${opts.label}.${vb.name}` } : {}),
    });
    vertexBuffers.set(vb.name, res);
    vertexLayouts[vb.slot] = {
      arrayStride: stride,
      stepMode,
      attributes: [{ shaderLocation: vb.slot, offset: 0, format: vb.format }],
    };
  }

  // Index buffer — `indexFormat` from BufferView wins; defaults to uint32.
  let indexBuffer: AdaptiveResource<GPUBuffer> | undefined;
  let indexFormat: GPUIndexFormat | undefined;
  if (obj.indices !== undefined) {
    const bufAval = obj.indices.map(v => v.buffer);
    indexBuffer = prepareAdaptiveBuffer(device, bufAval, {
      usage: BufferUsage.INDEX,
      ...(opts.label !== undefined ? { label: `${opts.label}.indices` } : {}),
    });
    const initial = obj.indices.force();
    indexFormat = initial.indexFormat
      ?? (initial.format === "uint16" ? "uint16" : "uint32");
  }

  // Per-group entry collection. Use `slot` (real shape) instead of `binding`.
  const slotOf = (e: { group: number; slot: number }) => e.slot;
  const groupOf = (e: { group: number }) => e.group;
  const maxGroup = Math.max(
    -1,
    ...iface.uniformBlocks.map(groupOf),
    ...iface.samplers.map(groupOf),
    ...iface.textures.map(groupOf),
    ...iface.storageBuffers.map(groupOf),
  );
  const perGroup: EntryDesc[][] = [];
  for (let g = 0; g <= maxGroup; g++) perGroup.push([]);

  for (const ub of iface.uniformBlocks) {
    // Merge user-supplied uniforms (`obj.uniforms`) with effect-bound
    // avals (`compiledEffect.avalBindings`). User entries win on name
    // conflict — the rendering caller is the source of truth.
    const merged = mergeUniformInputs(obj.uniforms, effect.avalBindings, ub);
    const block = ubAsBlock(ub);
    const res = prepareUniformBuffer(device, block, merged, {
      ...(opts.label !== undefined ? { label: `${opts.label}.${ub.name}` } : {}),
    });
    perGroup[ub.group]!.push({ kind: "ubuf", binding: slotOf(ub), resource: res });
  }

  for (const t of iface.textures) {
    const av = obj.textures.tryFind(t.name);
    if (av === undefined) throw new Error(`prepareRenderObject: missing texture "${t.name}"`);
    const res = prepareAdaptiveTexture(device, av, {
      ...(opts.label !== undefined ? { label: `${opts.label}.${t.name}` } : {}),
    });
    perGroup[t.group]!.push({ kind: "tex", binding: slotOf(t), resource: res, sampleType: sampleTypeFor(t.type) });
  }

  for (const s of iface.samplers) {
    const av = obj.samplers.tryFind(s.name);
    if (av === undefined) throw new Error(`prepareRenderObject: missing sampler "${s.name}"`);
    const res = prepareAdaptiveSampler(device, av);
    perGroup[s.group]!.push({ kind: "sampler", binding: slotOf(s), resource: res });
  }

  for (const sb of iface.storageBuffers) {
    const av = obj.storageBuffers?.tryFind(sb.name);
    if (av === undefined) throw new Error(`prepareRenderObject: missing storage buffer "${sb.name}"`);
    const usage = BufferUsage.STORAGE
      | (sb.access === "read_write" ? BufferUsage.COPY_DST : 0);
    const res = prepareAdaptiveBuffer(device, av, {
      usage,
      ...(opts.label !== undefined ? { label: `${opts.label}.${sb.name}` } : {}),
    });
    perGroup[sb.group]!.push({ kind: "sbuf", binding: slotOf(sb), resource: res, access: sb.access });
  }

  const groups = buildGroups(device, perGroup);

  // Pipeline.
  const colorTargets = colorTargetsFor(iface, signature, obj.pipelineState.blends);
  const vsStage = effect.stages.find(s => s.stage === "vertex");
  const fsStage = effect.stages.find(s => s.stage === "fragment");
  if (vsStage === undefined) throw new Error("prepareRenderObject: CompiledEffect has no vertex stage");
  if (fsStage === undefined) throw new Error("prepareRenderObject: CompiledEffect has no fragment stage");

  const ds = depthStencilStateFor(signature, obj.pipelineState);
  const multisample: GPUMultisampleState | undefined =
    signature.sampleCount > 1 || obj.pipelineState.alphaToCoverage === true
      ? {
          count: signature.sampleCount,
          ...(obj.pipelineState.alphaToCoverage === true ? { alphaToCoverageEnabled: true } : {}),
        }
      : undefined;
  const pipelineDesc: CompileRenderPipelineDescription = {
    ...(opts.label !== undefined ? { label: opts.label } : {}),
    ...(opts.effectId !== undefined ? { effectId: opts.effectId } : {}),
    vertexShaderSource: vsStage.source,
    fragmentShaderSource: fsStage.source,
    ...(vsStage.sourceMap ? { vertexSourceMap: vsStage.sourceMap } : {}),
    ...(fsStage.sourceMap ? { fragmentSourceMap: fsStage.sourceMap } : {}),
    vertexEntryPoint: vsStage.entryName,
    fragmentEntryPoint: fsStage.entryName,
    vertexBufferLayouts: vertexLayouts,
    bindGroupLayouts: groups.map(g => g.layout),
    colorTargets,
    primitive: {
      topology: obj.pipelineState.rasterizer.topology,
      ...(obj.pipelineState.rasterizer.cullMode !== "none" ? { cullMode: obj.pipelineState.rasterizer.cullMode } : {}),
      frontFace: obj.pipelineState.rasterizer.frontFace,
    },
    ...(ds !== undefined ? { depthStencil: ds } : {}),
    ...(multisample !== undefined ? { multisample } : {}),
  };
  const pipeline = compileRenderPipeline(device, pipelineDesc);

  return new PreparedRenderObject(
    device,
    vertexBindings, vertexBuffers,
    indexBuffer, indexFormat, obj.indices,
    groups,
    obj.drawCall,
    pipeline,
  );
}

// ---------------------------------------------------------------------------
// Uniform input merging — `obj.uniforms` wins; `avalBindings` is fallback.
// ---------------------------------------------------------------------------

function mergeUniformInputs(
  user: HashMap<string, aval<unknown>>,
  bindings: Readonly<Record<string, () => unknown>> | undefined,
  block: import("@aardworx/wombat.rendering-core").UniformBlockInfo,
): HashMap<string, aval<unknown>> {
  let merged = user;
  if (bindings === undefined) return merged;
  for (const f of block.fields) {
    if (merged.tryFind(f.name) !== undefined) continue;
    const getter = bindings[f.name];
    if (getter === undefined) continue;
    const v = getter();
    // wombat.shader's avalBindings store either an `aval<T>` or a
    // raw value; wrap the raw case as a constant aval.
    if (isAval(v)) merged = merged.add(f.name, v as aval<unknown>);
    else merged = merged.add(f.name, cval(v));
  }
  return merged;
}

function isAval(v: unknown): boolean {
  return typeof v === "object" && v !== null && typeof (v as { getValue?: unknown }).getValue === "function";
}

/**
 * Adapter — wombat.shader's `UniformBlockInfo` and our internal
 * UBO packer expect the same field shape (`offset`, `size`, etc.),
 * so this is just a structural pass-through. Keeps the call sites
 * type-clean.
 */
function ubAsBlock(ub: import("@aardworx/wombat.rendering-core").UniformBlockInfo): import("@aardworx/wombat.rendering-core").UniformBlockInfo {
  return ub;
}
