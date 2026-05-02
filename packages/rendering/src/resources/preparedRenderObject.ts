// prepareRenderObject — turn a `RenderObject` + `CompiledEffect`
// + `FramebufferSignature` into a runnable `PreparedRenderObject`.
//
// Reads the wombat.shader `ProgramInterface` for everything needed
// to lay out vertex buffers, bind groups, color targets, and draw
// state.
//
// Pipeline state is fully reactive: the pipeline-influencing avals
// (rasterizer / depth / stencil ops / blends / alphaToCoverage) are
// snapshotted at token-evaluation time into a cache key; the
// resulting `GPURenderPipeline` is cached per-key in the
// PreparedRenderObject so flipping a non-pipeline-affecting field
// (`stencilReference`, `blendConstant`) does NOT trigger a recompile.

import {
  AdaptiveResource,
  type CompiledEffect,
  type FramebufferSignature,
  type ProgramInterface,
  type RenderObject,
} from "../core/index.js";
import {
  type AdaptiveToken,
  type aval,
  HashMap,
  cval,
} from "@aardworx/wombat.adaptive";
import type { BufferView } from "../core/bufferView.js";
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
  return iface.attributes.map(a => ({
    name: a.name,
    slot: a.location,
    format: a.format as GPUVertexFormat,
    byteSize: a.byteSize,
  }));
}

// ---------------------------------------------------------------------------
// Plain pipeline-state snapshot — what we evaluate avals into per-frame.
// ---------------------------------------------------------------------------

interface BlendComponentSnap {
  readonly operation: GPUBlendOperation;
  readonly srcFactor: GPUBlendFactor;
  readonly dstFactor: GPUBlendFactor;
}
interface BlendSnap {
  readonly color: BlendComponentSnap;
  readonly alpha: BlendComponentSnap;
  readonly writeMask: number;
}
interface StencilFaceSnap {
  readonly compare: GPUCompareFunction;
  readonly failOp: GPUStencilOperation;
  readonly depthFailOp: GPUStencilOperation;
  readonly passOp: GPUStencilOperation;
}
interface StencilSnap {
  readonly enabled: boolean;
  readonly readMask: number;
  readonly writeMask: number;
  readonly front: StencilFaceSnap;
  readonly back: StencilFaceSnap;
}
interface PipelineSnap {
  readonly topology: GPUPrimitiveTopology;
  readonly cullMode: import("../core/index.js").CullMode;
  readonly frontFace: import("../core/index.js").FrontFace;
  readonly depthBias: { readonly constant: number; readonly slopeScale: number; readonly clamp: number } | undefined;
  readonly depthWrite: boolean | undefined;
  readonly depthCompare: GPUCompareFunction | undefined;
  readonly depthClamp: boolean | undefined;
  readonly stencil: StencilSnap | undefined;
  readonly blends: ReadonlyArray<readonly [string, BlendSnap]> | undefined;
  readonly alphaToCoverage: boolean;
}

function snapshotPipeline(
  ps: import("../core/index.js").PipelineState,
  token: AdaptiveToken,
): PipelineSnap {
  const r = ps.rasterizer;
  const depthBias = r.depthBias !== undefined ? r.depthBias.getValue(token) : undefined;
  let stencil: StencilSnap | undefined;
  if (ps.stencil !== undefined) {
    const s = ps.stencil;
    stencil = {
      enabled: s.enabled.getValue(token),
      readMask: s.readMask.getValue(token),
      writeMask: s.writeMask.getValue(token),
      front: {
        compare: s.front.compare.getValue(token),
        failOp: s.front.failOp.getValue(token),
        depthFailOp: s.front.depthFailOp.getValue(token),
        passOp: s.front.passOp.getValue(token),
      },
      back: {
        compare: s.back.compare.getValue(token),
        failOp: s.back.failOp.getValue(token),
        depthFailOp: s.back.depthFailOp.getValue(token),
        passOp: s.back.passOp.getValue(token),
      },
    };
  }
  let blends: ReadonlyArray<readonly [string, BlendSnap]> | undefined;
  if (ps.blends !== undefined) {
    const m = ps.blends.getValue(token);
    const arr: [string, BlendSnap][] = [];
    for (const [k, b] of m) {
      arr.push([k, {
        color: {
          operation: b.color.operation.getValue(token),
          srcFactor: b.color.srcFactor.getValue(token),
          dstFactor: b.color.dstFactor.getValue(token),
        },
        alpha: {
          operation: b.alpha.operation.getValue(token),
          srcFactor: b.alpha.srcFactor.getValue(token),
          dstFactor: b.alpha.dstFactor.getValue(token),
        },
        writeMask: b.writeMask.getValue(token),
      }]);
    }
    arr.sort((x, y) => (x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : 0));
    blends = arr;
  }
  return {
    topology: r.topology.getValue(token),
    cullMode: r.cullMode.getValue(token),
    frontFace: r.frontFace.getValue(token),
    depthBias,
    depthWrite: ps.depth?.write.getValue(token),
    depthCompare: ps.depth?.compare.getValue(token),
    depthClamp: ps.depth?.clamp?.getValue(token),
    stencil,
    blends,
    alphaToCoverage: ps.alphaToCoverage?.getValue(token) ?? false,
  };
}

function pipelineKeyOf(snap: PipelineSnap): string {
  return JSON.stringify(snap);
}

function colorTargetsForSnap(
  iface: ProgramInterface,
  sig: FramebufferSignature,
  blends: ReadonlyArray<readonly [string, BlendSnap]> | undefined,
): GPUColorTargetState[] {
  const out: GPUColorTargetState[] = [];
  const blendMap = blends !== undefined ? new Map<string, BlendSnap>(blends as readonly (readonly [string, BlendSnap])[]) : undefined;
  for (const o of iface.fragmentOutputs) {
    const fmt = sig.colors.tryFind(o.name);
    if (fmt === undefined) {
      throw new Error(`prepareRenderObject: fragment output "${o.name}" has no matching signature attachment`);
    }
    const target: GPUColorTargetState = { format: fmt };
    const blend = blendMap?.get(o.name);
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

function depthStencilStateForSnap(
  sig: FramebufferSignature,
  snap: PipelineSnap,
): GPUDepthStencilState | undefined {
  if (sig.depthStencil === undefined) return undefined;
  if (snap.depthCompare === undefined && snap.depthWrite === undefined && snap.stencil === undefined) return undefined;
  const out: GPUDepthStencilState = {
    format: sig.depthStencil.format,
    depthWriteEnabled: snap.depthWrite ?? false,
    depthCompare: snap.depthCompare ?? "always",
  };
  if (snap.stencil !== undefined && snap.stencil.enabled) {
    out.stencilFront = snap.stencil.front;
    out.stencilBack = snap.stencil.back;
    out.stencilReadMask = snap.stencil.readMask;
    out.stencilWriteMask = snap.stencil.writeMask;
  }
  if (snap.depthBias !== undefined) {
    out.depthBias = snap.depthBias.constant;
    out.depthBiasSlopeScale = snap.depthBias.slopeScale;
    out.depthBiasClamp = snap.depthBias.clamp;
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

// Per-handle identity counter for the bind-group cache key.
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

interface PipelineBuildContext {
  readonly device: GPUDevice;
  readonly iface: ProgramInterface;
  readonly signature: FramebufferSignature;
  readonly vertexLayouts: readonly GPUVertexBufferLayout[];
  readonly bindGroupLayouts: readonly GPUBindGroupLayout[];
  readonly vsSource: string;
  readonly fsSource: string;
  readonly vsEntry: string;
  readonly fsEntry: string;
  readonly vsSourceMap: import("@aardworx/wombat.shader/ir").SourceMap | null | undefined;
  readonly fsSourceMap: import("@aardworx/wombat.shader/ir").SourceMap | null | undefined;
  readonly label?: string | undefined;
  readonly effectId?: string | undefined;
}

function buildPipelineForSnap(
  ctx: PipelineBuildContext,
  snap: PipelineSnap,
): GPURenderPipeline {
  const colorTargets = colorTargetsForSnap(ctx.iface, ctx.signature, snap.blends);
  const ds = depthStencilStateForSnap(ctx.signature, snap);
  const multisample: GPUMultisampleState | undefined =
    ctx.signature.sampleCount > 1 || snap.alphaToCoverage
      ? {
          count: ctx.signature.sampleCount,
          ...(snap.alphaToCoverage ? { alphaToCoverageEnabled: true } : {}),
        }
      : undefined;
  const desc: CompileRenderPipelineDescription = {
    ...(ctx.label !== undefined ? { label: ctx.label } : {}),
    ...(ctx.effectId !== undefined ? { effectId: ctx.effectId } : {}),
    vertexShaderSource: ctx.vsSource,
    fragmentShaderSource: ctx.fsSource,
    ...(ctx.vsSourceMap ? { vertexSourceMap: ctx.vsSourceMap } : {}),
    ...(ctx.fsSourceMap ? { fragmentSourceMap: ctx.fsSourceMap } : {}),
    vertexEntryPoint: ctx.vsEntry,
    fragmentEntryPoint: ctx.fsEntry,
    vertexBufferLayouts: ctx.vertexLayouts,
    bindGroupLayouts: ctx.bindGroupLayouts,
    colorTargets,
    primitive: {
      topology: snap.topology,
      ...(snap.cullMode !== "none" ? { cullMode: snap.cullMode } : {}),
      frontFace: snap.frontFace,
      ...(snap.depthClamp === true ? { unclippedDepth: true } : {}),
    },
    ...(ds !== undefined ? { depthStencil: ds } : {}),
    ...(multisample !== undefined ? { multisample } : {}),
  };
  return compileRenderPipeline(ctx.device, desc);
}

export class PreparedRenderObject {
  /**
   * Most recently resolved pipeline. Set by `update(token)`. Reading
   * this before `update` was ever called yields a sentinel object
   * suitable only for identity comparison; the actual pipeline is
   * resolved on the first `update`/`record` call.
   */
  pipeline: GPURenderPipeline;
  readonly groups: readonly GroupDesc[];

  // Per-PreparedRO pipeline cache keyed by snapshot string.
  private readonly _pipelineCache = new Map<string, GPURenderPipeline>();
  private _pipelineCtx: PipelineBuildContext;
  private _pipelineState: import("../core/index.js").PipelineState;

  // Per-group bind group cache.
  private readonly _bindGroupCache: { key: string | null; group: GPUBindGroup | null }[];

  constructor(
    private readonly device: GPUDevice,
    private readonly vertexBindings: readonly VertexBindingInfo[],
    private readonly vertexBuffers: ReadonlyMap<string, AdaptiveResource<GPUBuffer>>,
    private readonly indexBuffer: AdaptiveResource<GPUBuffer> | undefined,
    private readonly indexFormat: GPUIndexFormat | undefined,
    private readonly indices: aval<import("../core/index.js").BufferView> | undefined,
    groups: readonly GroupDesc[],
    private readonly drawCall: aval<import("../core/index.js").DrawCall>,
    pipelineCtx: PipelineBuildContext,
    pipelineState: import("../core/index.js").PipelineState,
  ) {
    this.groups = groups;
    this._pipelineCtx = pipelineCtx;
    this._pipelineState = pipelineState;
    this._bindGroupCache = groups.map(() => ({ key: null, group: null }));
    // Sentinel pipeline — replaced on the first `update(token)`.
    this.pipeline = SENTINEL_PIPELINE;
  }

  /** Internal — exposed for tests. The PipelineState the RO is reading. */
  get pipelineState(): import("../core/index.js").PipelineState { return this._pipelineState; }

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

  /**
   * Resolve the pipeline for the current values of the
   * pipeline-influencing avals. Updates `this.pipeline` so that the
   * surrounding sort / record path can read it without forcing.
   */
  update(token: AdaptiveToken): void {
    const snap = snapshotPipeline(this._pipelineState, token);
    const key = pipelineKeyOf(snap);
    let pipeline = this._pipelineCache.get(key);
    if (pipeline === undefined) {
      pipeline = buildPipelineForSnap(this._pipelineCtx, snap);
      this._pipelineCache.set(key, pipeline);
    }
    this.pipeline = pipeline;
  }

  record(pass: GPURenderPassEncoder, token: AdaptiveToken): void {
    // Resolve current pipeline. If `update` was already called by the
    // walker this is a no-op cache hit; otherwise we still pick the
    // right pipeline for the token-evaluated values.
    this.update(token);
    pass.setPipeline(this.pipeline);

    // Per-frame state — does NOT influence the pipeline cache key.
    const ps = this._pipelineState;
    if (ps.blendConstant !== undefined) {
      const c = ps.blendConstant.getValue(token);
      pass.setBlendConstant(c);
    }
    if (ps.stencil !== undefined) {
      const ref = ps.stencil.reference.getValue(token);
      pass.setStencilReference(ref);
    }

    for (let gi = 0; gi < this.groups.length; gi++) {
      const g = this.groups[gi]!;
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

  // The set of attribute names is fixed structurally; only the
  // per-attribute BufferView avals are reactive. We resolve names →
  // step-mode now (vertex vs instance) from the maps directly.
  const vertexMap = obj.vertexAttributes;
  const instanceMap = obj.instanceAttributes ?? emptyAttrMap();

  const vertexBuffers = new Map<string, AdaptiveResource<GPUBuffer>>();
  const vertexLayouts: GPUVertexBufferLayout[] = [];
  for (const vb of vertexBindings) {
    const fromVertex = vertexMap.tryFind(vb.name);
    const fromInstance = instanceMap.tryFind(vb.name);
    if (fromVertex === undefined && fromInstance === undefined) {
      throw new Error(`prepareRenderObject: missing vertex attribute "${vb.name}"`);
    }
    const isInstance = fromVertex === undefined && fromInstance !== undefined;
    const stepMode: GPUVertexStepMode = isInstance ? "instance" : "vertex";

    const viewAval = (fromVertex ?? fromInstance)!;
    // Honour an explicit `stride: 0` from the BufferView for the
    // single-value-broadcast path (a 1-element vertex buffer read by
    // every vertex). Sampled at build time — switching stride
    // dynamically is rare and would require a fresh pipeline anyway.
    const initialView = viewAval.force();
    const explicitZeroStride = initialView.stride === 0;
    const stride = explicitZeroStride
      ? 0
      : (vb.byteSize !== undefined && vb.byteSize > 0
        ? vb.byteSize
        : vertexFormatStride(vb.format));

    const bufAval = viewAval.map(view => view.buffer);
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
    // Construction-time read to discover the index format. The
    // BufferView aval is expected to settle on a stable format —
    // changing index format would require a fresh PreparedRO.
    const initial = obj.indices.force();
    indexFormat = initial.indexFormat
      ?? (initial.format === "uint16" ? "uint16" : "uint32");
  }

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

  // Pipeline — set up the build context and produce the initial
  // GPURenderPipeline. We snapshot via AdaptiveToken.top here; that's
  // a one-time construction-boundary read, NOT inside an adaptive
  // computation. Subsequent recompiles happen inside `update(token)`.
  const vsStage = effect.stages.find(st => st.stage === "vertex");
  const fsStage = effect.stages.find(st => st.stage === "fragment");
  if (vsStage === undefined) throw new Error("prepareRenderObject: CompiledEffect has no vertex stage");
  if (fsStage === undefined) throw new Error("prepareRenderObject: CompiledEffect has no fragment stage");

  const pipelineCtx: PipelineBuildContext = {
    device,
    iface,
    signature,
    vertexLayouts,
    bindGroupLayouts: groups.map(g => g.layout),
    vsSource: vsStage.source,
    fsSource: fsStage.source,
    vsEntry: vsStage.entryName,
    fsEntry: fsStage.entryName,
    vsSourceMap: vsStage.sourceMap,
    fsSourceMap: fsStage.sourceMap,
    ...(opts.label !== undefined ? { label: opts.label } : {}),
    ...(opts.effectId !== undefined ? { effectId: opts.effectId } : {}),
  };

  return new PreparedRenderObject(
    device,
    vertexBindings, vertexBuffers,
    indexBuffer, indexFormat, obj.indices,
    groups,
    obj.drawCall,
    pipelineCtx,
    obj.pipelineState,
  );
}

function emptyAttrMap(): HashMap<string, aval<BufferView>> {
  return HashMap.empty<string, aval<BufferView>>();
}

/**
 * Sentinel pipeline placeholder. PreparedRenderObject's `pipeline`
 * field is non-null for ergonomic typing; before the first
 * `update(token)` it points at this sentinel which is only used as
 * a sort-rank identity (never bound to a render pass — `record`
 * always calls `update` first).
 */
const SENTINEL_PIPELINE = { __sentinel: "PreparedRenderObject.pipeline" } as unknown as GPURenderPipeline;

// ---------------------------------------------------------------------------
// Uniform input merging — `obj.uniforms` wins; `avalBindings` is fallback.
// ---------------------------------------------------------------------------

function mergeUniformInputs(
  user: HashMap<string, aval<unknown>>,
  bindings: Readonly<Record<string, () => unknown>> | undefined,
  block: import("../core/index.js").UniformBlockInfo,
): HashMap<string, aval<unknown>> {
  let merged = user;
  if (bindings === undefined) return merged;
  for (const f of block.fields) {
    if (merged.tryFind(f.name) !== undefined) continue;
    const getter = bindings[f.name];
    if (getter === undefined) continue;
    const v = getter();
    if (isAval(v)) merged = merged.add(f.name, v as aval<unknown>);
    else merged = merged.add(f.name, cval(v));
  }
  return merged;
}

function isAval(v: unknown): boolean {
  return typeof v === "object" && v !== null && typeof (v as { getValue?: unknown }).getValue === "function";
}

function ubAsBlock(ub: import("../core/index.js").UniformBlockInfo): import("../core/index.js").UniformBlockInfo {
  return ub;
}
