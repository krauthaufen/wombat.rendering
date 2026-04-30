// prepareRenderObject — turn a `RenderObject` + `CompiledEffect`
// + `FramebufferSignature` into a runnable `PreparedRenderObject`.
//
// Slice landed in this revision:
//   - Vertex attributes by name.
//   - Uniform buffers (one per `ProgramInterface.uniformBuffers`).
//   - Textures + samplers (one bind-group entry per
//     `ProgramInterface.textures` / `.samplers`).
//   - Storage buffers (one bind-group entry per
//     `ProgramInterface.storageBuffers`; user passes the source as
//     `aval<IBuffer>`, runtime wraps via `prepareAdaptiveBuffer`).
//   - Index buffer / draw call.
//   - Pipeline (cached).
//
// Not yet handled (see TODO.md):
//   - Instance attributes / per-attribute stepMode.
//   - Blend / stencil / multisample plumbing.
//   - `BufferView.indexFormat` (currently hardcoded `uint32`).

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
} from "@aardworx/wombat.adaptive";
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
}

function vertexBufferLayoutsFor(iface: ProgramInterface): VertexBindingInfo[] {
  return iface.vertexAttributes.map(a => ({ name: a.name, slot: a.location, format: a.format }));
}

function colorTargetsFor(iface: ProgramInterface, sig: FramebufferSignature): GPUColorTargetState[] {
  const out: GPUColorTargetState[] = [];
  for (const o of iface.fragmentOutputs) {
    const fmt = sig.colors.tryFind(o.name);
    if (fmt === undefined) {
      throw new Error(`prepareRenderObject: fragment output "${o.name}" has no matching signature attachment`);
    }
    out[o.location] = { format: fmt };
  }
  return out;
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

// ---------------------------------------------------------------------------
// PreparedRenderObject
// ---------------------------------------------------------------------------

export class PreparedRenderObject {
  readonly pipeline: GPURenderPipeline;
  readonly groups: readonly GroupDesc[];

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

  /** Encode this object into an open render pass. */
  record(pass: GPURenderPassEncoder, token: AdaptiveToken): void {
    pass.setPipeline(this.pipeline);

    for (const g of this.groups) {
      const entries: GPUBindGroupEntry[] = g.entries.map(e => {
        switch (e.kind) {
          case "ubuf":
          case "sbuf":
            return { binding: e.binding, resource: { buffer: e.resource.getValue(token) } };
          case "tex":
            return { binding: e.binding, resource: e.resource.getValue(token).createView() };
          case "sampler":
            return { binding: e.binding, resource: e.resource.getValue(token) };
        }
      });
      const bg = this.device.createBindGroup({ layout: g.layout, entries });
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
}

export function prepareRenderObject(
  device: GPUDevice,
  obj: RenderObject,
  effect: CompiledEffect,
  signature: FramebufferSignature,
  opts: PrepareRenderObjectOptions = {},
): PreparedRenderObject {
  const iface = effect.interface;
  const vertexBindings = vertexBufferLayoutsFor(iface);

  // Vertex buffers.
  const vertexBuffers = new Map<string, AdaptiveResource<GPUBuffer>>();
  const vertexLayouts: GPUVertexBufferLayout[] = [];
  for (const vb of vertexBindings) {
    const av = obj.vertexAttributes.tryFind(vb.name);
    if (av === undefined) throw new Error(`prepareRenderObject: missing vertex attribute "${vb.name}"`);
    const stride = vertexFormatStride(vb.format);
    const bufAval = av.map(view => view.buffer);
    const res = prepareAdaptiveBuffer(device, bufAval, {
      usage: BufferUsage.VERTEX,
      ...(opts.label !== undefined ? { label: `${opts.label}.${vb.name}` } : {}),
    });
    vertexBuffers.set(vb.name, res);
    vertexLayouts[vb.slot] = {
      arrayStride: stride,
      attributes: [{ shaderLocation: vb.slot, offset: 0, format: vb.format }],
    };
  }

  // Index buffer.
  let indexBuffer: AdaptiveResource<GPUBuffer> | undefined;
  let indexFormat: GPUIndexFormat | undefined;
  if (obj.indices !== undefined) {
    const bufAval = obj.indices.map(v => v.buffer);
    indexBuffer = prepareAdaptiveBuffer(device, bufAval, {
      usage: BufferUsage.INDEX,
      ...(opts.label !== undefined ? { label: `${opts.label}.indices` } : {}),
    });
    indexFormat = "uint32"; // TODO: pull from BufferView once `indexFormat` field lands
  }

  // Per-group entry collection.
  const maxGroup = Math.max(
    -1,
    ...iface.uniformBuffers.map(u => u.group),
    ...iface.samplers.map(s => s.group),
    ...iface.textures.map(t => t.group),
    ...iface.storageBuffers.map(s => s.group),
  );
  const perGroup: EntryDesc[][] = [];
  for (let g = 0; g <= maxGroup; g++) perGroup.push([]);

  for (const ub of iface.uniformBuffers) {
    const res = prepareUniformBuffer(device, ub, obj.uniforms, {
      ...(opts.label !== undefined ? { label: `${opts.label}.${ub.name}` } : {}),
    });
    perGroup[ub.group]!.push({ kind: "ubuf", binding: ub.binding, resource: res });
  }

  for (const t of iface.textures) {
    const av = obj.textures.tryFind(t.name);
    if (av === undefined) throw new Error(`prepareRenderObject: missing texture "${t.name}"`);
    const res = prepareAdaptiveTexture(device, av, {
      ...(opts.label !== undefined ? { label: `${opts.label}.${t.name}` } : {}),
    });
    perGroup[t.group]!.push({ kind: "tex", binding: t.binding, resource: res, sampleType: t.sampleType });
  }

  for (const s of iface.samplers) {
    const av = obj.samplers.tryFind(s.name);
    if (av === undefined) throw new Error(`prepareRenderObject: missing sampler "${s.name}"`);
    const res = prepareAdaptiveSampler(device, av);
    perGroup[s.group]!.push({ kind: "sampler", binding: s.binding, resource: res });
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
    perGroup[sb.group]!.push({ kind: "sbuf", binding: sb.binding, resource: res, access: sb.access });
  }

  const groups = buildGroups(device, perGroup);

  // Pipeline.
  const colorTargets = colorTargetsFor(iface, signature);
  const vsStage = effect.stages.find(s => s.stage === "vertex");
  const fsStage = effect.stages.find(s => s.stage === "fragment");
  if (vsStage === undefined) throw new Error("prepareRenderObject: CompiledEffect has no vertex stage");
  if (fsStage === undefined) throw new Error("prepareRenderObject: CompiledEffect has no fragment stage");

  const pipelineDesc: CompileRenderPipelineDescription = {
    ...(opts.label !== undefined ? { label: opts.label } : {}),
    vertexShaderSource: vsStage.source,
    fragmentShaderSource: fsStage.source,
    vertexEntryPoint: "main",
    fragmentEntryPoint: "main",
    vertexBufferLayouts: vertexLayouts,
    bindGroupLayouts: groups.map(g => g.layout),
    colorTargets,
    primitive: {
      topology: obj.pipelineState.rasterizer.topology,
      ...(obj.pipelineState.rasterizer.cullMode !== "none" ? { cullMode: obj.pipelineState.rasterizer.cullMode } : {}),
      frontFace: obj.pipelineState.rasterizer.frontFace,
    },
    ...(signature.depthStencil !== undefined && obj.pipelineState.depth !== undefined
      ? { depthStencil: {
            format: signature.depthStencil.format,
            depthWriteEnabled: obj.pipelineState.depth.write,
            depthCompare: obj.pipelineState.depth.compare,
          } }
      : {}),
    ...(signature.sampleCount > 1 ? { multisample: { count: signature.sampleCount } } : {}),
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
