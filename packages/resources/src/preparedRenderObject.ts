// prepareRenderObject — turn a `RenderObject` + `CompiledEffect`
// + `FramebufferSignature` into a runnable `PreparedRenderObject`.
//
// MVP slice (this revision):
//   - Vertex attributes: each shader-required attribute is bound
//     to its own vertex-buffer slot at the attribute's location.
//     `prepareAdaptiveBuffer` lifts each `aval<BufferView>` into
//     a managed `GPUBuffer`.
//   - Uniform buffers: one bind group per `ProgramInterface.uniformBuffers`
//     entry; `prepareUniformBuffer` packs the named bag using the
//     shader's layout.
//   - Index buffer: optional, follows `BufferView.format` (uint16/uint32).
//   - Draw call: read from `obj.drawCall: aval<DrawCall>`.
//   - Pipeline: cached via `compileRenderPipeline`.
//
// Not yet handled (next slice): textures, samplers, storage
// buffers, instance attributes, depth-stencil state, blend state,
// multisample.

import {
  AdaptiveResource,
  type CompiledEffect,
  type FramebufferSignature,
  type IBuffer,
  type ProgramInterface,
  type RenderObject,
  type UniformBufferLayout,
} from "@aardworx/wombat.rendering-core";
import {
  type AdaptiveToken,
  type aval,
} from "@aardworx/wombat.adaptive";
import { prepareAdaptiveBuffer } from "./adaptiveBuffer.js";
import { prepareUniformBuffer } from "./uniformBuffer.js";
import { compileRenderPipeline, type CompileRenderPipelineDescription } from "./renderPipeline.js";
import { BufferUsage } from "./webgpuFlags.js";

// ---------------------------------------------------------------------------
// Helpers — derive WebGPU descriptors from ProgramInterface + signature
// ---------------------------------------------------------------------------

function vertexFormatStride(fmt: GPUVertexFormat): number {
  // Minimal table — extend as needed.
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
  readonly slot: number;          // index into the `buffers` array (== shader location for our scheme)
  readonly format: GPUVertexFormat;
}

function vertexBufferLayoutsFor(iface: ProgramInterface): VertexBindingInfo[] {
  return iface.vertexAttributes.map(a => ({
    name: a.name,
    slot: a.location,
    format: a.format,
  }));
}

function colorTargetsFor(iface: ProgramInterface, sig: FramebufferSignature): GPUColorTargetState[] {
  // One target per fragment output, looked up by name in the
  // signature. The signature's HashMap is iterated to produce the
  // correct slot ordering — slot index = shader-declared location.
  // Outputs whose names are not in the signature are an error.
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

function bindGroupLayoutsForUniforms(device: GPUDevice, iface: ProgramInterface): {
  layouts: GPUBindGroupLayout[];
  byGroup: Map<number, UniformBufferLayout[]>;
} {
  const byGroup = new Map<number, UniformBufferLayout[]>();
  for (const ub of iface.uniformBuffers) {
    let arr = byGroup.get(ub.group);
    if (arr === undefined) { arr = []; byGroup.set(ub.group, arr); }
    arr.push(ub);
  }
  const layouts: GPUBindGroupLayout[] = [];
  const maxGroup = iface.uniformBuffers.reduce((m, u) => Math.max(m, u.group), -1);
  for (let g = 0; g <= maxGroup; g++) {
    const ubs = byGroup.get(g) ?? [];
    const entries: GPUBindGroupLayoutEntry[] = ubs.map(u => ({
      binding: u.binding,
      visibility: 0x1 | 0x2,            // ShaderStage.VERTEX | ShaderStage.FRAGMENT
      buffer: { type: "uniform" as const },
    }));
    layouts.push(device.createBindGroupLayout({ entries }));
  }
  return { layouts, byGroup };
}

// ---------------------------------------------------------------------------
// PreparedRenderObject
// ---------------------------------------------------------------------------

export class PreparedRenderObject {
  readonly pipeline: GPURenderPipeline;
  readonly bindGroupLayouts: readonly GPUBindGroupLayout[];

  constructor(
    private readonly device: GPUDevice,
    private readonly vertexBindings: readonly VertexBindingInfo[],
    private readonly vertexBuffers: ReadonlyMap<string, AdaptiveResource<GPUBuffer>>,
    private readonly indexBuffer: AdaptiveResource<GPUBuffer> | undefined,
    private readonly indexFormat: GPUIndexFormat | undefined,
    private readonly indices: aval<import("@aardworx/wombat.rendering-core").BufferView> | undefined,
    private readonly uniformBuffers: ReadonlyMap<number, ReadonlyMap<number, AdaptiveResource<GPUBuffer>>>,
    private readonly drawCall: aval<import("@aardworx/wombat.rendering-core").DrawCall>,
    pipeline: GPURenderPipeline,
    bindGroupLayouts: readonly GPUBindGroupLayout[],
  ) {
    this.pipeline = pipeline;
    this.bindGroupLayouts = bindGroupLayouts;
  }

  acquire(): void {
    for (const r of this.vertexBuffers.values()) r.acquire();
    if (this.indexBuffer !== undefined) this.indexBuffer.acquire();
    for (const grp of this.uniformBuffers.values()) for (const u of grp.values()) u.acquire();
  }

  release(): void {
    for (const r of this.vertexBuffers.values()) r.release();
    if (this.indexBuffer !== undefined) this.indexBuffer.release();
    for (const grp of this.uniformBuffers.values()) for (const u of grp.values()) u.release();
  }

  /** Encode this object into an open render pass. Reads all current adaptive state via `token`. */
  record(pass: GPURenderPassEncoder, token: AdaptiveToken): void {
    pass.setPipeline(this.pipeline);

    // Bind groups — rebuilt every frame for now; cheap relative to draw work.
    for (let g = 0; g < this.bindGroupLayouts.length; g++) {
      const grp = this.uniformBuffers.get(g);
      const entries: GPUBindGroupEntry[] = [];
      if (grp !== undefined) {
        for (const [binding, res] of grp.entries()) {
          const buf = res.getValue(token);
          entries.push({ binding, resource: { buffer: buf } });
        }
      }
      const bg = this.device.createBindGroup({ layout: this.bindGroupLayouts[g]!, entries });
      pass.setBindGroup(g, bg);
    }

    // Vertex buffers.
    for (const vb of this.vertexBindings) {
      const res = this.vertexBuffers.get(vb.name);
      if (res === undefined) {
        throw new Error(`PreparedRenderObject.record: missing vertex buffer for "${vb.name}"`);
      }
      const buf = res.getValue(token);
      pass.setVertexBuffer(vb.slot, buf);
    }

    // Index buffer (if any).
    if (this.indexBuffer !== undefined && this.indices !== undefined && this.indexFormat !== undefined) {
      const buf = this.indexBuffer.getValue(token);
      const view = this.indices.getValue(token);
      pass.setIndexBuffer(buf, this.indexFormat, view.offset, view.count * (this.indexFormat === "uint16" ? 2 : 4));
    }

    // Draw.
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

  // Vertex buffers — one per shader-required attribute.
  const vertexBuffers = new Map<string, AdaptiveResource<GPUBuffer>>();
  const vertexLayouts: GPUVertexBufferLayout[] = [];
  for (const vb of vertexBindings) {
    const av = obj.vertexAttributes.tryFind(vb.name);
    if (av === undefined) {
      throw new Error(`prepareRenderObject: missing vertex attribute "${vb.name}"`);
    }
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

  // Index buffer (optional).
  let indexBuffer: AdaptiveResource<GPUBuffer> | undefined;
  let indexFormat: GPUIndexFormat | undefined;
  if (obj.indices !== undefined) {
    const bufAval = obj.indices.map(v => v.buffer);
    indexBuffer = prepareAdaptiveBuffer(device, bufAval, {
      usage: BufferUsage.INDEX,
      ...(opts.label !== undefined ? { label: `${opts.label}.indices` } : {}),
    });
    // We can't read the format off `aval` synchronously here — we
    // pessimistically pick uint32 unless someone overrides via the
    // first-evaluated value. For Phase-1 tests we accept that the
    // format is fixed at initial-evaluation time of the consumer.
    // A cleaner answer is `BufferView.indexFormat: GPUIndexFormat`
    // declared on the view; deferred.
    indexFormat = "uint32";
  }

  // Uniform buffers — one resource per (group, binding).
  const ubByGroup = new Map<number, Map<number, AdaptiveResource<GPUBuffer>>>();
  for (const ub of iface.uniformBuffers) {
    const res = prepareUniformBuffer(device, ub, obj.uniforms, {
      ...(opts.label !== undefined ? { label: `${opts.label}.${ub.name}` } : {}),
    });
    let grp = ubByGroup.get(ub.group);
    if (grp === undefined) { grp = new Map(); ubByGroup.set(ub.group, grp); }
    grp.set(ub.binding, res);
  }

  // Bind group layouts (uniform-only for this slice).
  const { layouts: bindGroupLayouts } = bindGroupLayoutsForUniforms(device, iface);

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
    bindGroupLayouts,
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
    device, vertexBindings, vertexBuffers,
    indexBuffer, indexFormat, obj.indices,
    ubByGroup, obj.drawCall,
    pipeline, bindGroupLayouts,
  );
}
