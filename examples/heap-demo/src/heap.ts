// heap.ts — Phase 2 of the heap-everything architecture, prototyped
// in the demo before being ported into wombat.rendering.
//
//   Phase 1 (prior commit): one shared GPURenderPipeline + bind
//     group, per-draw uniforms in a storage buffer, firstInstance
//     routing into the heap. Per-RO vertex / index / color buffers.
//
//   Phase 2 (this file): vertex pulling. Positions and normals come
//     from packed storage buffers (one for each attribute). The
//     per-RO color folds into the heap entry. Indices for all draws
//     concatenate into one shared GPUBuffer; the per-draw firstIndex
//     selects each RO's range. Net result: ZERO vertex buffers, ONE
//     index buffer, THREE storage-buffer bindings, N drawIndexed
//     calls — all sharing one pipeline + one bind group.
//
// What stays from Phase 1:
//   - drawIdx routing via @builtin(instance_index) (= firstInstance,
//     since instanceCount=1).
//   - wombat.shader matrix convention: row-major upload, WGSL uses
//     row-vec dual.

import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import type { aval } from "@aardworx/wombat.adaptive";
import type { CanvasAttachment } from "@aardworx/wombat.rendering.experimental/window";
import type { GeometryData } from "./geometry.js";

// ---------------------------------------------------------------------------
// DrawHeader layout (std430, 16-byte aligned)
// ---------------------------------------------------------------------------
//
//   struct DrawHeader {
//     modelTrafo:    mat4x4<f32>,   // 64
//     modelTrafoInv: mat4x4<f32>,   // 64
//     viewProjTrafo: mat4x4<f32>,   // 64
//     lightLocation: vec4<f32>,     // 16  (.xyz used, .w padding)
//     color:         vec4<f32>,     // 16
//     vertexBase:    u32,           //  4
//     _pad:          vec3<u32>,     // 12  (align next entry to 16)
//   };  // 240 bytes total
//
// Tighter layouts are possible (collapse pad with vertexBase if we
// ever add more u32s) — leave room here for future per-draw scalars
// (PickId, Time, …).

const DRAW_HEADER_BYTES = 64 * 3 + 16 + 16 + 16;
const DRAW_HEADER_FLOATS = DRAW_HEADER_BYTES / 4;

function packMat44(m: import("@aardworx/wombat.base").M44d, dst: Float32Array, off: number): void {
  const r = m._data;
  for (let i = 0; i < 16; i++) dst[off + i] = r[i]!;
}

function packDrawHeader(
  modelTrafo: Trafo3d,
  viewProj: Trafo3d,
  lightLocation: V3d,
  color: V4f,
  vertexBase: number,
  dst: Float32Array,
  drawIndex: number,
): void {
  const off = drawIndex * DRAW_HEADER_FLOATS;
  packMat44(modelTrafo.forward,  dst, off +  0);
  packMat44(modelTrafo.backward, dst, off + 16);
  packMat44(viewProj.forward,    dst, off + 32);
  dst[off + 48] = lightLocation.x as number;
  dst[off + 49] = lightLocation.y as number;
  dst[off + 50] = lightLocation.z as number;
  dst[off + 51] = 0;
  dst[off + 52] = color.x;
  dst[off + 53] = color.y;
  dst[off + 54] = color.z;
  dst[off + 55] = color.w;
  // u32 vertexBase via the underlying buffer view.
  new Uint32Array(dst.buffer, dst.byteOffset, dst.length)[off + 56] = vertexBase;
  // [off + 57..59] left as pad.
}

// ---------------------------------------------------------------------------
// Hand-rolled WGSL — vertex pull positions + normals from storage,
// no vertex buffers. Color comes from the per-draw heap entry.
// ---------------------------------------------------------------------------

const SHADER_WGSL = /* wgsl */`
struct DrawHeader {
  modelTrafo:    mat4x4<f32>,
  modelTrafoInv: mat4x4<f32>,
  viewProjTrafo: mat4x4<f32>,
  lightLocation: vec4<f32>,
  color:         vec4<f32>,
  vertexBase:    u32,
  _pad0:         u32,
  _pad1:         u32,
  _pad2:         u32,
};

@group(0) @binding(0) var<storage, read> draws:     array<DrawHeader>;
@group(0) @binding(1) var<storage, read> positions: array<f32>;     // 3 floats per vertex
@group(0) @binding(2) var<storage, read> normals:   array<f32>;     // 3 floats per vertex

struct VsOut {
  @builtin(position) clipPos:  vec4<f32>,
  @location(0)       worldPos: vec3<f32>,
  @location(1)       normal:   vec3<f32>,
  @location(2)       color:    vec4<f32>,
  @location(3)       lightLoc: vec3<f32>,
};

fn fetchVec3(buf: ptr<storage, array<f32>, read>, base: u32) -> vec3<f32> {
  return vec3<f32>((*buf)[base + 0u], (*buf)[base + 1u], (*buf)[base + 2u]);
}

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) drawIdx: u32) -> VsOut {
  let d = draws[drawIdx];
  let base = (d.vertexBase + vid) * 3u;
  let pos = fetchVec3(&positions, base);
  let nor = fetchVec3(&normals,   base);
  // wombat.shader convention: matrices uploaded row-major; WGSL reads
  // them as col-major = transposed. v * M is the row-vec dual of
  // M.mul(v) (= M*v math).
  let wp = vec4<f32>(pos, 1.0) * d.modelTrafo;
  let n  = (vec4<f32>(nor, 0.0) * d.modelTrafo).xyz;
  var out: VsOut;
  out.clipPos  = wp * d.viewProjTrafo;
  out.worldPos = wp.xyz;
  out.normal   = n;
  out.color    = d.color;
  out.lightLoc = d.lightLocation.xyz;
  return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let n = normalize(in.normal);
  let l = normalize(in.lightLoc - in.worldPos);
  let ambient = 0.2;
  let diffuse = abs(dot(l, n));
  let k = ambient + (1.0 - ambient) * diffuse;
  return vec4<f32>(in.color.xyz * k, in.color.w);
}
`;

// ---------------------------------------------------------------------------
// Geometry heap — pack all draws' positions, normals, indices into
// three big buffers, and remember the per-draw offsets.
// ---------------------------------------------------------------------------

interface PackedGeometry {
  /** Storage: tightly-packed Float32 positions across all draws. */
  readonly positionsBuf: GPUBuffer;
  /** Storage: tightly-packed Float32 normals across all draws. */
  readonly normalsBuf:   GPUBuffer;
  /** Index: concatenated Uint32 indices across all draws. */
  readonly indexBuf:     GPUBuffer;
  /** Per-draw [vertexBase, indexCount, firstIndex]. */
  readonly entries:      ReadonlyArray<{ vertexBase: number; indexCount: number; firstIndex: number }>;
}

function packGeometry(device: GPUDevice, geos: readonly GeometryData[]): PackedGeometry {
  let totalVerts = 0, totalIndices = 0;
  for (const g of geos) {
    totalVerts   += g.positions.length / 3;
    totalIndices += g.indices.length;
  }
  const positions = new Float32Array(totalVerts * 3);
  const normals   = new Float32Array(totalVerts * 3);
  const indices   = new Uint32Array(totalIndices);
  const entries: { vertexBase: number; indexCount: number; firstIndex: number }[] = [];

  let vOff = 0, iOff = 0;
  for (const g of geos) {
    const vCount = g.positions.length / 3;
    positions.set(g.positions, vOff * 3);
    normals.set(g.normals,     vOff * 3);
    indices.set(g.indices, iOff);
    entries.push({ vertexBase: vOff, indexCount: g.indices.length, firstIndex: iOff });
    vOff += vCount;
    iOff += g.indices.length;
  }

  const mk = (data: ArrayBufferView, usage: GPUBufferUsageFlags, label: string): GPUBuffer => {
    const buf = device.createBuffer({
      size: alignUp(data.byteLength, 4),
      usage: usage | GPUBufferUsage.COPY_DST,
      label,
    });
    device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
    return buf;
  };
  return {
    positionsBuf: mk(positions, GPUBufferUsage.STORAGE, "heap-demo: positions"),
    normalsBuf:   mk(normals,   GPUBufferUsage.STORAGE, "heap-demo: normals"),
    indexBuf:     mk(indices,   GPUBufferUsage.INDEX,   "heap-demo: indices"),
    entries,
  };
}

function alignUp(n: number, a: number): number { return (n + a - 1) & ~(a - 1); }

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface HeapDrawSpec {
  readonly geo: GeometryData;
  readonly modelTrafo: Trafo3d;
  readonly color: V4f;
}

export interface HeapRenderer {
  /** Render one frame; called from the runFrame callback. */
  frame(viewProj: Trafo3d, lightLocation: V3d): void;
  dispose(): void;
}

export function buildHeapRenderer(
  device: GPUDevice,
  attach: CanvasAttachment,
  draws: readonly HeapDrawSpec[],
): HeapRenderer {
  const sig = attach.signature;
  const colorFormat = sig.colors.tryFind("outColor");
  if (colorFormat === undefined) throw new Error("buildHeapRenderer: attachment has no 'outColor'");
  const depthFormat = sig.depthStencil?.format;

  // ---- Geometry heap ----
  const packed = packGeometry(device, draws.map(d => d.geo));

  // ---- Draw header heap (storage) ----
  const heapBytes = DRAW_HEADER_BYTES * draws.length;
  const heap = device.createBuffer({
    size: alignUp(heapBytes, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "heap-demo: draws",
  });
  const heapStaging = new Float32Array(DRAW_HEADER_FLOATS * draws.length);

  // ---- Pipeline ----
  const module = device.createShaderModule({ code: SHADER_WGSL, label: "heap-demo: shader" });
  const bindGroupLayout = device.createBindGroupLayout({
    label: "heap-demo: bgl",
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const pipeline = device.createRenderPipeline({
    label: "heap-demo: pipeline",
    layout: pipelineLayout,
    vertex:   { module, entryPoint: "vs", buffers: [] },
    fragment: { module, entryPoint: "fs", targets: [{ format: colorFormat }] },
    primitive: { topology: "triangle-list", cullMode: "back", frontFace: "ccw" },
    depthStencil: depthFormat !== undefined
      ? { format: depthFormat, depthWriteEnabled: true, depthCompare: "less" }
      : undefined,
  });

  const bindGroup = device.createBindGroup({
    label: "heap-demo: bg",
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: heap } },
      { binding: 1, resource: { buffer: packed.positionsBuf } },
      { binding: 2, resource: { buffer: packed.normalsBuf } },
    ],
  });

  function frame(viewProj: Trafo3d, lightLocation: V3d): void {
    // Pack the draw header heap.
    for (let i = 0; i < draws.length; i++) {
      const d = draws[i]!;
      packDrawHeader(d.modelTrafo, viewProj, lightLocation, d.color,
                     packed.entries[i]!.vertexBase, heapStaging, i);
    }
    device.queue.writeBuffer(heap, 0, heapStaging.buffer, heapStaging.byteOffset, heapBytes);

    const fb = forceFramebuffer(attach);
    const colorView = fb.colors.tryFind("outColor")!;
    const depthView = fb.depthStencil;

    const enc = device.createCommandEncoder({ label: "heap-demo: encoder" });
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: colorView,
        clearValue: { r: 0.07, g: 0.07, b: 0.08, a: 1.0 },
        loadOp: "clear", storeOp: "store",
      }],
      ...(depthView !== undefined && depthFormat !== undefined ? {
        depthStencilAttachment: {
          view: depthView,
          depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
        } satisfies GPURenderPassDepthStencilAttachment,
      } : {}),
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setIndexBuffer(packed.indexBuf, "uint32");

    for (let i = 0; i < draws.length; i++) {
      const e = packed.entries[i]!;
      // firstInstance = i routes the VS to draws[i]; baseVertex = 0
      // because we offset via d.vertexBase inside the shader.
      pass.drawIndexed(e.indexCount, 1, e.firstIndex, 0, i);
    }

    pass.end();
    device.queue.submit([enc.finish()]);
  }

  function dispose(): void {
    heap.destroy();
    packed.positionsBuf.destroy();
    packed.normalsBuf.destroy();
    packed.indexBuf.destroy();
  }

  return { frame, dispose };
}

// ---------------------------------------------------------------------------
// Internal — force the canvas attachment's framebuffer aval (we
// bypass the runtime here, so we read it ourselves once per frame).
// ---------------------------------------------------------------------------

function forceFramebuffer(attach: CanvasAttachment): import("@aardworx/wombat.rendering.experimental/core").IFramebuffer {
  return (attach.framebuffer as aval<import("@aardworx/wombat.rendering.experimental/core").IFramebuffer>)
    .force(/* allow-force */);
}
