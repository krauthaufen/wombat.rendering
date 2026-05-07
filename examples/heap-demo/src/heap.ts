// heap.ts — Phase 3: split heap.
//
//   Globals (uniform buffer): viewProj, lightLocation. Re-uploaded
//     every frame (camera moves) but small (≤ 96 bytes regardless
//     of draw count).
//   Per-draw heap (storage buffer): modelTrafo, modelTrafoInv,
//     color, vertexBase. Uploaded ONCE at construction; only
//     re-uploaded if a specific draw's transform / color changes —
//     adaptive slot writers can target the precise byte range.
//
// Phase 1 packed everything into per-draw entries (240 B × N each
// frame). Phase 2 added vertex pulling. Phase 3 splits per-frame
// state from per-draw state — the same shape every real renderer
// converges on, and exactly what makes the adaptive-slot story
// crisp: globals cost is independent of N; per-draw cost is paid
// only when an aval marks.

import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import type { aval, IDisposable } from "@aardworx/wombat.adaptive";
import { AVal, addMarkingCallback } from "@aardworx/wombat.adaptive";
import type { CanvasAttachment } from "@aardworx/wombat.rendering.experimental/window";
import type { GeometryData } from "./geometry.js";

// ---------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------
//
// struct Globals {              // uniform buffer, 16-byte aligned
//   viewProj:      mat4x4<f32>, // 64
//   lightLocation: vec4<f32>,   // 16
// };  // 80 B
//
// struct DrawHeader {            // storage buffer, std430
//   modelTrafo:    mat4x4<f32>,  // 64
//   modelTrafoInv: mat4x4<f32>,  // 64
//   color:         vec4<f32>,    // 16
//   vertexBase:    u32,          //  4
//   _pad0,_pad1,_pad2: u32,      // 12  (align next to 16)
// };  // 160 B

const GLOBALS_BYTES = 64 + 16;
const GLOBALS_FLOATS = GLOBALS_BYTES / 4;
const DRAW_HEADER_BYTES = 64 * 2 + 16 + 16;
const DRAW_HEADER_FLOATS = DRAW_HEADER_BYTES / 4;

function packMat44(m: import("@aardworx/wombat.base").M44d, dst: Float32Array, off: number): void {
  const r = m._data;
  for (let i = 0; i < 16; i++) dst[off + i] = r[i]!;
}

function packGlobals(
  viewProj: Trafo3d,
  lightLocation: V3d,
  dst: Float32Array,
): void {
  packMat44(viewProj.forward, dst, 0);
  dst[16] = lightLocation.x as number;
  dst[17] = lightLocation.y as number;
  dst[18] = lightLocation.z as number;
  dst[19] = 0;
}

function packDrawHeader(
  modelTrafo: Trafo3d,
  color: V4f,
  vertexBase: number,
  dst: Float32Array,
  drawIndex: number,
): void {
  const off = drawIndex * DRAW_HEADER_FLOATS;
  packMat44(modelTrafo.forward,  dst, off +  0);
  packMat44(modelTrafo.backward, dst, off + 16);
  dst[off + 32] = color.x;
  dst[off + 33] = color.y;
  dst[off + 34] = color.z;
  dst[off + 35] = color.w;
  // u32 vertexBase via the underlying buffer view.
  new Uint32Array(dst.buffer, dst.byteOffset, dst.length)[off + 36] = vertexBase;
  // [off + 37..39] left as pad.
}

// ---------------------------------------------------------------------------
// WGSL — pulls geometry from storage; reads globals + per-draw header
// per vertex.
// ---------------------------------------------------------------------------

const SHADER_WGSL = /* wgsl */`
struct Globals {
  viewProj:      mat4x4<f32>,
  lightLocation: vec4<f32>,
};

struct DrawHeader {
  modelTrafo:    mat4x4<f32>,
  modelTrafoInv: mat4x4<f32>,
  color:         vec4<f32>,
  vertexBase:    u32,
  _pad0:         u32,
  _pad1:         u32,
  _pad2:         u32,
};

@group(0) @binding(0) var<uniform>             globals:   Globals;
@group(0) @binding(1) var<storage, read>       draws:     array<DrawHeader>;
@group(0) @binding(2) var<storage, read>       positions: array<f32>;
@group(0) @binding(3) var<storage, read>       normals:   array<f32>;

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
  // wombat.shader convention: matrices uploaded row-major; WGSL
  // reads them as col-major = transposed. v * M is the row-vec dual
  // of M.mul(v) (= M*v math).
  let wp = vec4<f32>(pos, 1.0) * d.modelTrafo;
  let n  = (vec4<f32>(nor, 0.0) * d.modelTrafo).xyz;
  var out: VsOut;
  out.clipPos  = wp * globals.viewProj;
  out.worldPos = wp.xyz;
  out.normal   = n;
  out.color    = d.color;
  out.lightLoc = globals.lightLocation.xyz;
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
// Geometry heap (unchanged from Phase 2)
// ---------------------------------------------------------------------------

interface PackedGeometry {
  readonly positionsBuf: GPUBuffer;
  readonly normalsBuf:   GPUBuffer;
  readonly indexBuf:     GPUBuffer;
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
  /** Per-draw model transform; mark fires → slot's matrix block re-uploaded. */
  readonly modelTrafo: aval<Trafo3d> | Trafo3d;
  /** Per-draw colour broadcast; mark fires → slot's colour bytes re-uploaded. */
  readonly color: aval<V4f> | V4f;
}

export interface HeapStats {
  /** Bytes uploaded to globals (every frame). */
  readonly globalsBytes: number;
  /** Bytes uploaded to per-draw heap on this frame. */
  drawBytes: number;
  /** Bytes uploaded to geometry heap (positions + normals + indices) total at construction. */
  readonly geometryBytes: number;
}

export interface HeapRenderer {
  /** Render one frame. */
  frame(viewProj: Trafo3d, lightLocation: V3d): void;
  /** Mark draw `i` dirty so its header is re-uploaded next frame.
   *  Used by the demo's slot-writer test; in production the
   *  rendering layer would subscribe to per-aval marks. */
  markDirty(drawIndex: number): void;
  /** Upload counters; updated each `frame()`. */
  readonly stats: HeapStats;
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

  const packed = packGeometry(device, draws.map(d => d.geo));

  // Lift per-draw inputs to avals so the rest of the renderer is
  // shape-uniform — caller may pass either a plain value or an aval.
  function asAval<T>(v: aval<T> | T): aval<T> {
    return (typeof v === "object" && v !== null && typeof (v as { getValue?: unknown }).getValue === "function")
      ? (v as aval<T>)
      : AVal.constant(v as T);
  }
  const modelTrafos: aval<Trafo3d>[] = draws.map(d => asAval(d.modelTrafo));
  const colors:      aval<V4f>[]    = draws.map(d => asAval(d.color));

  // ---- Globals UBO ----
  const globalsBuf = device.createBuffer({
    size: alignUp(GLOBALS_BYTES, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: "heap-demo: globals",
  });
  const globalsStaging = new Float32Array(GLOBALS_FLOATS);

  // ---- Per-draw heap (storage). Pack ONCE at construction. ----
  const heapBytes = DRAW_HEADER_BYTES * draws.length;
  const drawHeap = device.createBuffer({
    size: alignUp(heapBytes, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "heap-demo: draws",
  });
  const drawStaging = new Float32Array(DRAW_HEADER_FLOATS * draws.length);
  for (let i = 0; i < draws.length; i++) {
    packDrawHeader(
      modelTrafos[i]!.force(/* allow-force */),
      colors[i]!.force(/* allow-force */),
      packed.entries[i]!.vertexBase, drawStaging, i,
    );
  }
  device.queue.writeBuffer(drawHeap, 0, drawStaging.buffer, drawStaging.byteOffset, heapBytes);

  // Slot-writer dirty set — coalesces consecutive marks into a
  // single writeBuffer per range. For the demo this is just a Set;
  // the production version would coalesce into byte ranges.
  const dirty = new Set<number>();

  // Subscribe to each per-draw aval so marks propagate to `dirty`.
  // addMarkingCallback fires once per mark cycle; pulling the new
  // value happens lazily during the next frame() pack.
  const subs: IDisposable[] = [];
  for (let i = 0; i < draws.length; i++) {
    subs.push(addMarkingCallback(modelTrafos[i] as never, () => dirty.add(i)));
    subs.push(addMarkingCallback(colors[i]      as never, () => dirty.add(i)));
  }

  // ---- Pipeline ----
  const module = device.createShaderModule({ code: SHADER_WGSL, label: "heap-demo: shader" });
  const bindGroupLayout = device.createBindGroupLayout({
    label: "heap-demo: bgl",
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
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
      { binding: 0, resource: { buffer: globalsBuf } },
      { binding: 1, resource: { buffer: drawHeap } },
      { binding: 2, resource: { buffer: packed.positionsBuf } },
      { binding: 3, resource: { buffer: packed.normalsBuf } },
    ],
  });

  // ---- Geometry-bytes accounting (for stats only) ----
  const geometryBytes =
    packed.entries.reduce((acc, e) => acc + e.indexCount * 4, 0) +
    draws.reduce((acc, d) => acc + d.geo.positions.byteLength + d.geo.normals.byteLength, 0);

  const stats: HeapStats = { globalsBytes: GLOBALS_BYTES, drawBytes: 0, geometryBytes };

  function frame(viewProj: Trafo3d, lightLocation: V3d): void {
    // 1. Globals — small, every frame.
    packGlobals(viewProj, lightLocation, globalsStaging);
    device.queue.writeBuffer(globalsBuf, 0, globalsStaging.buffer, globalsStaging.byteOffset, GLOBALS_BYTES);

    // 2. Per-draw heap — only dirty slots. (Empty for a static scene.)
    let dirtyBytes = 0;
    if (dirty.size > 0) {
      // For each dirty slot, force the latest aval values and upload
      // exactly the 160-byte entry. Production would batch contiguous
      // ranges; uploaded per-slot here to make the cost explicit.
      for (const i of dirty) {
        packDrawHeader(
          modelTrafos[i]!.force(/* allow-force */),
          colors[i]!.force(/* allow-force */),
          packed.entries[i]!.vertexBase, drawStaging, i,
        );
        const byteOff = i * DRAW_HEADER_BYTES;
        device.queue.writeBuffer(
          drawHeap, byteOff,
          drawStaging.buffer, drawStaging.byteOffset + byteOff,
          DRAW_HEADER_BYTES,
        );
        dirtyBytes += DRAW_HEADER_BYTES;
      }
      dirty.clear();
    }
    stats.drawBytes = dirtyBytes;

    // 3. Encode + submit.
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
      pass.drawIndexed(e.indexCount, 1, e.firstIndex, 0, i);
    }
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  function markDirty(drawIndex: number): void {
    if (drawIndex < 0 || drawIndex >= draws.length) {
      throw new RangeError(`markDirty: index ${drawIndex} out of range`);
    }
    dirty.add(drawIndex);
  }

  function dispose(): void {
    for (const s of subs) s.dispose();
    globalsBuf.destroy();
    drawHeap.destroy();
    packed.positionsBuf.destroy();
    packed.normalsBuf.destroy();
    packed.indexBuf.destroy();
  }

  return { frame, markDirty, stats, dispose };
}

function forceFramebuffer(attach: CanvasAttachment): import("@aardworx/wombat.rendering.experimental/core").IFramebuffer {
  return (attach.framebuffer as aval<import("@aardworx/wombat.rendering.experimental/core").IFramebuffer>)
    .force(/* allow-force */);
}
