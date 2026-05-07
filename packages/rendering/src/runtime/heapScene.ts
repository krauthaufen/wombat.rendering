// heapScene — multi-group heap-everything render path.
//
// Public-API extraction of the experimental heap-demo (commits in
// the `experimental/heap-arch` branch document the design). One
// shared GPURenderPipeline + GPUBindGroup per pipeline-state group,
// per-draw uniforms in a storage buffer, vertex pulling from packed
// position / normal slabs, indexed draws via a shared index buffer
// and `firstInstance` routing into the heap.
//
// Per-frame uploads:
//   - Globals (one UBO per group): viewProj + lightLocation. ~80 B.
//     Independent of draw count.
//   - DrawHeader (one storage buffer per group): only slots whose
//     `aval<Trafo3d>` or `aval<V4f>` were marked since last frame
//     are re-uploaded — 160 B per dirty slot.
//
// Geometry is uploaded ONCE at construction (positions + normals +
// indices, packed across all draws in a group).
//
// The user supplies a fragment-shader body per group key. The vertex
// stage is provided by this module and reads from a fixed layout:
//
//   @group(0) @binding(0) var<uniform>             globals:   Globals;
//   @group(0) @binding(1) var<storage, read>       draws:     array<DrawHeader>;
//   @group(0) @binding(2) var<storage, read>       positions: array<f32>;
//   @group(0) @binding(3) var<storage, read>       normals:   array<f32>;
//
//   struct VsOut {
//     @builtin(position) clipPos:  vec4<f32>,
//     @location(0)       worldPos: vec3<f32>,
//     @location(1)       normal:   vec3<f32>,
//     @location(2)       color:    vec4<f32>,
//     @location(3)       lightLoc: vec3<f32>,
//   };
//
// User WGSL must declare `@fragment fn fs(in: VsOut) -> @location(0) vec4<f32>`.

import { Trafo3d, V3d, V4f, type M44d } from "@aardworx/wombat.base";
import { AVal, addMarkingCallback } from "@aardworx/wombat.adaptive";
import type { aval, IDisposable } from "@aardworx/wombat.adaptive";
import type { CanvasAttachment } from "../window/index.js";

// ---------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------

const GLOBALS_BYTES = 64 + 16;                                    // mat4 + vec4
const GLOBALS_FLOATS = GLOBALS_BYTES / 4;
const DRAW_HEADER_BYTES = 64 * 2 + 16 + 16;                       // 2 mat4 + vec4 + (u32 + 3*pad)
const DRAW_HEADER_FLOATS = DRAW_HEADER_BYTES / 4;

function packMat44(m: M44d, dst: Float32Array, off: number): void {
  // M44d._data is the row-major Float64Array; not part of the public
  // type surface. toArray() round-trips through a fresh number[] which
  // we copy into the f32 staging buffer below.
  const r = m.toArray();
  for (let i = 0; i < 16; i++) dst[off + i] = r[i]!;
}

function packGlobals(viewProj: Trafo3d, lightLocation: V3d, dst: Float32Array): void {
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
  dst[off + 32] = color.x; dst[off + 33] = color.y;
  dst[off + 34] = color.z; dst[off + 35] = color.w;
  new Uint32Array(dst.buffer, dst.byteOffset, dst.length)[off + 36] = vertexBase;
}

// ---------------------------------------------------------------------------
// Shared WGSL prelude (struct + bindings + VS)
// ---------------------------------------------------------------------------

const SHADER_PRELUDE = /* wgsl */`
struct Globals {
  viewProj:      mat4x4<f32>,
  lightLocation: vec4<f32>,
};
struct DrawHeader {
  modelTrafo:    mat4x4<f32>,
  modelTrafoInv: mat4x4<f32>,
  color:         vec4<f32>,
  vertexBase:    u32,
  _pad0: u32, _pad1: u32, _pad2: u32,
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
  // wombat.shader convention: matrices uploaded row-major; WGSL reads
  // them as col-major = transposed. v * M is the row-vec dual of
  // M.mul(v) (= M*v math).
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
`;

// ---------------------------------------------------------------------------
// Geometry packing
// ---------------------------------------------------------------------------

interface PackedGeometry {
  readonly positionsBuf: GPUBuffer;
  readonly normalsBuf:   GPUBuffer;
  readonly indexBuf:     GPUBuffer;
  readonly entries:      ReadonlyArray<{ vertexBase: number; indexCount: number; firstIndex: number }>;
  readonly bytes:        number;
}

/**
 * Geometry triple. Tightly-packed Float32 positions / normals (3
 * floats per vertex) plus Uint32 indices. The heap-scene packer
 * concatenates all draws' geometry into per-group slabs.
 */
export interface HeapGeometry {
  readonly positions: Float32Array;
  readonly normals:   Float32Array;
  readonly indices:   Uint32Array;
}

function packGeometry(device: GPUDevice, geos: readonly HeapGeometry[], label: string): PackedGeometry {
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
  const mk = (data: ArrayBufferView, usage: GPUBufferUsageFlags, lbl: string): GPUBuffer => {
    const buf = device.createBuffer({
      size: alignUp(data.byteLength, 4),
      usage: usage | GPUBufferUsage.COPY_DST,
      label: lbl,
    });
    device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
    return buf;
  };
  return {
    positionsBuf: mk(positions, GPUBufferUsage.STORAGE, `${label}/pos`),
    normalsBuf:   mk(normals,   GPUBufferUsage.STORAGE, `${label}/nor`),
    indexBuf:     mk(indices,   GPUBufferUsage.INDEX,   `${label}/idx`),
    entries,
    bytes: positions.byteLength + normals.byteLength + indices.byteLength,
  };
}

function alignUp(n: number, a: number): number { return (n + a - 1) & ~(a - 1); }

function asAval<T>(v: aval<T> | T): aval<T> {
  return (typeof v === "object" && v !== null && typeof (v as { getValue?: unknown }).getValue === "function")
    ? (v as aval<T>)
    : AVal.constant(v as T);
}

// ---------------------------------------------------------------------------
// Per-group internal state
// ---------------------------------------------------------------------------

interface Group {
  readonly key: string;
  readonly pipeline: GPURenderPipeline;
  readonly bindGroup: GPUBindGroup;
  readonly globals: GPUBuffer;
  readonly globalsStaging: Float32Array;
  readonly drawHeap: GPUBuffer;
  readonly drawStaging: Float32Array;
  readonly packed: PackedGeometry;
  readonly trafos: aval<Trafo3d>[];
  readonly colors: aval<V4f>[];
  readonly dirty: Set<number>;
  readonly subs: IDisposable[];
}

function buildGroup(
  device: GPUDevice,
  attach: CanvasAttachment,
  key: string,
  fragmentWgsl: string,
  draws: readonly { spec: HeapDrawSpec; sourceIndex: number }[],
  modelTrafos: aval<Trafo3d>[],
  colors: aval<V4f>[],
): Group {
  const sig = attach.signature;
  const colorAttachmentName = sig.colorNames[0];
  if (colorAttachmentName === undefined) {
    throw new Error("buildHeapScene: framebuffer signature has no color attachment");
  }
  const colorFormat = sig.colors.tryFind(colorAttachmentName)!;
  const depthFormat = sig.depthStencil?.format;

  const memberIndices = draws.map(d => d.sourceIndex);
  const myTrafos = memberIndices.map(i => modelTrafos[i]!);
  const myColors = memberIndices.map(i => colors[i]!);

  const packed = packGeometry(device, draws.map(d => d.spec.geo), `heapScene/${key}`);

  const globals = device.createBuffer({
    size: alignUp(GLOBALS_BYTES, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: `heapScene/${key}/globals`,
  });
  const globalsStaging = new Float32Array(GLOBALS_FLOATS);

  const heapBytes = DRAW_HEADER_BYTES * draws.length;
  const drawHeap = device.createBuffer({
    size: alignUp(heapBytes, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: `heapScene/${key}/draws`,
  });
  const drawStaging = new Float32Array(DRAW_HEADER_FLOATS * draws.length);
  for (let i = 0; i < draws.length; i++) {
    packDrawHeader(
      myTrafos[i]!.force(/* allow-force */),
      myColors[i]!.force(/* allow-force */),
      packed.entries[i]!.vertexBase, drawStaging, i,
    );
  }
  device.queue.writeBuffer(drawHeap, 0, drawStaging.buffer, drawStaging.byteOffset, heapBytes);

  const dirty = new Set<number>();
  const subs: IDisposable[] = [];
  for (let i = 0; i < draws.length; i++) {
    subs.push(addMarkingCallback(myTrafos[i] as never, () => dirty.add(i)));
    subs.push(addMarkingCallback(myColors[i] as never, () => dirty.add(i)));
  }

  const module = device.createShaderModule({
    code: SHADER_PRELUDE + fragmentWgsl,
    label: `heapScene/${key}/shader`,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: `heapScene/${key}/bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const pipeline = device.createRenderPipeline({
    label: `heapScene/${key}/pipeline`,
    layout: pipelineLayout,
    vertex:   { module, entryPoint: "vs", buffers: [] },
    fragment: { module, entryPoint: "fs", targets: [{ format: colorFormat }] },
    primitive: { topology: "triangle-list", cullMode: "back", frontFace: "ccw" },
    ...(depthFormat !== undefined
      ? { depthStencil: { format: depthFormat, depthWriteEnabled: true, depthCompare: "less" as GPUCompareFunction } }
      : {}),
  });
  const bindGroup = device.createBindGroup({
    label: `heapScene/${key}/bg`,
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: globals } },
      { binding: 1, resource: { buffer: drawHeap } },
      { binding: 2, resource: { buffer: packed.positionsBuf } },
      { binding: 3, resource: { buffer: packed.normalsBuf } },
    ],
  });

  return {
    key, pipeline, bindGroup, globals, globalsStaging, drawHeap, drawStaging,
    packed, trafos: myTrafos, colors: myColors, dirty, subs,
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface HeapDrawSpec {
  readonly geo: HeapGeometry;
  readonly modelTrafo: aval<Trafo3d> | Trafo3d;
  readonly color: aval<V4f> | V4f;
  /** Group key — selects which pipeline-state group this draw joins. */
  readonly groupKey: string;
}

export interface HeapSceneStats {
  readonly groups: number;
  readonly totalDraws: number;
  readonly globalsBytes: number;
  drawBytes: number;
  readonly geometryBytes: number;
}

export interface HeapScene {
  /** Render one frame against the given globals. */
  frame(viewProj: Trafo3d, lightLocation: V3d): void;
  readonly stats: HeapSceneStats;
  dispose(): void;
}

export interface BuildHeapSceneOptions {
  /** WGSL fragment-stage source per group key — must declare `fs(in: VsOut) -> @location(0) vec4<f32>`. */
  readonly fragmentShaders: ReadonlyMap<string, string>;
}

/**
 * Build a multi-group heap-backed scene renderer. Buckets `draws` by
 * `groupKey`, builds one pipeline + bind group + heap per bucket,
 * subscribes per-RO avals so dirty marks propagate to per-slot
 * `writeBuffer`s on the next `frame()`.
 */
export function buildHeapScene(
  device: GPUDevice,
  attach: CanvasAttachment,
  draws: readonly HeapDrawSpec[],
  opts: BuildHeapSceneOptions,
): HeapScene {
  const modelTrafos: aval<Trafo3d>[] = draws.map(d => asAval(d.modelTrafo));
  const colors:      aval<V4f>[]    = draws.map(d => asAval(d.color));

  // Bucket by group key.
  const buckets = new Map<string, { spec: HeapDrawSpec; sourceIndex: number }[]>();
  for (let i = 0; i < draws.length; i++) {
    const d = draws[i]!;
    let bucket = buckets.get(d.groupKey);
    if (bucket === undefined) {
      bucket = [];
      buckets.set(d.groupKey, bucket);
    }
    bucket.push({ spec: d, sourceIndex: i });
  }
  const groups: Group[] = [];
  for (const [key, members] of buckets) {
    const fs = opts.fragmentShaders.get(key);
    if (fs === undefined) {
      throw new Error(`buildHeapScene: no fragment shader supplied for groupKey '${key}'`);
    }
    groups.push(buildGroup(device, attach, key, fs, members, modelTrafos, colors));
  }

  const stats: HeapSceneStats = {
    groups: groups.length,
    totalDraws: draws.length,
    globalsBytes: GLOBALS_BYTES * groups.length,
    drawBytes: 0,
    geometryBytes: groups.reduce((acc, g) => acc + g.packed.bytes, 0),
  };

  function frame(viewProj: Trafo3d, lightLocation: V3d): void {
    const fb = (attach.framebuffer as aval<import("../core/index.js").IFramebuffer>).force(/* allow-force */);
    const colorAttachmentName = attach.signature.colorNames[0]!;
    const colorView = fb.colors.tryFind(colorAttachmentName)!;
    const depthFormat = attach.signature.depthStencil?.format;
    const depthView = fb.depthStencil;

    let totalDirtyBytes = 0;

    const enc = device.createCommandEncoder({ label: "heapScene: encoder" });
    let firstPass = true;
    for (const g of groups) {
      packGlobals(viewProj, lightLocation, g.globalsStaging);
      device.queue.writeBuffer(g.globals, 0, g.globalsStaging.buffer, g.globalsStaging.byteOffset, GLOBALS_BYTES);

      if (g.dirty.size > 0) {
        for (const i of g.dirty) {
          packDrawHeader(
            g.trafos[i]!.force(/* allow-force */),
            g.colors[i]!.force(/* allow-force */),
            g.packed.entries[i]!.vertexBase, g.drawStaging, i,
          );
          const byteOff = i * DRAW_HEADER_BYTES;
          device.queue.writeBuffer(
            g.drawHeap, byteOff,
            g.drawStaging.buffer, g.drawStaging.byteOffset + byteOff,
            DRAW_HEADER_BYTES,
          );
          totalDirtyBytes += DRAW_HEADER_BYTES;
        }
        g.dirty.clear();
      }

      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: colorView,
          clearValue: { r: 0.07, g: 0.07, b: 0.08, a: 1.0 },
          loadOp: firstPass ? "clear" : "load",
          storeOp: "store",
        }],
        ...(depthView !== undefined && depthFormat !== undefined ? {
          depthStencilAttachment: {
            view: depthView,
            depthClearValue: 1.0,
            depthLoadOp: firstPass ? "clear" : "load",
            depthStoreOp: "store",
          } satisfies GPURenderPassDepthStencilAttachment,
        } : {}),
      });
      pass.setPipeline(g.pipeline);
      pass.setBindGroup(0, g.bindGroup);
      pass.setIndexBuffer(g.packed.indexBuf, "uint32");
      for (let i = 0; i < g.packed.entries.length; i++) {
        const e = g.packed.entries[i]!;
        pass.drawIndexed(e.indexCount, 1, e.firstIndex, 0, i);
      }
      pass.end();
      firstPass = false;
    }

    stats.drawBytes = totalDirtyBytes;
    device.queue.submit([enc.finish()]);
  }

  function dispose(): void {
    for (const g of groups) {
      for (const s of g.subs) s.dispose();
      g.globals.destroy();
      g.drawHeap.destroy();
      g.packed.positionsBuf.destroy();
      g.packed.normalsBuf.destroy();
      g.packed.indexBuf.destroy();
    }
  }

  return { frame, stats, dispose };
}
