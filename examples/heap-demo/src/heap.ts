// heap.ts — Phase 5: multi-group heap.
//
//   Group = (pipeline-state key) → its own
//     - GPURenderPipeline
//     - GPUBindGroup (globals UBO + per-draw heap + position/normal slabs)
//     - DrawHeader[] storage buffer
//     - geometry slab (positions + normals + indices)
//   Same per-group invariants as Phase 4. The only new thing is a
//   top-level dispatcher that buckets draws by group key, builds
//   one group per bucket, and rebinds + redraws per group on each
//   frame.
//
// Group key for the demo is just `kind: "lambert" | "flat"` — the
// shader. Production keys would also include framebuffer signature,
// rasterizer / depth / blend / topology / texture-set-id.

import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import type { aval, IDisposable } from "@aardworx/wombat.adaptive";
import { AVal, addMarkingCallback } from "@aardworx/wombat.adaptive";
import type { CanvasAttachment } from "@aardworx/wombat.rendering.experimental/window";
import type { GeometryData } from "./geometry.js";

// ---------------------------------------------------------------------------
// Layouts (same as Phase 3/4)
// ---------------------------------------------------------------------------

const GLOBALS_BYTES = 64 + 16;
const GLOBALS_FLOATS = GLOBALS_BYTES / 4;
const DRAW_HEADER_BYTES = 64 * 2 + 16 + 16;
const DRAW_HEADER_FLOATS = DRAW_HEADER_BYTES / 4;

function packMat44(m: import("@aardworx/wombat.base").M44d, dst: Float32Array, off: number): void {
  const r = m._data;
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
  modelTrafo: Trafo3d, color: V4f, vertexBase: number,
  dst: Float32Array, drawIndex: number,
): void {
  const off = drawIndex * DRAW_HEADER_FLOATS;
  packMat44(modelTrafo.forward,  dst, off +  0);
  packMat44(modelTrafo.backward, dst, off + 16);
  dst[off + 32] = color.x; dst[off + 33] = color.y;
  dst[off + 34] = color.z; dst[off + 35] = color.w;
  new Uint32Array(dst.buffer, dst.byteOffset, dst.length)[off + 36] = vertexBase;
}

// ---------------------------------------------------------------------------
// Per-kind shaders
// ---------------------------------------------------------------------------

const SHADER_COMMON_HEAD = /* wgsl */`
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

const SHADER_LAMBERT_FS = /* wgsl */`
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

const SHADER_FLAT_FS = /* wgsl */`
@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  return in.color;
}
`;

export type ShaderKind = "lambert" | "flat";

const WGSL_FOR: Record<ShaderKind, string> = {
  lambert: SHADER_COMMON_HEAD + SHADER_LAMBERT_FS,
  flat:    SHADER_COMMON_HEAD + SHADER_FLAT_FS,
};

// ---------------------------------------------------------------------------
// Geometry packing — per group. Same code as before, parameterised
// over which subset of the input draws falls into the group.
// ---------------------------------------------------------------------------

interface PackedGeometry {
  readonly positionsBuf: GPUBuffer;
  readonly normalsBuf:   GPUBuffer;
  readonly indexBuf:     GPUBuffer;
  readonly entries:      ReadonlyArray<{ vertexBase: number; indexCount: number; firstIndex: number }>;
  readonly bytes:        number;
}

function packGeometry(device: GPUDevice, geos: readonly GeometryData[], label: string): PackedGeometry {
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

// ---------------------------------------------------------------------------
// Per-group internal state
// ---------------------------------------------------------------------------

interface Group {
  readonly kind: ShaderKind;
  readonly pipeline: GPURenderPipeline;
  readonly bindGroup: GPUBindGroup;
  readonly globals: GPUBuffer;
  readonly globalsStaging: Float32Array;
  readonly drawHeap: GPUBuffer;
  readonly drawStaging: Float32Array;
  readonly packed: PackedGeometry;
  /** Source RO indices (in caller order) that were packed into this group. */
  readonly memberIndices: number[];
  readonly trafos: aval<Trafo3d>[];
  readonly colors: aval<V4f>[];
  readonly dirty: Set<number>;     // local-to-group slot indices
  readonly subs: IDisposable[];
}

function buildGroup(
  device: GPUDevice,
  attach: CanvasAttachment,
  kind: ShaderKind,
  draws: readonly { spec: HeapDrawSpec; sourceIndex: number }[],
  modelTrafos: aval<Trafo3d>[],
  colors: aval<V4f>[],
): Group {
  const sig = attach.signature;
  const colorFormat = sig.colors.tryFind("outColor")!;
  const depthFormat = sig.depthStencil?.format;

  const memberIndices = draws.map(d => d.sourceIndex);
  const myTrafos = memberIndices.map(i => modelTrafos[i]!);
  const myColors = memberIndices.map(i => colors[i]!);

  const packed = packGeometry(device, draws.map(d => d.spec.geo), `heap-demo/${kind}`);

  const globals = device.createBuffer({
    size: alignUp(GLOBALS_BYTES, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: `heap-demo/${kind}/globals`,
  });
  const globalsStaging = new Float32Array(GLOBALS_FLOATS);

  const heapBytes = DRAW_HEADER_BYTES * draws.length;
  const drawHeap = device.createBuffer({
    size: alignUp(heapBytes, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: `heap-demo/${kind}/draws`,
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

  const module = device.createShaderModule({ code: WGSL_FOR[kind], label: `heap-demo/${kind}/shader` });
  const bindGroupLayout = device.createBindGroupLayout({
    label: `heap-demo/${kind}/bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const pipeline = device.createRenderPipeline({
    label: `heap-demo/${kind}/pipeline`,
    layout: pipelineLayout,
    vertex:   { module, entryPoint: "vs", buffers: [] },
    fragment: { module, entryPoint: "fs", targets: [{ format: colorFormat }] },
    primitive: { topology: "triangle-list", cullMode: "back", frontFace: "ccw" },
    depthStencil: depthFormat !== undefined
      ? { format: depthFormat, depthWriteEnabled: true, depthCompare: "less" }
      : undefined,
  });
  const bindGroup = device.createBindGroup({
    label: `heap-demo/${kind}/bg`,
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: globals } },
      { binding: 1, resource: { buffer: drawHeap } },
      { binding: 2, resource: { buffer: packed.positionsBuf } },
      { binding: 3, resource: { buffer: packed.normalsBuf } },
    ],
  });

  return {
    kind, pipeline, bindGroup, globals, globalsStaging, drawHeap, drawStaging,
    packed, memberIndices, trafos: myTrafos, colors: myColors, dirty, subs,
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface HeapDrawSpec {
  readonly geo: GeometryData;
  readonly modelTrafo: aval<Trafo3d> | Trafo3d;
  readonly color: aval<V4f> | V4f;
  /** Group key — selects which pipeline-state group this RO joins. */
  readonly kind: ShaderKind;
}

export interface HeapStats {
  readonly groups: number;
  readonly totalDraws: number;
  readonly globalsBytes: number;     // total per-frame globals across all groups
  drawBytes: number;                 // per-frame mutated heap bytes across all groups (this frame)
  readonly geometryBytes: number;    // total one-time geometry across all groups
}

export interface HeapRenderer {
  frame(viewProj: Trafo3d, lightLocation: V3d): void;
  readonly stats: HeapStats;
  dispose(): void;
}

export function buildHeapRenderer(
  device: GPUDevice,
  attach: CanvasAttachment,
  draws: readonly HeapDrawSpec[],
): HeapRenderer {
  // Lift inputs to avals (shape-uniform downstream).
  function asAval<T>(v: aval<T> | T): aval<T> {
    return (typeof v === "object" && v !== null && typeof (v as { getValue?: unknown }).getValue === "function")
      ? (v as aval<T>)
      : AVal.constant(v as T);
  }
  const modelTrafos: aval<Trafo3d>[] = draws.map(d => asAval(d.modelTrafo));
  const colors:      aval<V4f>[]    = draws.map(d => asAval(d.color));

  // Bucket by group key. Each group sees only its members, but we
  // remember source indices so per-group dirty state stays aligned
  // with the user's RO list.
  const buckets = new Map<ShaderKind, { spec: HeapDrawSpec; sourceIndex: number }[]>();
  for (let i = 0; i < draws.length; i++) {
    const d = draws[i]!;
    let bucket = buckets.get(d.kind);
    if (bucket === undefined) {
      bucket = [];
      buckets.set(d.kind, bucket);
    }
    bucket.push({ spec: d, sourceIndex: i });
  }
  const groups: Group[] = [];
  for (const [kind, members] of buckets) {
    groups.push(buildGroup(device, attach, kind, members, modelTrafos, colors));
  }

  const stats: HeapStats = {
    groups: groups.length,
    totalDraws: draws.length,
    globalsBytes: GLOBALS_BYTES * groups.length,
    drawBytes: 0,
    geometryBytes: groups.reduce((acc, g) => acc + g.packed.bytes, 0),
  };

  function frame(viewProj: Trafo3d, lightLocation: V3d): void {
    const fb = forceFramebuffer(attach);
    const colorView = fb.colors.tryFind("outColor")!;
    const depthFormat = attach.signature.depthStencil?.format;
    const depthView = fb.depthStencil;

    let totalDirtyBytes = 0;

    // Encode a single render pass with all groups stitched together.
    // First group does loadOp=clear; subsequent groups load to keep
    // prior groups' pixels.
    const enc = device.createCommandEncoder({ label: "heap-demo: encoder" });

    let firstPass = true;
    for (const g of groups) {
      // 1. Globals — small, every frame, per group.
      packGlobals(viewProj, lightLocation, g.globalsStaging);
      device.queue.writeBuffer(g.globals, 0, g.globalsStaging.buffer, g.globalsStaging.byteOffset, GLOBALS_BYTES);

      // 2. Per-draw dirty slots.
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

      // 3. Render the group.
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

function forceFramebuffer(attach: CanvasAttachment): import("@aardworx/wombat.rendering.experimental/core").IFramebuffer {
  return (attach.framebuffer as aval<import("@aardworx/wombat.rendering.experimental/core").IFramebuffer>)
    .force(/* allow-force */);
}
