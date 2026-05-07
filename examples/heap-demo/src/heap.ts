// heap.ts — standalone heap-backed render path for the demo. Phase 1
// of the heap-everything architecture, prototyped here outside the
// runtime so iteration is cheap. Once visually equivalent to the
// per-RO baseline (and we've stress-tested the
// firstInstance→heap-index trick on real WebGPU), we port it into
// wombat.rendering as a proper subsystem.
//
// What this proves out:
//   - One shared `GPURenderPipeline` for the whole group of draws.
//   - One shared `GPUBindGroup` whose group(0) is a storage buffer
//     holding `N × DrawUniforms` back-to-back.
//   - Per-draw routing via `firstInstance` — the WGSL VS reads
//     `heap[instance_index]` and the `instanceCount=1` draw makes
//     `instance_index = firstInstance = drawIndex`.
//   - One `setBindGroup` for the whole frame (vs N per-RO bind groups
//     in the baseline).
//
// What it does NOT do yet (Phase 2+):
//   - Vertex pulling (positions / normals stay in per-RO vertex
//     buffers via setVertexBuffer).
//   - Multi-draw-indirect (chromium-experimental flag, not standard
//     WebGPU yet — issue plain `draw` per RO for now; the wins come
//     from the bind-group + pipeline collapse, not the call count).
//   - Texture-set keying (none of the demo's ROs use textures).
//   - User-supplied `kind: "gpu"` buffer fallback.

import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import type { aval } from "@aardworx/wombat.adaptive";
import type { CanvasAttachment } from "@aardworx/wombat.rendering.experimental/window";
import type { GeometryData } from "./geometry.js";

// ---------------------------------------------------------------------------
// DrawUniforms layout (std430 / WGSL storage-buffer)
// ---------------------------------------------------------------------------
//
//   struct DrawUniforms {
//     modelTrafo:    mat4x4<f32>,   // 64 bytes
//     modelTrafoInv: mat4x4<f32>,   // 64
//     viewProjTrafo: mat4x4<f32>,   // 64
//     lightLocation: vec4<f32>,     // 16 (.xyz used; .w padding)
//   };
//
// 208 bytes per entry, 16-byte aligned. We pack matrices in
// column-major order — WGSL `mat4x4<f32>` treats stored bytes as
// column-major.

const DRAW_UNIFORMS_BYTES = 64 * 3 + 16;
const DRAW_UNIFORMS_FLOATS = DRAW_UNIFORMS_BYTES / 4;

/** Row-major M44d → column-major Float32 view, written into `dst[off..off+16]`. */
function packMat44ColMajor(m: import("@aardworx/wombat.base").M44d, dst: Float32Array, off: number): void {
  // `m._data` is row-major (rows × cols). Transpose to column-major.
  const r = m._data;
  // col 0
  dst[off +  0] = r[0]!;  dst[off +  1] = r[4]!;  dst[off +  2] = r[8]!;   dst[off +  3] = r[12]!;
  // col 1
  dst[off +  4] = r[1]!;  dst[off +  5] = r[5]!;  dst[off +  6] = r[9]!;   dst[off +  7] = r[13]!;
  // col 2
  dst[off +  8] = r[2]!;  dst[off +  9] = r[6]!;  dst[off + 10] = r[10]!;  dst[off + 11] = r[14]!;
  // col 3
  dst[off + 12] = r[3]!;  dst[off + 13] = r[7]!;  dst[off + 14] = r[11]!;  dst[off + 15] = r[15]!;
}

function packDrawUniforms(
  modelTrafo: Trafo3d,
  viewProj: Trafo3d,
  lightLocation: V3d,
  dst: Float32Array,
  drawIndex: number,
): void {
  const off = drawIndex * DRAW_UNIFORMS_FLOATS;
  packMat44ColMajor(modelTrafo.forward,  dst, off +  0);
  packMat44ColMajor(modelTrafo.backward, dst, off + 16);
  packMat44ColMajor(viewProj.forward,    dst, off + 32);
  dst[off + 48] = lightLocation.x as number;
  dst[off + 49] = lightLocation.y as number;
  dst[off + 50] = lightLocation.z as number;
  dst[off + 51] = 0;
}

// ---------------------------------------------------------------------------
// Hand-rolled WGSL — vertex pulls per-instance uniforms from the heap.
// ---------------------------------------------------------------------------

const SHADER_WGSL = /* wgsl */`
struct DrawUniforms {
  modelTrafo:    mat4x4<f32>,
  modelTrafoInv: mat4x4<f32>,
  viewProjTrafo: mat4x4<f32>,
  lightLocation: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> heap: array<DrawUniforms>;

struct VsIn {
  @location(0) position: vec3<f32>,
  @location(1) normal:   vec3<f32>,
  @location(2) color:    vec4<f32>,
};

struct VsOut {
  @builtin(position) clipPos:        vec4<f32>,
  @location(0)       worldPos:       vec3<f32>,
  @location(1)       normal:         vec3<f32>,
  @location(2)       color:          vec4<f32>,
  @location(3)       lightLoc:       vec3<f32>,
};

@vertex
fn vs(in: VsIn, @builtin(instance_index) drawIdx: u32) -> VsOut {
  let u = heap[drawIdx];
  let wp = u.modelTrafo * vec4<f32>(in.position, 1.0);
  // Inv-transpose normal trick: row-vec multiplication n * M^-1
  // equals (M^-1)^T * n = NormalMatrix * n.
  let n = (vec4<f32>(in.normal, 0.0) * u.modelTrafoInv).xyz;
  var out: VsOut;
  out.clipPos  = u.viewProjTrafo * wp;
  out.worldPos = wp.xyz;
  out.normal   = n;
  out.color    = in.color;
  out.lightLoc = u.lightLocation.xyz;
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
// Per-RO GPU resource bundle — vertex / index buffers stay per-draw
// in Phase 1.
// ---------------------------------------------------------------------------

interface RoGpu {
  readonly positions: GPUBuffer;
  readonly normals:   GPUBuffer;
  readonly colors:    GPUBuffer;       // 16 bytes, stride 0 (broadcast)
  readonly indices:   GPUBuffer;
  readonly indexCount: number;
}

function uploadGeometry(device: GPUDevice, geo: GeometryData, color: V4f): RoGpu {
  const mk = (data: ArrayBufferView, usage: GPUBufferUsageFlags): GPUBuffer => {
    const buf = device.createBuffer({
      size: alignUp(data.byteLength, 4),
      usage: usage | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
    return buf;
  };
  const colorBytes = new Float32Array([color.x, color.y, color.z, color.w]);
  return {
    positions: mk(geo.positions, GPUBufferUsage.VERTEX),
    normals:   mk(geo.normals,   GPUBufferUsage.VERTEX),
    colors:    mk(colorBytes,    GPUBufferUsage.VERTEX),
    indices:   mk(geo.indices,   GPUBufferUsage.INDEX),
    indexCount: geo.indices.length,
  };
}

function alignUp(n: number, a: number): number {
  return (n + a - 1) & ~(a - 1);
}

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
  const colorFormat = sig.colors.tryFind("color");
  if (colorFormat === undefined) throw new Error("buildHeapRenderer: attachment has no 'color' attachment");
  const depthFormat = sig.depthStencil?.format;

  // ---- Per-RO GPU resources ----
  const ros: RoGpu[] = draws.map(d => uploadGeometry(device, d.geo, d.color));

  // ---- Heap (storage buffer) ----
  const heapBytes = DRAW_UNIFORMS_BYTES * draws.length;
  const heap = device.createBuffer({
    size: alignUp(heapBytes, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "heap-demo: uniform heap",
  });
  const heapStaging = new Float32Array(DRAW_UNIFORMS_FLOATS * draws.length);

  // ---- Pipeline ----
  const module = device.createShaderModule({
    code: SHADER_WGSL,
    label: "heap-demo: heap-shader",
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: "heap-demo: bgl",
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const pipeline = device.createRenderPipeline({
    label: "heap-demo: pipeline",
    layout: pipelineLayout,
    vertex: {
      module,
      entryPoint: "vs",
      buffers: [
        { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] },
        { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }] },
        // arrayStride 0 → broadcast: every vertex reads element 0 of the colors buffer.
        { arrayStride:  0, attributes: [{ shaderLocation: 2, offset: 0, format: "float32x4" }] },
      ],
    },
    fragment: {
      module,
      entryPoint: "fs",
      targets: [{ format: colorFormat }],
    },
    primitive: { topology: "triangle-list", cullMode: "back", frontFace: "ccw" },
    depthStencil: depthFormat !== undefined
      ? { format: depthFormat, depthWriteEnabled: true, depthCompare: "less" }
      : undefined,
  });

  // ---- Bind group ----
  const bindGroup = device.createBindGroup({
    label: "heap-demo: bg",
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: heap } }],
  });

  // ---- Frame ----
  function frame(viewProj: Trafo3d, lightLocation: V3d): void {
    // 1. Pack the heap.
    for (let i = 0; i < draws.length; i++) {
      packDrawUniforms(draws[i]!.modelTrafo, viewProj, lightLocation, heapStaging, i);
    }
    device.queue.writeBuffer(heap, 0, heapStaging.buffer, heapStaging.byteOffset, heapBytes);

    // 2. Acquire framebuffer.
    const fb = forceFramebuffer(attach);
    const colorView = fb.colors.tryFind("color");
    if (colorView === undefined) throw new Error("heap-demo: framebuffer has no 'color'");
    const depthView = fb.depthStencil?.view;

    // 3. Encode the pass.
    const enc = device.createCommandEncoder({ label: "heap-demo: encoder" });
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: colorView.view,
        clearValue: { r: 0.07, g: 0.07, b: 0.08, a: 1.0 },
        loadOp: "clear", storeOp: "store",
      }],
      ...(depthView !== undefined && depthFormat !== undefined ? {
        depthStencilAttachment: {
          view: depthView,
          depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
        },
      } : {}),
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);

    for (let i = 0; i < ros.length; i++) {
      const r = ros[i]!;
      pass.setVertexBuffer(0, r.positions);
      pass.setVertexBuffer(1, r.normals);
      pass.setVertexBuffer(2, r.colors);
      pass.setIndexBuffer(r.indices, "uint32");
      // Routing: firstInstance = drawIndex, instanceCount = 1, so the
      // VS sees `instance_index = drawIndex` — the heap index.
      pass.drawIndexed(r.indexCount, 1, 0, 0, i);
    }

    pass.end();
    device.queue.submit([enc.finish()]);
  }

  function dispose(): void {
    for (const r of ros) {
      r.positions.destroy(); r.normals.destroy(); r.colors.destroy(); r.indices.destroy();
    }
    heap.destroy();
  }

  return { frame, dispose };
}

// ---------------------------------------------------------------------------
// Internals — force the canvas attachment's framebuffer aval (we
// bypass the runtime here, so we read it ourselves once per frame).
// ---------------------------------------------------------------------------

function forceFramebuffer(attach: CanvasAttachment): import("@aardworx/wombat.rendering.experimental/core").IFramebuffer {
  // attach.framebuffer is `aval<IFramebuffer>` — force without a token.
  return (attach.framebuffer as aval<import("@aardworx/wombat.rendering.experimental/core").IFramebuffer>)
    .force(/* allow-force */);
}
