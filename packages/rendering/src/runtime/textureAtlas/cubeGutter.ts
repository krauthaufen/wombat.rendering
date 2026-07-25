// GPU cross-face cube-gutter kernel.
//
// A cube map packed as a COLS×ROWS grid of F×F faces, each ringed by a
// G-px gutter (cell pitch = F+2G), can be sampled as a plain 2D texture by
// projecting the view direction onto a cube face — no `texture_cube` (which
// WebKit/Metal rejects as a view over an array texture). But hardware
// bilinear at a face's interior edge must blend into the CORRECT cube
// neighbour, not the adjacent grid cell. This kernel fills each face's
// gutter ring with the neighbour face's edge by RAY-PROJECTING every output
// pixel through the same cube convention the sampler uses: interior pixels
// map back to their own face (a copy), gutter pixels map onto a neighbour.
//
// Core WebGPU 1.0 only: a sampled input texture + a write-only storage
// texture (no read_write storage textures). Single pass, one dispatch.
//
// Face order is Z-up [+X,-X,+Y,-Y,+Z,-Z] (+Z = up), matching the viewer's
// skybox shader. This is the GPU port of the JS decode-time fill that was
// validated seamless.

export interface CubeGutterLayout {
  /** Face size in px (interior, per face). */
  readonly faceSize: number;
  /** Gutter ring width in px on each side. */
  readonly gutter: number;
  /** Grid columns / rows (COLS*ROWS must be >= 6; 3x2 by default). */
  readonly cols: number;
  readonly rows: number;
}

const WGSL = /* wgsl */`
struct Params { F:u32, G:u32, cols:u32, rows:u32, imw:u32, imh:u32, _a:u32, _b:u32 }
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSmp: sampler;
@group(0) @binding(2) var dstTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> P: Params;

// Z-up cube: (face,u,v) -> direction. Inverse of dirToFace below.
fn faceToDir(f:u32, u:f32, v:f32) -> vec3<f32> {
  let s = 2.0*u - 1.0; let t = 2.0*v - 1.0;
  if (f==0u) { return vec3<f32>( 1.0,   s,   t); }  // +X: u->+Y, v->+Z
  if (f==1u) { return vec3<f32>(-1.0,  -s,   t); }  // -X
  if (f==2u) { return vec3<f32>(  -s, 1.0,   t); }  // +Y
  if (f==3u) { return vec3<f32>(   s,-1.0,   t); }  // -Y
  if (f==4u) { return vec3<f32>(   s,   t, 1.0); }  // +Z (up)
  return          vec3<f32>(   s,  -t,-1.0);        // -Z (down)
}
// direction -> (face, u, v)
fn dirToFace(d:vec3<f32>) -> vec3<f32> {
  let ax = abs(d.x); let ay = abs(d.y); let az = abs(d.z);
  var f:f32; var s:f32; var t:f32; var m:f32;
  if (az >= ax && az >= ay) { m = az; if (d.z > 0.0) { f=4.0; s= d.x; t= d.y; } else { f=5.0; s= d.x; t=-d.y; } }
  else if (ax >= ay)        { m = ax; if (d.x > 0.0) { f=0.0; s= d.y; t= d.z; } else { f=1.0; s=-d.y; t= d.z; } }
  else                      { m = ay; if (d.y > 0.0) { f=2.0; s=-d.x; t= d.z; } else { f=3.0; s= d.x; t= d.z; } }
  return vec3<f32>(f, 0.5*(s/m + 1.0), 0.5*(t/m + 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  if (gid.x >= P.imw || gid.y >= P.imh) { return; }
  let CELL = P.F + 2u*P.G;
  let col = gid.x / CELL; let row = gid.y / CELL;
  let i = row * P.cols + col;
  if (i >= 6u) { return; }
  let bx = gid.x - col*CELL; let by = gid.y - row*CELL;      // px within cell (incl. gutter)
  let u = (f32(bx) - f32(P.G) + 0.5) / f32(P.F);             // interior [G,G+F) -> [0,1]
  let v = (f32(by) - f32(P.G) + 0.5) / f32(P.F);
  let fuv = dirToFace(faceToDir(i, u, v));                   // interior->self, gutter->neighbour
  let fj = u32(fuv.x + 0.5);
  let cj = fj % P.cols; let rj = fj / P.cols;
  // clamp to the neighbour face's interior (half-texel inset) so bilinear
  // never reads that face's own gutter.
  let iu = clamp(fuv.y, 0.5/f32(P.F), 1.0 - 0.5/f32(P.F));
  let iv = clamp(fuv.z, 0.5/f32(P.F), 1.0 - 0.5/f32(P.F));
  let sx = (f32(cj*CELL + P.G) + iu*f32(P.F)) / f32(P.imw);
  let sy = (f32(rj*CELL + P.G) + iv*f32(P.F)) / f32(P.imh);
  let c = textureSampleLevel(srcTex, srcSmp, vec2<f32>(sx, sy), 0.0);
  textureStore(dstTex, vec2<i32>(i32(gid.x), i32(gid.y)), c);
}
`;

interface Cache {
  pipeline: GPUComputePipeline;
  bgl: GPUBindGroupLayout;
  sampler: GPUSampler;
}
const caches = new WeakMap<GPUDevice, Cache>();

function getKernel(device: GPUDevice): Cache {
  const cached = caches.get(device);
  if (cached !== undefined) return cached;
  const module = device.createShaderModule({ code: WGSL, label: "atlas/cubeGutter" });
  const bgl = device.createBindGroupLayout({
    label: "atlas/cubeGutter/bgl",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "2d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm", viewDimension: "2d" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module, entryPoint: "main" },
    label: "atlas/cubeGutter",
  });
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
  const entry = { pipeline, bgl, sampler };
  caches.set(device, entry);
  return entry;
}

/** Grid pixel dimensions for a layout. */
export function cubeGridSize(layout: CubeGutterLayout): { width: number; height: number } {
  const cell = layout.faceSize + 2 * layout.gutter;
  return { width: layout.cols * cell, height: layout.rows * cell };
}

/**
 * Fill the cross-face gutters of a packed cube grid on the GPU.
 * `srcView` must contain the 6 face interiors at their cell origins (offset
 * by `gutter`); `dstTex` receives the full grid (interiors + cross-face
 * gutters). `src` and `dst` may be the SAME texture only if it supports both
 * TEXTURE_BINDING and STORAGE_BINDING and the caller accepts in-place
 * hazards — prefer distinct textures. Encodes into `encoder` if given,
 * otherwise submits its own.
 */
export function fillCubeGutters(
  device: GPUDevice,
  srcView: GPUTextureView,
  dstView: GPUTextureView,
  layout: CubeGutterLayout,
  encoder?: GPUCommandEncoder,
): void {
  const k = getKernel(device);
  const { width, height } = cubeGridSize(layout);
  const params = new Uint32Array([layout.faceSize, layout.gutter, layout.cols, layout.rows, width, height, 0, 0]);
  const ubo = device.createBuffer({ size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "atlas/cubeGutter/params" });
  device.queue.writeBuffer(ubo, 0, params);
  const bindGroup = device.createBindGroup({
    layout: k.bgl,
    entries: [
      { binding: 0, resource: srcView },
      { binding: 1, resource: k.sampler },
      { binding: 2, resource: dstView },
      { binding: 3, resource: { buffer: ubo } },
    ],
  });
  const enc = encoder ?? device.createCommandEncoder({ label: "atlas/cubeGutter" });
  const pass = enc.beginComputePass({ label: "atlas/cubeGutter" });
  pass.setPipeline(k.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8), 1);
  pass.end();
  if (encoder === undefined) device.queue.submit([enc.finish()]);
}
