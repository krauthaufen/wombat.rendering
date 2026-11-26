// Mip-map generation via compute pass.
//
// Standard 2× downsample shader: each invocation reads a 2×2 block
// of source-mip texels and writes the average to the destination
// mip. Iterated mip(N) → mip(N+1) for each pair of adjacent levels.
//
// `generateMips(device, encoder, texture)` records the dispatches
// into the encoder; the user is responsible for submission.
// Pipelines + bind groups are cached per (device, format) so
// repeat calls share the GPU work.

const WGSL_MIP_GEN_TEMPLATE = (storageFormat: string) => `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<${storageFormat}, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dstSize: vec2<u32> = textureDimensions(dst);
    if (id.x >= dstSize.x || id.y >= dstSize.y) { return; }
    let srcSize: vec2<u32> = textureDimensions(src);
    let i: vec2<i32> = vec2<i32>(i32(id.x) * 2, i32(id.y) * 2);
    let i2: vec2<i32> = vec2<i32>(min(i.x + 1, i32(srcSize.x) - 1), min(i.y + 1, i32(srcSize.y) - 1));
    let a: vec4<f32> = textureLoad(src, vec2<i32>(i.x,  i.y),  0);
    let b: vec4<f32> = textureLoad(src, vec2<i32>(i2.x, i.y),  0);
    let c: vec4<f32> = textureLoad(src, vec2<i32>(i.x,  i2.y), 0);
    let d: vec4<f32> = textureLoad(src, vec2<i32>(i2.x, i2.y), 0);
    textureStore(dst, vec2<i32>(i32(id.x), i32(id.y)), (a + b + c + d) * 0.25);
}
`;

interface MipPipelineCache {
  pipelines: Map<GPUTextureFormat, { pipeline: GPUComputePipeline; layout: GPUBindGroupLayout }>;
}

const caches = new WeakMap<GPUDevice, MipPipelineCache>();

function pipelineFor(device: GPUDevice, format: GPUTextureFormat): { pipeline: GPUComputePipeline; layout: GPUBindGroupLayout } {
  let cache = caches.get(device);
  if (cache === undefined) {
    cache = { pipelines: new Map() };
    caches.set(device, cache);
  }
  const existing = cache.pipelines.get(format);
  if (existing !== undefined) return existing;

  const storageFormat = storageFormatFor(format);
  const module = device.createShaderModule({ code: WGSL_MIP_GEN_TEMPLATE(storageFormat), label: `mipgen-${format}` });
  const layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: 0x4, texture: { sampleType: "float" } },
      { binding: 1, visibility: 0x4, storageTexture: { access: "write-only", format: storageFormat as GPUTextureFormat } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module, entryPoint: "main" },
    label: `mipgen-${format}`,
  });
  const entry = { pipeline, layout };
  cache.pipelines.set(format, entry);
  return entry;
}

/**
 * WebGPU storage-texture formats are a constrained subset; we map
 * common sampled formats to the closest writable storage format.
 * For BGRA / sRGB / non-storage formats we fall back to
 * `rgba8unorm` and accept the colour-space caveat (callers using
 * sRGB textures should generate their mips in linear).
 */
function storageFormatFor(format: GPUTextureFormat): GPUTextureFormat {
  switch (format) {
    case "rgba8unorm":
    case "rgba8snorm":
    case "rgba16float":
    case "rgba32float":
    case "r32float":
    case "rg32float":
    case "r32uint":
    case "rg32uint":
    case "rgba32uint":
      return format;
    default:
      return "rgba8unorm";
  }
}

export interface GenerateMipsOptions {
  /** Generate mips starting from this base level. Default 0. */
  readonly baseMipLevel?: number;
  /** Generate up to this many additional mips beyond the base. Default: down to 1×1. */
  readonly mipLevelCount?: number;
  /** Optional debug label. */
  readonly label?: string;
}

/**
 * Records compute dispatches that downsample `texture`'s mip
 * chain. The texture must have been created with
 * `STORAGE_BINDING | TEXTURE_BINDING` and the appropriate
 * `mipLevelCount`. Each level is computed from the level above
 * via a 2×2 box filter.
 */
export function generateMips(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
  texture: GPUTexture,
  opts: GenerateMipsOptions = {},
): void {
  const baseLevel = opts.baseMipLevel ?? 0;
  const totalMips = texture.mipLevelCount;
  const lastLevel = opts.mipLevelCount !== undefined
    ? Math.min(totalMips - 1, baseLevel + opts.mipLevelCount)
    : totalMips - 1;
  if (lastLevel <= baseLevel) return;

  const { pipeline, layout } = pipelineFor(device, texture.format);
  const pass = encoder.beginComputePass(opts.label !== undefined ? { label: opts.label } : {});
  pass.setPipeline(pipeline);

  for (let mip = baseLevel; mip < lastLevel; mip++) {
    const dstWidth = Math.max(1, texture.width >> (mip + 1));
    const dstHeight = Math.max(1, texture.height >> (mip + 1));
    const srcView = texture.createView({
      baseMipLevel: mip, mipLevelCount: 1,
      baseArrayLayer: 0, arrayLayerCount: 1,
      dimension: "2d",
    });
    const dstView = texture.createView({
      baseMipLevel: mip + 1, mipLevelCount: 1,
      baseArrayLayer: 0, arrayLayerCount: 1,
      dimension: "2d",
    });
    const bg = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: srcView },
        { binding: 1, resource: dstView },
      ],
    });
    pass.setBindGroup(0, bg);
    const workX = Math.ceil(dstWidth / 8);
    const workY = Math.ceil(dstHeight / 8);
    pass.dispatchWorkgroups(workX, workY, 1);
  }
  pass.end();
}


// ─── Mip chain by rendering ────────────────────────────────────────────
//
// `fillMips(device, texture)` fills levels 1… of a texture from its level
// 0, each level a linear downsample of the one above, drawn with a
// fullscreen triangle. Unlike the compute path above it needs no storage
// usage and works for every renderable, filterable format (sRGB included:
// sampling decodes, writing encodes, so the average is taken in linear
// light). It encodes and submits on its own, in queue order after
// whatever was written to level 0 before the call.
//
// This is what an uploaded image with `generateMips` gets: allocating the
// levels without filling them leaves them BLACK, and anything sampled at a
// distance then reads black.

const WGSL_MIP_BLIT = `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    // (a destination texel covers two source texels each way: its centre sits on their corner)
    let uv = pos.xy * 2.0 / vec2<f32>(textureDimensions(src));
    return textureSampleLevel(src, smp, uv, 0.0);
}
`;

interface MipBlitCache {
  readonly module: GPUShaderModule;
  readonly sampler: GPUSampler;
  readonly layout: GPUBindGroupLayout;
  readonly pipelineLayout: GPUPipelineLayout;
  readonly pipelines: Map<GPUTextureFormat, GPURenderPipeline>;
}
const blitCaches = new WeakMap<GPUDevice, MipBlitCache>();

export function fillMips(device: GPUDevice, texture: GPUTexture): void {
  if (texture.mipLevelCount <= 1) return;
  let cache = blitCaches.get(device);
  if (cache === undefined) {
    const layout = device.createBindGroupLayout({
      label: "mipblit",
      entries: [
        { binding: 0, visibility: 0x2, texture: { sampleType: "float" } },
        { binding: 1, visibility: 0x2, sampler: { type: "filtering" } },
      ],
    });
    cache = {
      module: device.createShaderModule({ code: WGSL_MIP_BLIT, label: "mipblit" }),
      sampler: device.createSampler({ magFilter: "linear", minFilter: "linear" }),
      layout, pipelineLayout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      pipelines: new Map(),
    };
    blitCaches.set(device, cache);
  }
  let pipeline = cache.pipelines.get(texture.format);
  if (pipeline === undefined) {
    pipeline = device.createRenderPipeline({
      label: `mipblit-${texture.format}`,
      layout: cache.pipelineLayout,
      vertex: { module: cache.module, entryPoint: "vs" },
      fragment: { module: cache.module, entryPoint: "fs", targets: [{ format: texture.format }] },
      primitive: { topology: "triangle-list" },
    });
    cache.pipelines.set(texture.format, pipeline);
  }
  const encoder = device.createCommandEncoder({ label: "mipblit" });
  for (let level = 1; level < texture.mipLevelCount; level++) {
    const from = texture.createView({ baseMipLevel: level - 1, mipLevelCount: 1, dimension: "2d", baseArrayLayer: 0, arrayLayerCount: 1 });
    const to = texture.createView({ baseMipLevel: level, mipLevelCount: 1, dimension: "2d", baseArrayLayer: 0, arrayLayerCount: 1 });
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: to, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: cache.layout,
      entries: [{ binding: 0, resource: from }, { binding: 1, resource: cache.sampler }],
    }));
    pass.draw(3, 1, 0, 0);
    pass.end();
  }
  device.queue.submit([encoder.finish()]);
}
