// Atlas-variant heap-bucket integration. Mock-GPU; verifies the
// bucket-keying / bind-group / drawHeader plumbing for the atlas-
// tier texture path (PR 3 of docs/heap-textures-plan.md).
//
// Pixel correctness is NOT covered here — the shader-side
// `textureSample` rewrite is the next PR; until then atlas-variant
// ROs render in placeholder colors.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken } from "@aardworx/wombat.adaptive";
import { ITexture } from "../packages/rendering/src/core/texture.js";
import { ISampler } from "../packages/rendering/src/core/sampler.js";
import { buildHeapScene, type HeapDrawSpec, type HeapTextureSet } from "../packages/rendering/src/runtime/heapScene.js";
import { AtlasPool, atlasFormatIndex } from "../packages/rendering/src/runtime/textureAtlas/atlasPool.js";
import { createFramebufferSignature } from "../packages/rendering/src/resources/framebufferSignature.js";
import { MockGPU } from "./_mockGpu.js";

// Mock GPU lacks the WebGPU global enums; AtlasPool / heapScene call
// `device.createTexture({ usage: GPUTextureUsage.… })` etc. Stub the
// few constants used by these paths.
if (typeof (globalThis as { GPUTextureUsage?: unknown }).GPUTextureUsage === "undefined") {
  (globalThis as Record<string, unknown>).GPUTextureUsage = {
    COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04,
    STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10,
  };
}
if (typeof (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage === "undefined") {
  (globalThis as Record<string, unknown>).GPUBufferUsage = {
    MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
    INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
    INDIRECT: 0x0100, QUERY_RESOLVE: 0x0200,
  };
}
if (typeof (globalThis as { GPUShaderStage?: unknown }).GPUShaderStage === "undefined") {
  (globalThis as Record<string, unknown>).GPUShaderStage = {
    VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4,
  };
}

const IDENTITY44 = (() => { const a = new Float64Array(16); a[0]=1; a[5]=1; a[10]=1; a[15]=1; return a; })();
const trafoIdentity = { forward: { toArray: () => IDENTITY44 } } as unknown;
const v3 = (x: number, y: number, z: number) => ({ x, y, z }) as unknown;
const v4 = (x: number, y: number, z: number, w: number) => ({ x, y, z, w }) as unknown;

// Raw VS/FS pair driving the textured RAW_TEXTURED_SCHEMA on the heap
// path. The shader bodies don't matter for these tests (we never
// dispatch GPU work) — they just need to parse far enough for the
// regex rewriter and the bind-group construction.
const VS_WGSL = /* wgsl */`
@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) drawIdx: u32) -> VsOut {
  var out: VsOut;
  out.clipPos  = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  out.worldPos = vec3<f32>(0.0);
  out.normal   = vec3<f32>(0.0, 0.0, 1.0);
  out.color    = vec4<f32>(1.0);
  out.lightLoc = vec3<f32>(0.0);
  return out;
}
`;
const FS_WGSL = /* wgsl */`
@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  return in.color;
}
`;
const sharedShader = { vs: VS_WGSL, fs: FS_WGSL } as const;

function geomSpec(pool: AtlasPool, format: "rgba8unorm" | "rgba8unorm-srgb"): {
  spec: HeapDrawSpec; textures: HeapTextureSet & { kind: "atlas" };
} {
  const tex = ITexture.fromRaw({
    data: new Uint8Array(64 * 64 * 4),
    width: 64, height: 64,
    format,
  });
  const texAval = AVal.constant(tex);
  const acq = pool.acquire(format, texAval, 64, 64);
  const sampler = ISampler.fromDescriptor({
    magFilter: "linear", minFilter: "linear",
    addressModeU: "repeat", addressModeV: "clamp-to-edge",
  });
  const textures: HeapTextureSet & { kind: "atlas" } = {
    kind: "atlas",
    format, pageId: acq.pageId,
    origin: acq.origin, size: acq.size,
    numMips: acq.numMips,
    sampler, page: acq.page,
    poolRef: acq.ref,
    release: () => pool.release(acq.ref),
  };
  const spec: HeapDrawSpec = {
    effect: sharedShader,
    inputs: {
      Positions: AVal.constant(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0])),
      Normals:   AVal.constant(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1])),
      ModelTrafo:    AVal.constant(trafoIdentity),
      Color:         AVal.constant(v4(1, 0, 0, 1)),
      ViewProjTrafo: AVal.constant(trafoIdentity),
      LightLocation: AVal.constant(v3(0, 0, 1)),
    },
    indices: AVal.constant(new Uint32Array([0, 1, 2])),
    textures,
  };
  return { spec, textures };
}

const sig = () => createFramebufferSignature({
  colors: { outColor: "rgba8unorm" },
  depthStencil: { format: "depth24plus" },
});

describe("heap-atlas bucket plumbing", () => {
  it("two atlas-variant ROs sharing the same atlas page → same bucket", () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(gpu.device);
    const { spec: a } = geomSpec(pool, "rgba8unorm");
    const { spec: b } = geomSpec(pool, "rgba8unorm");
    const scene = buildHeapScene(gpu.device, sig(), [a, b], { atlasPool: pool });
    expect(scene.stats.groups).toBe(1);
  });

  it("atlas-variant ROs with different formats but same effect/pipeline → same bucket", () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(gpu.device);
    const { spec: a } = geomSpec(pool, "rgba8unorm");
    const { spec: b } = geomSpec(pool, "rgba8unorm-srgb");
    const scene = buildHeapScene(gpu.device, sig(), [a, b], { atlasPool: pool });
    // Both ROs share `(effect, pipelineState, "atlas")` bucket key.
    expect(scene.stats.groups).toBe(1);
    // Bind group should reference both format binding-arrays. The
    // last-built bind group descriptor for an atlas-variant bucket
    // includes 13 bindings: 4 heap views + 2 atlas binding_arrays
    // (linear + srgb) + 1 atlas sampler = 7 visible entries. We
    // assert the atlas-array entries exist by binding number.
    const lastBg = gpu.bindGroups[gpu.bindGroups.length - 1]!;
    const bindings = (lastBg.entries as readonly GPUBindGroupEntry[]).map(e => e.binding);
    expect(bindings).toContain(11); // ATLAS_BINDING_LINEAR
    expect(bindings).toContain(12); // ATLAS_BINDING_SRGB
    expect(bindings).toContain(13); // ATLAS_BINDING_SAMPLER
  });

  it("addDraw of an atlas-variant RO writes the four drawHeader fields with expected values", () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(gpu.device);
    const { spec: a, textures } = geomSpec(pool, "rgba8unorm-srgb");
    const scene = buildHeapScene(gpu.device, sig(), [a], { atlasPool: pool });
    // The drawHeader writeBuffer is staged in addDraw + flushed in update.
    scene.update(AdaptiveToken.top);
    // The drawHeader is uploaded via a writeBuffer call against the
    // bucket's drawHeap GPUBuffer (label includes "drawHeap"). Pick
    // the latest such write.
    const drawHeapWrites = gpu.writeBufferCalls.filter(c => /drawHeap/.test(c.buffer.label ?? ""));
    expect(drawHeapWrites.length).toBeGreaterThan(0);
    const w = drawHeapWrites[drawHeapWrites.length - 1]!;
    // The written region covers the slot's full drawHeader. Decode it
    // into a u32+f32 view; the texture-ref subfields live AFTER the
    // schema's uniform/attribute ref slots. The textured raw schema
    // declares 4 uniform refs + 2 attribute refs = 24 bytes; the
    // `checker` texture-ref block follows at offset 24.
    //
    // The data parameter to writeBuffer is the staging Float32Array's
    // ArrayBuffer; the call carries (buffer, offset, data, dataOffset, size).
    // We slice out the slot's 56 bytes (24 ref + 24 atlas + 8 pad).
    const ab = w.data as ArrayBuffer | ArrayBufferView;
    const buf = ab instanceof ArrayBuffer ? ab : ab.buffer;
    const slotBytes = w.size;
    const slotOff   = (ab instanceof ArrayBuffer ? 0 : ab.byteOffset) + w.dataOffset;
    const u32 = new Uint32Array(buf, slotOff, slotBytes / 4);
    const f32 = new Float32Array(buf, slotOff, slotBytes / 4);
    // Atlas block starts at byteOffset 24 / float index 6.
    const ATLAS_OFF_U32 = 24 / 4;
    const pageRef    = u32[ATLAS_OFF_U32 + 0];
    const formatBits = u32[ATLAS_OFF_U32 + 1];
    const originX    = f32[ATLAS_OFF_U32 + 2];
    const originY    = f32[ATLAS_OFF_U32 + 3];
    const sizeX      = f32[ATLAS_OFF_U32 + 4];
    const sizeY      = f32[ATLAS_OFF_U32 + 5];

    expect(pageRef).toBe(0); // first page slot in the bucket
    // formatBits low bit = format index (srgb = 1).
    expect(formatBits! & 0x1).toBe(atlasFormatIndex("rgba8unorm-srgb"));
    // Sampler state: addrU = repeat (1), addrV = clamp (0), mag/min = linear.
    expect((formatBits! >>> 4) & 0x3).toBe(1); // addrU = repeat
    expect((formatBits! >>> 6) & 0x3).toBe(0); // addrV = clamp
    expect((formatBits! >>> 8) & 0x3).toBe(1); // mag = linear
    expect((formatBits! >>> 10) & 0x3).toBe(1); // min = linear

    expect(originX).toBeCloseTo(textures.origin.x, 6);
    expect(originY).toBeCloseTo(textures.origin.y, 6);
    expect(sizeX).toBeCloseTo(textures.size.x, 6);
    expect(sizeY).toBeCloseTo(textures.size.y, 6);
  });

  it("removeDraw doesn't shrink the bucket's page set (MVP — pages stay bound)", () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(gpu.device);
    const { spec: a } = geomSpec(pool, "rgba8unorm");
    const { spec: b } = geomSpec(pool, "rgba8unorm-srgb");
    const scene = buildHeapScene(gpu.device, sig(), [a, b], { atlasPool: pool });
    const idA = 0; // initialDraws are added in array order with sequential ids
    void idA;
    // Count distinct atlas binding-array views *before* and *after*
    // removing one draw: the bucket should still bind both pages.
    const bgBefore = gpu.bindGroups[gpu.bindGroups.length - 1]!;
    const bindingsBefore = (bgBefore.entries as readonly GPUBindGroupEntry[]).map(e => e.binding);
    expect(bindingsBefore).toContain(11);
    expect(bindingsBefore).toContain(12);
    scene.removeDraw(0);
    // No page-set shrink → no fresh bind-group rebuild driven by it.
    // The latest bind-group on record either remains the pre-remove
    // one or a post-resize rebuild that still carries both formats.
    const bgAfter = gpu.bindGroups[gpu.bindGroups.length - 1]!;
    const bindingsAfter = (bgAfter.entries as readonly GPUBindGroupEntry[]).map(e => e.binding);
    expect(bindingsAfter).toContain(11);
    expect(bindingsAfter).toContain(12);
  });
});
