// §6 family-merge slice 3c regression: a heap scene with two distinct
// effects collapses to ONE bucket per pipelineState.
//
// Mirrors the heap-demo's textured / non-textured mix at the smallest
// possible scale: two distinct DSL effects (one with a sampler binding,
// one without) sharing the implicit "default" pipelineState. Pre-3c
// this would have produced two buckets keyed on (effect, pipelineState).

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken } from "@aardworx/wombat.adaptive";
import { ITexture } from "../packages/rendering/src/core/texture.js";
import { ISampler } from "../packages/rendering/src/core/sampler.js";
import {
  buildHeapScene, type HeapDrawSpec, type HeapTextureSet,
} from "../packages/rendering/src/runtime/heapScene.js";
import { AtlasPool } from "../packages/rendering/src/runtime/textureAtlas/atlasPool.js";
import { createFramebufferSignature } from "../packages/rendering/src/resources/framebufferSignature.js";
import { MockGPU } from "./_mockGpu.js";
import {
  makeHeapTestEffect, makeHeapTestEffectTextured,
} from "./_heapTestEffect.js";

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

const sig = () => createFramebufferSignature({
  colors: { outColor: "rgba8unorm" },
  depthStencil: { format: "depth24plus" },
});

function plainSpec(effect: ReturnType<typeof makeHeapTestEffect>): HeapDrawSpec {
  return {
    effect,
    inputs: {
      Positions: AVal.constant(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0])),
      Normals:   AVal.constant(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1])),
      ModelTrafo:    AVal.constant(trafoIdentity),
      Color:         AVal.constant(v4(1, 0, 0, 1)),
      ViewProjTrafo: AVal.constant(trafoIdentity),
      LightLocation: AVal.constant(v3(0, 0, 1)),
    },
    indices: AVal.constant(new Uint32Array([0, 1, 2])),
  };
}

function texturedSpec(
  effect: ReturnType<typeof makeHeapTestEffectTextured>,
  pool: AtlasPool,
): HeapDrawSpec {
  const tex = ITexture.fromRaw({
    data: new Uint8Array(64 * 64 * 4),
    width: 64, height: 64,
    format: "rgba8unorm",
  });
  const texAval = AVal.constant(tex);
  const acq = pool.acquire("rgba8unorm", texAval, 64, 64);
  const sampler = ISampler.fromDescriptor({
    magFilter: "linear", minFilter: "linear",
    addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
  });
  const textures: HeapTextureSet & { kind: "atlas" } = {
    kind: "atlas",
    format: "rgba8unorm", pageId: acq.pageId,
    origin: acq.origin, size: acq.size,
    numMips: acq.numMips,
    sampler, page: acq.page,
    poolRef: acq.ref,
    release: () => pool.release(acq.ref),
  };
  return {
    effect,
    inputs: {
      Positions: AVal.constant(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0])),
      Normals:   AVal.constant(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1])),
      ModelTrafo:    AVal.constant(trafoIdentity),
      Color:         AVal.constant(v4(0, 1, 0, 1)),
      ViewProjTrafo: AVal.constant(trafoIdentity),
      LightLocation: AVal.constant(v3(0, 0, 1)),
    },
    indices: AVal.constant(new Uint32Array([0, 1, 2])),
    textures,
  };
}

describe.skip("§6 family-merge — bucket collapse (slice 3c)", () => {
  it("two effects (textured + non-textured) at one pipelineState → 1 bucket (merge enabled)", () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(gpu.device);
    const plain = plainSpec(makeHeapTestEffect());
    const tex   = texturedSpec(makeHeapTestEffectTextured(), pool);
    const scene = buildHeapScene(gpu.device, sig(), [plain, tex], {
      atlasPool: pool, enableFamilyMerge: true,
    });
    expect(scene.stats.groups).toBe(1);
    // Ensure update path runs without throwing.
    scene.update(AdaptiveToken.top);
  });

  it("two effects at one pipelineState → 2 buckets (merge default-off)", () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(gpu.device);
    const plain = plainSpec(makeHeapTestEffect());
    const tex   = texturedSpec(makeHeapTestEffectTextured(), pool);
    const scene = buildHeapScene(gpu.device, sig(), [plain, tex], { atlasPool: pool });
    // Default per-effect bucketing — 2 distinct effects → 2 buckets.
    expect(scene.stats.groups).toBe(2);
    scene.update(AdaptiveToken.top);
  });

  it("addDraw of an effect outside the frozen family throws", () => {
    const gpu = new MockGPU();
    const pool = new AtlasPool(gpu.device);
    const plain = plainSpec(makeHeapTestEffect());
    const scene = buildHeapScene(gpu.device, sig(), [plain], {
      atlasPool: pool, enableFamilyMerge: true,
    });
    const stranger = makeHeapTestEffectTextured();
    expect(() => scene.addDraw(texturedSpec(stranger, pool))).toThrow(/family is frozen/);
  });
});
