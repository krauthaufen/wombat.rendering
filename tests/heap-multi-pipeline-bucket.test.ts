// Phase 5b regression: bucket keying by VALUE (encoded modeKey), not
// by aval identity (the old psIdOf). Pre-5b every distinct cval for
// cullMode produced its own bucket; 20k cvals all valued "back" →
// 20k buckets. Post-5b, identical value collapses into ONE bucket.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, cval, transact } from "@aardworx/wombat.adaptive";
import {
  buildHeapScene, type HeapDrawSpec,
} from "../packages/rendering/src/runtime/heapScene.js";
import type { CullMode, PipelineState } from "../packages/rendering/src/core/pipelineState.js";
import { createFramebufferSignature } from "../packages/rendering/src/resources/framebufferSignature.js";
import { MockGPU } from "./_mockGpu.js";
import { makeHeapTestEffect } from "./_heapTestEffect.js";

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
const trafoIdentity = { forward: {
  toArray: () => IDENTITY44,
  copyTo: (dst: Float32Array | Float64Array | number[], off = 0): void => {
    if (Array.isArray(dst)) { for (let i = 0; i < 16; i++) dst[off + i] = IDENTITY44[i]!; }
    else dst.set(IDENTITY44, off);
  },
} } as unknown;
const v3 = (x: number, y: number, z: number) => ({ x, y, z }) as unknown;
const v4 = (x: number, y: number, z: number, w: number) => ({ x, y, z, w }) as unknown;

const sig = () => createFramebufferSignature({
  colors: { outColor: "rgba8unorm" },
  depthStencil: { format: "depth24plus" },
});

function pipelineStateWith(cullMode: CullMode | typeof cval): PipelineState {
  // Each call constructs FRESH avals — so the aval-identity keying
  // path would fragment them. With value-keying they collapse.
  return {
    rasterizer: {
      topology:  cval<GPUPrimitiveTopology>("triangle-list"),
      cullMode:  cval<CullMode>(cullMode as CullMode),
      frontFace: cval<"ccw" | "cw">("ccw"),
    },
  };
}

function spec(
  effect: ReturnType<typeof makeHeapTestEffect>,
  pipelineState: PipelineState,
): HeapDrawSpec {
  return {
    effect,
    pipelineState,
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

describe("Phase 5b — bucket key uses modeKey VALUE, not aval identity", () => {
  it("N distinct cvals for cullMode, all valued 'back', collapse to ONE bucket", () => {
    const gpu = new MockGPU();
    const eff = makeHeapTestEffect();
    const N = 50; // small enough to be quick; scales to 20k same way
    const specs: HeapDrawSpec[] = [];
    for (let i = 0; i < N; i++) {
      specs.push(spec(eff, pipelineStateWith("back")));
    }
    const scene = buildHeapScene(gpu.device, sig(), specs);
    scene.update(AdaptiveToken.top);
    // ALL N ROs share one bucket — pre-5b this would have been N buckets.
    expect(scene.stats.groups).toBe(1);
  });

  it("two cvals with different VALUES produce two slots in one bucket", () => {
    const gpu = new MockGPU();
    const eff = makeHeapTestEffect();
    const specs: HeapDrawSpec[] = [
      spec(eff, pipelineStateWith("back")),
      spec(eff, pipelineStateWith("front")),
    ];
    const scene = buildHeapScene(gpu.device, sig(), specs);
    scene.update(AdaptiveToken.top);
    // Phase 5c.2: one bucket per (effect, textures); distinct
    // modeKeys now create per-bucket slots, not separate buckets.
    expect(scene.stats.groups).toBe(1);
    expect(scene.stats.slotCount).toBe(2);
  });

  it("mix of identical-value cvals + one different value: 1 bucket / 2 slots", () => {
    const gpu = new MockGPU();
    const eff = makeHeapTestEffect();
    const specs: HeapDrawSpec[] = [];
    for (let i = 0; i < 20; i++) specs.push(spec(eff, pipelineStateWith("back")));
    specs.push(spec(eff, pipelineStateWith("none")));
    for (let i = 0; i < 20; i++) specs.push(spec(eff, pipelineStateWith("back")));
    const scene = buildHeapScene(gpu.device, sig(), specs);
    scene.update(AdaptiveToken.top);
    expect(scene.stats.groups).toBe(1);
    expect(scene.stats.slotCount).toBe(2);
  });

  it("createRenderPipeline is called with the right cullMode value (not aval identity)", () => {
    const gpu = new MockGPU();
    const eff = makeHeapTestEffect();
    // 5 ROs, fresh cval per RO, all valued 'back'. Pre-5b each would
    // have triggered its own createRenderPipeline call with cull
    // 'back'. Post-5b: ONE pipeline created.
    const specs: HeapDrawSpec[] = [];
    for (let i = 0; i < 5; i++) specs.push(spec(eff, pipelineStateWith("back")));
    const beforeBuild = gpu.pipelines.length;
    buildHeapScene(gpu.device, sig(), specs);
    const created = gpu.pipelines.length - beforeBuild;
    // Exactly one heap-render pipeline (the bucket's). The scan
    // pipelines are compute, tracked in `computePipelines` not
    // `pipelines`, so they don't count.
    expect(created).toBe(1);
    const cull = (gpu.pipelines[gpu.pipelines.length - 1]!.primitive!).cullMode;
    expect(cull).toBe("back");
  });

  it("reactive cullMode flip moves the RO into a different slot (same bucket)", () => {
    const gpu = new MockGPU();
    const eff = makeHeapTestEffect();
    // Two ROs, both initially "back" — should share one bucket+slot.
    const cullA = cval<CullMode>("back");
    const cullB = cval<CullMode>("back");
    const psA: PipelineState = {
      rasterizer: {
        topology:  cval<GPUPrimitiveTopology>("triangle-list"),
        cullMode:  cullA,
        frontFace: cval<"ccw" | "cw">("ccw"),
      },
    };
    const psB: PipelineState = {
      rasterizer: {
        topology:  cval<GPUPrimitiveTopology>("triangle-list"),
        cullMode:  cullB,
        frontFace: cval<"ccw" | "cw">("ccw"),
      },
    };
    const scene = buildHeapScene(gpu.device, sig(), [spec(eff, psA), spec(eff, psB)]);
    scene.update(AdaptiveToken.top);
    expect(scene.stats.groups).toBe(1);
    expect(scene.stats.slotCount).toBe(1); // both ROs in slot 'back'

    transact(() => { cullA.value = "front"; });
    scene.update(AdaptiveToken.top);
    expect(scene.stats.groups).toBe(1);    // same bucket
    expect(scene.stats.slotCount).toBe(2); // A in 'front' slot, B still in 'back'

    transact(() => { cullB.value = "front"; });
    scene.update(AdaptiveToken.top);
    expect(scene.stats.groups).toBe(1);
    // Both in 'front' slot now; 'back' slot has 0 records but still
    // exists as a cached slot (v1 doesn't GC empty slots).
    expect(scene.stats.slotCount).toBeGreaterThanOrEqual(1);
  });

  it("bucket-level fast path: 100 ROs sharing one cullCval rebuild a single pipeline (not 100)", () => {
    const gpu = new MockGPU();
    const eff = makeHeapTestEffect();
    // ALL ROs share the same cullCval. Pre-fast-path: flipping it
    // would trigger 100 remove+add cycles, creating 100 fresh
    // pipelines (each addDraw + findOrCreateBucket runs create).
    // With the bucket-level fast path, the whole bucket transitions
    // as a unit -> ONE new pipeline.
    const cullShared = cval<CullMode>("back");
    const ps: PipelineState = {
      rasterizer: {
        topology:  cval<GPUPrimitiveTopology>("triangle-list"),
        cullMode:  cullShared,
        frontFace: cval<"ccw" | "cw">("ccw"),
      },
    };
    const specs: HeapDrawSpec[] = [];
    for (let i = 0; i < 100; i++) specs.push(spec(eff, ps));
    const scene = buildHeapScene(gpu.device, sig(), specs);
    scene.update(AdaptiveToken.top);
    const renderPipelinesBefore = gpu.pipelines.length;

    transact(() => { cullShared.value = "front"; });
    scene.update(AdaptiveToken.top);

    const newPipelines = gpu.pipelines.length - renderPipelinesBefore;
    // Exactly one new pipeline for 'front'; not 100.
    expect(newPipelines).toBe(1);
    // Still one bucket — the existing one was renamed in-place.
    expect(scene.stats.groups).toBe(1);
    const lastDesc = gpu.pipelines[gpu.pipelines.length - 1]!;
    expect(lastDesc.primitive!.cullMode).toBe("front");
  });
});
