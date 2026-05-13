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

  it("two cvals with different VALUES produce two buckets", () => {
    const gpu = new MockGPU();
    const eff = makeHeapTestEffect();
    const specs: HeapDrawSpec[] = [
      spec(eff, pipelineStateWith("back")),
      spec(eff, pipelineStateWith("front")),
    ];
    const scene = buildHeapScene(gpu.device, sig(), specs);
    scene.update(AdaptiveToken.top);
    // Distinct values → distinct buckets, as expected.
    expect(scene.stats.groups).toBe(2);
  });

  it("a mix of identical-value cvals + one different value: 2 buckets", () => {
    const gpu = new MockGPU();
    const eff = makeHeapTestEffect();
    const specs: HeapDrawSpec[] = [];
    for (let i = 0; i < 20; i++) specs.push(spec(eff, pipelineStateWith("back")));
    specs.push(spec(eff, pipelineStateWith("none")));
    for (let i = 0; i < 20; i++) specs.push(spec(eff, pipelineStateWith("back")));
    const scene = buildHeapScene(gpu.device, sig(), specs);
    scene.update(AdaptiveToken.top);
    expect(scene.stats.groups).toBe(2);
  });

  it("reactive cullMode flip: cval mutation moves the RO to a different bucket", () => {
    const gpu = new MockGPU();
    const eff = makeHeapTestEffect();
    // Two ROs, both initially "back" — should share one bucket.
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
    expect(scene.stats.groups).toBe(1); // collapse — both back.

    // Flip A's cullMode value. Pre-5b.2, this was silently ignored
    // (bucket key was identity-based; aval identity unchanged).
    // With reactive rebucket, the modeKey changes -> RO A moves
    // to a fresh bucket with cullMode='front'.
    transact(() => { cullA.value = "front"; });
    scene.update(AdaptiveToken.top);
    expect(scene.stats.groups).toBe(2); // A and B now in distinct buckets.

    // Flip B to also front; both end up in the same bucket again
    // and the now-empty 'back' bucket is GC'd by removeDraw.
    transact(() => { cullB.value = "front"; });
    scene.update(AdaptiveToken.top);
    expect(scene.stats.groups).toBe(1);
  });
});
