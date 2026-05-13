// End-to-end test: a HeapDrawSpec with a `modeRules.cull` rule that
// reads a uniform value routes each RO to the rule-evaluated bucket.
// Mutating the input uniform reactively rebuckets via the dirty-mode-
// key flush in heapScene.update().

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, cval, transact } from "@aardworx/wombat.adaptive";
import {
  buildHeapScene, type HeapDrawSpec,
} from "../packages/rendering/src/runtime/heapScene.js";
import {
  derivedMode, flipCull,
} from "@aardworx/wombat.rendering/runtime";
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

function specWithRule(
  effect: ReturnType<typeof makeHeapTestEffect>,
  mirroredFlag: ReturnType<typeof cval<boolean>>,
): HeapDrawSpec {
  return {
    effect,
    inputs: {
      Positions: AVal.constant(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0])),
      Normals:   AVal.constant(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1])),
      ModelTrafo:    AVal.constant(trafoIdentity),
      Color:         AVal.constant(v4(1, 0, 0, 1)),
      ViewProjTrafo: AVal.constant(trafoIdentity),
      LightLocation: AVal.constant(v3(0, 0, 1)),
      // Custom uniform the rule reads. Not declared in the test
      // effect's schema; the rule's proxy reads it through
      // spec.inputs map. (For "real" rules where the uniform is
      // also consumed by the shader, the user would put it in
      // spec.inputs as today.)
      Mirrored: mirroredFlag,
    },
    indices: AVal.constant(new Uint32Array([0, 1, 2])),
    modeRules: {
      cull: derivedMode("cull", (u, declared) =>
        u.Mirrored ? flipCull(declared) : declared),
    },
  };
}

describe("Task 2: derivedMode integration through heap scene", () => {
  it("rule output partitions ROs into per-output buckets", () => {
    const gpu = new MockGPU();
    const eff = makeHeapTestEffect();
    const a = cval<boolean>(false);   // -> declared (back)
    const b = cval<boolean>(true);    // -> flipped (front)
    const scene = buildHeapScene(gpu.device, sig(), [
      specWithRule(eff, a),
      specWithRule(eff, b),
    ]);
    scene.update(AdaptiveToken.top);
    // Phase 5c.2: one bucket per (effect, textures); distinct rule
    // outputs split into per-bucket slots.
    expect(scene.stats.groups).toBe(1);
    expect(scene.stats.slotCount).toBe(2);
    // Pipelines reflect rule output.
    const culls = gpu.pipelines.map(p => p.primitive!.cullMode).filter(c => c === "back" || c === "front");
    expect(new Set(culls)).toEqual(new Set(["back", "front"]));
  });

  it("rule sharing one input collapses into one bucket and a single pipeline", () => {
    const gpu = new MockGPU();
    const eff = makeHeapTestEffect();
    // Both ROs read the SAME 'Mirrored' aval. Both evaluate to the
    // same rule output -> same modeKey -> one bucket.
    const shared = cval<boolean>(true);
    const scene = buildHeapScene(gpu.device, sig(), [
      specWithRule(eff, shared),
      specWithRule(eff, shared),
      specWithRule(eff, shared),
    ]);
    scene.update(AdaptiveToken.top);
    expect(scene.stats.groups).toBe(1);
  });

  it("flipping the rule-input aval reactively rebuckets via dirty pass", () => {
    const gpu = new MockGPU();
    const eff = makeHeapTestEffect();
    const flag = cval<boolean>(false);
    const scene = buildHeapScene(gpu.device, sig(), [
      specWithRule(eff, flag),
      specWithRule(eff, flag),
    ]);
    scene.update(AdaptiveToken.top);
    expect(scene.stats.groups).toBe(1);
    const culls0 = gpu.pipelines.map(p => p.primitive!.cullMode);
    expect(culls0[culls0.length - 1]).toBe("back");

    transact(() => { flag.value = true; });
    scene.update(AdaptiveToken.top);
    expect(scene.stats.groups).toBe(1);
    const lastCull = gpu.pipelines[gpu.pipelines.length - 1]!.primitive!.cullMode;
    expect(lastCull).toBe("front");
  });
});
