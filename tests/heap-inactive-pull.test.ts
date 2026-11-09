// Inactive draws do not pull: a uniform used only by inactive draws is
// not read while it changes, and is read when one of them turns active.
import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, cset, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { buildHeapScene, type HeapDrawSpec } from "../packages/rendering/src/runtime/heapScene.js";
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


/** A draw whose `Color` is its own cell over the shared `tint`, counting reads. */
function spec(effect: ReturnType<typeof makeHeapTestEffect>, tint: aval<number>, active: aval<boolean>, reads: { n: number }): HeapDrawSpec {
  return {
    effect,
    active,
    inputs: {
      Positions: AVal.constant(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0])),
      Normals:   AVal.constant(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1])),
      ModelTrafo:    AVal.constant(trafoIdentity),
      Color:         tint.map((t) => { reads.n++; return v4(t, 0, 0, 1); }),
      ViewProjTrafo: AVal.constant(trafoIdentity),
      LightLocation: AVal.constant(v3(0, 0, 1)),
    },
    indices: AVal.constant(new Uint32Array([0, 1, 2])),
  };
}

describe("heap: inactive draws do not pull", () => {
  it("a uniform of hidden draws is not read while it changes, and is read once on show", () => {
    const gpu = new MockGPU();
    const eff = makeHeapTestEffect();
    const tint = cval(0);
    const hidden = cval(false);
    const reads = { n: 0 };
    const specs = [0, 1, 2].map(() => spec(eff, tint, hidden, reads));
    const scene = buildHeapScene(gpu.device, sig(), specs);
    scene.update(AdaptiveToken.top);
    expect(reads.n).toBe(3);
    for (let i = 1; i <= 10; i++) {
      transact(() => { tint.value = i; });
      scene.update(AdaptiveToken.top);
    }
    expect(reads.n).toBe(3);
    // shown: each read once, in the update that shows them
    transact(() => { hidden.value = true; });
    scene.update(AdaptiveToken.top);
    expect(reads.n).toBe(6);
    transact(() => { tint.value = 11; });
    scene.update(AdaptiveToken.top);
    expect(reads.n).toBe(9);
  });

  it("an aval shared with an active draw keeps being read", () => {
    const gpu = new MockGPU();
    const eff = makeHeapTestEffect();
    const tint = cval(0);
    const reads = { n: 0 };
    const color = tint.map((t) => { reads.n++; return v4(t, 0, 0, 1); });
    const shared = (active: aval<boolean>): HeapDrawSpec => ({ ...spec(eff, tint, active, { n: 0 }), inputs: { ...spec(eff, tint, active, { n: 0 }).inputs, Color: color } });
    const scene = buildHeapScene(gpu.device, sig(), [shared(cval(false)), shared(AVal.constant(true))]);
    scene.update(AdaptiveToken.top);
    const before = reads.n;
    transact(() => { tint.value = 1; });
    scene.update(AdaptiveToken.top);
    expect(reads.n).toBe(before + 1);
  });

  it("a draw removed while its uniform is set aside, then added shown, reads the current value", () => {
    const gpu = new MockGPU();
    const eff = makeHeapTestEffect();
    const tint = cval(0);
    const active = cval(false);
    const reads = { n: 0 };
    const s = spec(eff, tint, active, reads);
    const specs = cset<HeapDrawSpec>([s]);
    const scene = buildHeapScene(gpu.device, sig(), specs);
    scene.update(AdaptiveToken.top);
    transact(() => { tint.value = 1; });
    scene.update(AdaptiveToken.top);
    expect(reads.n).toBe(1);
    transact(() => { specs.remove(s); });
    scene.update(AdaptiveToken.top);
    expect(scene.stats.totalDraws).toBe(0);
    transact(() => { tint.value = 2; active.value = true; specs.add(s); });
    scene.update(AdaptiveToken.top);
    expect(scene.stats.totalDraws).toBe(1);
    expect(reads.n).toBe(2);
    transact(() => { tint.value = 3; });
    scene.update(AdaptiveToken.top);
    expect(reads.n).toBe(3);
  });
});
