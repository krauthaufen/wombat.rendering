// heapEligibility — escalation-to-legacy texture rules.
//
// HybridScene partitions ROs by `isHeapEligible`. ROs whose textures
// can't be served by the heap path (backend-managed `kind: "gpu"`,
// oversized, or non-2D / array-layered) must escalate to the legacy
// per-RO ScenePass. These tests pin the predicate's behavior on each
// of those branches without spinning up a real GPU.

import { describe, expect, it } from "vitest";
import {
  AdaptiveToken, HashMap, cval, type aval,
} from "@aardworx/wombat.adaptive";
import { IBuffer } from "../packages/rendering/src/core/buffer.js";
import { ITexture } from "../packages/rendering/src/core/texture.js";
import { ISampler } from "../packages/rendering/src/core/sampler.js";
import { PipelineState } from "../packages/rendering/src/core/pipelineState.js";
import { ElementType } from "../packages/rendering/src/core/elementType.js";
import { AttributeProvider, UniformProvider } from "../packages/rendering/src/core/provider.js";
import type { BufferView } from "../packages/rendering/src/core/bufferView.js";
import type { DrawCall } from "../packages/rendering/src/core/drawCall.js";
import type { Effect } from "../packages/rendering/src/core/shader.js";
import type { RenderObject } from "../packages/rendering/src/core/renderObject.js";
import { isHeapEligible } from "../packages/rendering/src/runtime/heapEligibility.js";

// Eligibility never reads the effect / pipelineState — we just need
// a placeholder for typing. `as unknown as Effect` is safe because
// the predicate doesn't touch it.
const fakeEffect = {} as unknown as Effect;
const fakePipeline = PipelineState.constant({
  rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" },
});

function baseRO(extras: Partial<RenderObject> = {}): RenderObject {
  // 3-vertex non-instanced indexed draw with one V3f position
  // attribute (host-side) — the minimum that survives every other
  // wedge in `isHeapEligible`.
  const vbuf = cval<IBuffer>(IBuffer.fromHost(new ArrayBuffer(36)));
  const view: BufferView = {
    buffer: vbuf, offset: 0, stride: 12, elementType: ElementType.V3f,
  };
  const ibuf = cval<IBuffer>(IBuffer.fromHost(new Uint32Array([0, 1, 2]).buffer));
  const indices: BufferView = {
    buffer: ibuf, offset: 0, stride: 4, elementType: ElementType.U32,
  };
  return {
    effect: fakeEffect,
    pipelineState: fakePipeline,
    vertexAttributes: AttributeProvider.ofObject({ position: view }),
    uniforms: UniformProvider.empty,
    textures: HashMap.empty(),
    samplers: HashMap.empty(),
    indices,
    drawCall: cval<DrawCall>({
      kind: "indexed", indexCount: 3, instanceCount: 1,
      firstIndex: 0, baseVertex: 0, firstInstance: 0,
    }),
    ...extras,
  };
}

// Stub a `GPUSampler` — eligibility just sniffs `kind: "gpu"` on
// the sampler value, no fields read.
const stubSampler = ISampler.fromGPU({} as unknown as GPUSampler);

function withTexSampler(tex: ITexture, sampler: ISampler = stubSampler): RenderObject {
  const t: aval<ITexture> = cval(tex);
  const s: aval<ISampler> = cval(sampler);
  return baseRO({
    textures: HashMap.empty<string, aval<ITexture>>().add("tex", t),
    samplers: HashMap.empty<string, aval<ISampler>>().add("samp", s),
  });
}

/**
 * Stub a `GPUTexture` with just the fields heapEligibility reads.
 * The eligibility predicate currently only sniffs `kind: "gpu"` —
 * dimensions/layers/format aren't probed for the gpu branch
 * (rule 1 short-circuits). We still parameterize them for the
 * cubemap case for documentation / future use if rule 1 relaxes.
 */
function stubGPUTexture(opts: {
  width?: number; height?: number;
  dimension?: GPUTextureDimension;
  depthOrArrayLayers?: number;
  format?: GPUTextureFormat;
} = {}): GPUTexture {
  return {
    width: opts.width ?? 256,
    height: opts.height ?? 256,
    depthOrArrayLayers: opts.depthOrArrayLayers ?? 1,
    dimension: opts.dimension ?? "2d",
    format: opts.format ?? "rgba8unorm",
    mipLevelCount: 1,
    sampleCount: 1,
    usage: 0,
    label: "stub",
    destroy() {},
    createView() { return {} as GPUTextureView; },
  } as unknown as GPUTexture;
}

describe("isHeapEligible — texture escalation rules", () => {
  it("RO with no textures is eligible", () => {
    const ro = baseRO();
    expect(isHeapEligible(ro).getValue(AdaptiveToken.top)).toBe(true);
  });

  it("RO with a small host rgba8 texture is eligible", () => {
    const tex = ITexture.fromRaw({
      data: new Uint8Array(256 * 256 * 4),
      width: 256, height: 256, format: "rgba8unorm",
    });
    expect(isHeapEligible(withTexSampler(tex)).getValue(AdaptiveToken.top)).toBe(true);
  });

  it("RO with an oversized host texture (5000×500) is ineligible", () => {
    const tex = ITexture.fromRaw({
      data: new Uint8Array(4),     // payload size doesn't matter for the predicate
      width: 5000, height: 500, format: "rgba8unorm",
    });
    expect(isHeapEligible(withTexSampler(tex)).getValue(AdaptiveToken.top)).toBe(false);
  });

  it("RO with a kind:'gpu' texture is ineligible (backend-managed)", () => {
    const tex = ITexture.fromGPU(stubGPUTexture());
    expect(isHeapEligible(withTexSampler(tex)).getValue(AdaptiveToken.top)).toBe(false);
  });

  it("RO with a kind:'gpu' cubemap (depthOrArrayLayers=6) is ineligible", () => {
    const tex = ITexture.fromGPU(stubGPUTexture({
      depthOrArrayLayers: 6, dimension: "2d",
    }));
    expect(isHeapEligible(withTexSampler(tex)).getValue(AdaptiveToken.top)).toBe(false);
  });
});

describe("isHeapEligible — per-RO instancing", () => {
  it("RO with tight-stride instance attributes is eligible", () => {
    const ibuf = cval<IBuffer>(IBuffer.fromHost(new ArrayBuffer(48)));
    const view: BufferView = {
      buffer: ibuf, offset: 0, stride: 12, elementType: ElementType.V3f,
    };
    const ro = baseRO({
      instanceAttributes: AttributeProvider.ofObject({ offset: view }),
      drawCall: cval<DrawCall>({
        kind: "indexed", indexCount: 3, instanceCount: 4,
        firstIndex: 0, baseVertex: 0, firstInstance: 0,
      }),
    });
    expect(isHeapEligible(ro).getValue(AdaptiveToken.top)).toBe(true);
  });

  it("RO with high instanceCount and no per-instance attrs is eligible", () => {
    const ro = baseRO({
      drawCall: cval<DrawCall>({
        kind: "indexed", indexCount: 3, instanceCount: 1000,
        firstIndex: 0, baseVertex: 0, firstInstance: 0,
      }),
    });
    expect(isHeapEligible(ro).getValue(AdaptiveToken.top)).toBe(true);
  });

  it("RO with non-tight stride on instance attribute is ineligible", () => {
    const ibuf = cval<IBuffer>(IBuffer.fromHost(new ArrayBuffer(64)));
    const view: BufferView = {
      // V3f source (12B) but stride 16 — interleaved, not tight, not broadcast.
      buffer: ibuf, offset: 0, stride: 16, elementType: ElementType.V3f,
    };
    const ro = baseRO({
      instanceAttributes: AttributeProvider.ofObject({ offset: view }),
      drawCall: cval<DrawCall>({
        kind: "indexed", indexCount: 3, instanceCount: 4,
        firstIndex: 0, baseVertex: 0, firstInstance: 0,
      }),
    });
    expect(isHeapEligible(ro).getValue(AdaptiveToken.top)).toBe(false);
  });

  it("RO with firstInstance != 0 is ineligible (still rejected)", () => {
    const ro = baseRO({
      drawCall: cval<DrawCall>({
        kind: "indexed", indexCount: 3, instanceCount: 4,
        firstIndex: 0, baseVertex: 0, firstInstance: 1,
      }),
    });
    expect(isHeapEligible(ro).getValue(AdaptiveToken.top)).toBe(false);
  });
});
