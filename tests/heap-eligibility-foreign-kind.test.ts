// Regression: `isHeapServableTexture` must never THROW.
//
// It used to end its kind-dispatch with a bare `const src = t.source`,
// assuming anything that wasn't "gpu"/"url" was a host texture. Two ways
// that blew up in production:
//
//   1. `kind: "cube"` — added later. An older build of this package that
//      predates cube support reaches `t.source` on a cube ITexture, which
//      has no `source` field → `TypeError: ... reading 'kind'`.
//   2. Version skew generally — a duplicate/nested install means ITexture
//      values built by ANOTHER copy of this package flow into this one and
//      can carry any discriminant.
//
// The throw happened inside the per-frame callback, so it did not merely
// misclassify one texture: it killed the render loop (observed as a viewer
// that never came back after a GPU device loss). Eligibility is a
// PREDICATE — an unrecognised kind must answer "not heap-servable" and let
// the legacy path deal with it, never throw.

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

const fakeEffect = {} as unknown as Effect;
const fakePipeline = PipelineState.constant({
  rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" },
});

function roWithTexture(tex: ITexture): RenderObject {
  const vbuf = cval<IBuffer>(IBuffer.fromHost(new ArrayBuffer(36)));
  const view: BufferView = { buffer: vbuf, offset: 0, stride: 12, elementType: ElementType.V3f };
  const ibuf = cval<IBuffer>(IBuffer.fromHost(new Uint32Array([0, 1, 2]).buffer));
  const indices: BufferView = { buffer: ibuf, offset: 0, stride: 4, elementType: ElementType.U32 };
  const t: aval<ITexture> = cval(tex);
  const s: aval<ISampler> = cval(ISampler.fromGPU({} as unknown as GPUSampler));
  return {
    effect: fakeEffect,
    pipelineState: fakePipeline,
    vertexAttributes: AttributeProvider.ofObject({ position: view }),
    uniforms: UniformProvider.empty,
    textures: HashMap.empty<string, aval<ITexture>>().add("tex", t),
    samplers: HashMap.empty<string, aval<ISampler>>().add("samp", s),
    indices,
    drawCall: cval<DrawCall>({
      kind: "indexed", indexCount: 3, instanceCount: 1,
      firstIndex: 0, baseVertex: 0, firstInstance: 0,
    }),
  };
}

function faceCanvas(): OffscreenCanvas {
  // Node has OffscreenCanvas in recent versions; fall back to a duck-typed
  // stand-in since eligibility only ever reads width/height off a face.
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(8, 8);
  return { width: 8, height: 8 } as unknown as OffscreenCanvas;
}

describe("isHeapEligible — never throws on unexpected texture kinds", () => {
  it("classifies a cube ITexture as heap-servable", () => {
    const faces = Array.from({ length: 6 }, faceCanvas);
    const tex = ITexture.fromCubeFaces(faces);
    expect(isHeapEligible(roWithTexture(tex)).getValue(AdaptiveToken.top)).toBe(true);
  });

  it("does not throw on a foreign discriminant (duplicate-install skew)", () => {
    // Exactly the shape a nested/older copy of this package produces: a
    // discriminant this build has no branch for, and no `source` field.
    const foreign = { kind: "someFutureKind" } as unknown as ITexture;
    const ro = roWithTexture(foreign);
    expect(() => isHeapEligible(ro).getValue(AdaptiveToken.top)).not.toThrow();
    expect(isHeapEligible(ro).getValue(AdaptiveToken.top)).toBe(false);
  });

  it("does not throw when a host texture has no source", () => {
    const malformed = { kind: "host" } as unknown as ITexture;
    const ro = roWithTexture(malformed);
    expect(() => isHeapEligible(ro).getValue(AdaptiveToken.top)).not.toThrow();
    expect(isHeapEligible(ro).getValue(AdaptiveToken.top)).toBe(false);
  });
});
