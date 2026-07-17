// `RenderObject.heapAsserted` — producer-asserted heap eligibility.
// Asserted ROs must route to the heap path without the per-RO
// predicate (their adaptive drawCall would otherwise force a live
// custom aval each) and must still honour the scene-level
// `heapEnabled` toggle in BOTH directions.

import { describe, expect, it } from "vitest";
import {
  AdaptiveToken, HashMap, cval, transact, AVal,
} from "@aardworx/wombat.adaptive";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import {
  IBuffer,
  RenderTree,
  ElementType,
  PipelineState,
  type BufferView,
  type DrawCall,
  type RenderObject,
} from "@aardworx/wombat.rendering/core";
import { AttributeProvider, UniformProvider } from "../packages/rendering/src/core/provider.js";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { compileHybridScene } from "../packages/rendering/src/runtime/hybridScene.js";
import { makeEffect } from "./_makeEffect.js";
import { MockGPU } from "./_mockGpu.js";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

function flatEffect() {
  return makeEffect(
    `
      function vsMain(input: { position: V3f }): { gl_Position: V4f } {
        return { gl_Position: new V4f(input.position.x, input.position.y, input.position.z, 1.0) };
      }
      function fsMain(_input: {}): { outColor: V4f } {
        return { outColor: new V4f(1.0, 0.0, 0.0, 1.0) };
      }
    `,
    [
      { name: "vsMain", stage: "vertex",
        inputs: [{ name: "position", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] }],
      },
      { name: "fsMain", stage: "fragment",
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ],
  );
}

function ro(extras: Partial<RenderObject> = {}): RenderObject {
  const vbuf = AVal.constant(IBuffer.fromHost(new Float32Array(9).buffer));
  const view: BufferView = {
    buffer: vbuf, offset: 0, stride: 12, elementType: ElementType.V3f,
  };
  return {
    effect: flatEffect(),
    pipelineState: PipelineState.constant({
      rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" },
    }),
    vertexAttributes: AttributeProvider.ofObject({ position: view }),
    uniforms: UniformProvider.empty,
    textures: HashMap.empty(),
    samplers: HashMap.empty(),
    drawCall: AVal.constant<DrawCall>({
      kind: "non-indexed", vertexCount: 3, instanceCount: 1,
      firstVertex: 0, firstInstance: 0,
    }),
    ...extras,
  };
}

const sig = () => createFramebufferSignature({
  colors: { outColor: "rgba8unorm" },
  depthStencil: { format: "depth24plus" },
});

describe("hybrid partition — heapAsserted", () => {
  it("asserted RO with an ADAPTIVE drawCall rides the heap and follows heapEnabled", () => {
    const gpu = new MockGPU();
    const dc = cval<DrawCall>({
      kind: "non-indexed", vertexCount: 3, instanceCount: 1,
      firstVertex: 0, firstInstance: 0,
    });
    const asserted = ro({ drawCall: dc, heapAsserted: true });
    const plain = ro();
    const heapEnabled = cval(true);
    const tree = RenderTree.unordered(RenderTree.leaf(asserted), RenderTree.leaf(plain));
    const scene = compileHybridScene(gpu.device, sig(), tree, { heapEnabled });

    scene.update(AdaptiveToken.top);
    expect(scene.heapTotalDraws()).toBe(2);
    expect(scene.__legacyCount()).toBe(0);

    // Global toggle off → BOTH (asserted included) route to legacy.
    transact(() => { heapEnabled.value = false; });
    scene.update(AdaptiveToken.top);
    expect(scene.heapTotalDraws()).toBe(0);
    expect(scene.__legacyCount()).toBe(2);

    // Back on → asserted returns to the heap.
    transact(() => { heapEnabled.value = true; });
    scene.update(AdaptiveToken.top);
    expect(scene.heapTotalDraws()).toBe(2);
    expect(scene.__legacyCount()).toBe(0);
  });

  it("asserted instanceCount 0 stays on the heap and emits nothing", () => {
    const gpu = new MockGPU();
    const dc = cval<DrawCall>({
      kind: "non-indexed", vertexCount: 3, instanceCount: 0,
      firstVertex: 0, firstInstance: 0,
    });
    const asserted = ro({ drawCall: dc, heapAsserted: true });
    const scene = compileHybridScene(
      gpu.device, sig(), RenderTree.leaf(asserted), {});
    scene.update(AdaptiveToken.top);
    // On the heap (not escalated to legacy), but drawing nothing.
    expect(scene.heapTotalDraws()).toBe(1);
    expect(scene.__legacyCount()).toBe(0);
  });
});
