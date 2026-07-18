// The row store's hybrid consumption: a `Rows` node feeds the heap
// directly (one choose stage per set — no per-row RenderObject in the
// flat pipeline), reacts to row add/remove, and falls back to the
// legacy path only when the global heap toggle is off.

import { describe, expect, it } from "vitest";
import {
  AdaptiveToken, HashMap, cval, cset, transact, AVal,
  ChangeableHashSetOps as CSet,
} from "@aardworx/wombat.adaptive";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import {
  IBuffer,
  ElementType,
  PipelineState,
  type BufferView,
  type DrawCall,
  type RenderObject,
} from "@aardworx/wombat.rendering/core";
import { RenderTree } from "../packages/rendering/src/core/renderTree.js";
import type { RenderRow, RenderRowSet } from "../packages/rendering/src/core/rowSet.js";
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

const dc: DrawCall = {
  kind: "non-indexed", vertexCount: 3, instanceCount: 1,
  firstVertex: 0, firstInstance: 0,
};

function templateRO(): RenderObject {
  const view: BufferView = {
    buffer: AVal.constant(IBuffer.fromHost(new Float32Array(9).buffer)),
    offset: 0, stride: 12, elementType: ElementType.V3f,
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
    drawCall: AVal.constant(dc),
    heapAsserted: true,
  };
}

function row(pickId: number): RenderRow {
  return {
    uniforms: UniformProvider.empty,
    drawCall: AVal.constant(dc),
    pickId,
  };
}

const sig = () => createFramebufferSignature({
  colors: { outColor: "rgba8unorm" },
  depthStencil: { format: "depth24plus" },
});

describe("hybrid — Rows direct consumption", () => {
  it("rows feed the heap; add/remove reacts; toggle falls back to legacy", () => {
    const gpu = new MockGPU();
    const rows = cset<RenderRow>(new Set([row(1), row(2), row(3)]));
    const set: RenderRowSet = { template: templateRO(), rows };
    const heapEnabled = cval(true);
    const tree = RenderTree.unordered(RenderTree.rows(set));
    const scene = compileHybridScene(gpu.device, sig(), tree, { heapEnabled });

    scene.update(AdaptiveToken.top);
    expect(scene.heapTotalDraws()).toBe(3);
    expect(scene.__legacyCount()).toBe(0);

    const extra = row(4);
    transact(() => { CSet.add(rows, extra); });
    scene.update(AdaptiveToken.top);
    expect(scene.heapTotalDraws()).toBe(4);

    transact(() => { CSet.remove(rows, extra); });
    scene.update(AdaptiveToken.top);
    expect(scene.heapTotalDraws()).toBe(3);

    // Global toggle off → rows render through the legacy ScenePass.
    transact(() => { heapEnabled.value = false; });
    scene.update(AdaptiveToken.top);
    expect(scene.heapTotalDraws()).toBe(0);
    expect(scene.__legacyCount()).toBe(3);

    transact(() => { heapEnabled.value = true; });
    scene.update(AdaptiveToken.top);
    expect(scene.heapTotalDraws()).toBe(3);
    expect(scene.__legacyCount()).toBe(0);
  });
});
