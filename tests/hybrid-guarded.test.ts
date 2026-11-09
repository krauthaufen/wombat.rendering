// `RenderTree.guarded`: a subtree handed over hidden is registered at
// load when reading it does not raise (its first show then a flag flip),
// deferred until shown when it does; once shown a raise is an error.

import { describe, expect, it } from "vitest";
import {
  AdaptiveToken, HashMap, cval, transact, AVal, type aval,
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


const DC: DrawCall = { kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 };

/** A draw call valid only while `valid` holds (a value its guard checked). */
function partial(valid: aval<boolean>): aval<DrawCall> {
  return valid.map((v) => { if (!v) throw new Error("read outside its guard"); return DC; });
}

describe("hybrid scene — guarded subtrees", () => {
  it("a hidden subtree that does not raise is registered at load; showing it is a flag flip", () => {
    const gpu = new MockGPU();
    const active = cval(false);
    const tree = RenderTree.guarded(active, RenderTree.leaf(ro({ active })));
    const scene = compileHybridScene(gpu.device, sig(), tree, {});
    scene.update(AdaptiveToken.top);
    expect(scene.heapTotalDraws()).toBe(1);
    expect(scene.__addDrawCalls()).toBe(1);
    transact(() => { active.value = true; });
    scene.update(AdaptiveToken.top);
    transact(() => { active.value = false; });
    scene.update(AdaptiveToken.top);
    expect(scene.__addDrawCalls()).toBe(1);
    expect(scene.__removeDrawCalls()).toBe(0);
  });

  it("a hidden subtree that raises does not break the load; it is added when shown", () => {
    const gpu = new MockGPU();
    const active = cval(false);
    const valid = cval(false);
    const other = ro();
    const tree = RenderTree.unordered(
      RenderTree.leaf(other),
      RenderTree.guarded(active, RenderTree.leaf(ro({ active, drawCall: partial(valid) }))));
    const scene = compileHybridScene(gpu.device, sig(), tree, {});
    expect(() => scene.update(AdaptiveToken.top)).not.toThrow();
    expect(scene.heapTotalDraws()).toBe(1);
    // a change elsewhere while hidden: still deferred, nothing read
    transact(() => { valid.value = true; });
    scene.update(AdaptiveToken.top);
    expect(scene.heapTotalDraws()).toBe(1);
    // shown (its guard now holds): added as any draw
    transact(() => { active.value = true; });
    scene.update(AdaptiveToken.top);
    expect(scene.heapTotalDraws()).toBe(2);
    // taken in: hiding keeps it, a flag flip
    transact(() => { active.value = false; });
    scene.update(AdaptiveToken.top);
    expect(scene.heapTotalDraws()).toBe(2);
  });

  it("a structure that raises while hidden is deferred too", () => {
    const gpu = new MockGPU();
    const active = cval(false);
    const valid = cval(false);
    const leaf = RenderTree.leaf(ro({ active }));
    const tree = RenderTree.guarded(active, RenderTree.adaptive(valid.map((v) => {
      if (!v) throw new Error("structure outside its guard");
      return leaf;
    })));
    const scene = compileHybridScene(gpu.device, sig(), tree, {});
    expect(() => scene.update(AdaptiveToken.top)).not.toThrow();
    expect(scene.heapTotalDraws()).toBe(0);
    transact(() => { valid.value = true; active.value = true; });
    scene.update(AdaptiveToken.top);
    expect(scene.heapTotalDraws()).toBe(1);
  });

  it("a raise once shown surfaces", () => {
    const gpu = new MockGPU();
    const active = cval(false);
    const valid = cval(false);
    const tree = RenderTree.guarded(active, RenderTree.leaf(ro({ active, drawCall: partial(valid) })));
    const scene = compileHybridScene(gpu.device, sig(), tree, {});
    scene.update(AdaptiveToken.top);
    expect(scene.heapTotalDraws()).toBe(0);
    transact(() => { active.value = true; });
    expect(() => scene.update(AdaptiveToken.top)).toThrow(/outside its guard/);
  });

  it("an active draw that raises surfaces at load", () => {
    const gpu = new MockGPU();
    const active = cval(true);
    const tree = RenderTree.guarded(active, RenderTree.leaf(ro({ active, drawCall: partial(AVal.constant(false)) })));
    expect(() => compileHybridScene(gpu.device, sig(), tree, {}).update(AdaptiveToken.top)).toThrow(/outside its guard/);
  });
});
