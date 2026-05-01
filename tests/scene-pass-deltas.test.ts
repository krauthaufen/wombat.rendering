// ScenePass / NodeWalker — verifies the delta-driven scene
// resolution. Build a tree with N static leaves, then run M frames
// and confirm `prepareRenderObject` was called exactly N times
// (not N×M like the old per-frame `collectLeaves` walker).
//
// Then mutate a `clist<RenderTree>` subtree by appending one leaf
// and confirm exactly one extra prepare happens.

import { describe, expect, it } from "vitest";
import {
  AList,
  AdaptiveToken,
  AVal,
  HashMap,
  ChangeableIndexListOps as CList,
  cval,
  clist,
  cset,
  ChangeableHashSetOps as CSet,
  transact,
  type aval,
} from "@aardworx/wombat.adaptive";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import {
  IBuffer,
  RenderTree,
  type BufferView,
  type Command,
  type DrawCall,
  type RenderObject,
  PipelineState,
} from "@aardworx/wombat.rendering/core";
import {
  allocateFramebuffer,
  createFramebufferSignature,
} from "@aardworx/wombat.rendering/resources";
import { Runtime, ScenePass } from "@aardworx/wombat.rendering/runtime";
import { makeEffect } from "./_makeEffect.js";
import { MockGPU } from "./_mockGpu.js";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

function flatRedEffect() {
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

function makeRO(eff = flatRedEffect()): RenderObject {
  return {
    effect: eff,
    pipelineState: PipelineState.constant({ rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } }),
    vertexAttributes: AVal.constant(HashMap.empty<string, aval<BufferView>>().add("position", cval<BufferView>({
      buffer: IBuffer.fromHost(new ArrayBuffer(36)), offset: 0, count: 3, stride: 12, format: "float32x3",
    }))),
    uniforms: HashMap.empty(),
    textures: HashMap.empty(),
    samplers: HashMap.empty(),
    drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
  };
}

describe("ScenePass: delta-driven resolution", () => {
  it("static N-leaf scene: N prepareRenderObject calls total across many frames", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const eff = flatRedEffect();
    const N = 7;
    const leaves = Array.from({ length: N }, () => RenderTree.leaf(makeRO(eff)));
    const tree = RenderTree.ordered(...leaves);

    const scene = new ScenePass(gpu.device, sig, tree, e => e.compile({ target: "wgsl" }));
    expect(scene.stats.prepareCount).toBe(N);   // built at construction

    // Resolve M frames; prepareCount must not budge.
    for (let i = 0; i < 5; i++) {
      const out = scene.resolve(AdaptiveToken.top);
      expect(out).toHaveLength(N);
    }
    expect(scene.stats.prepareCount).toBe(N);

    scene.dispose();
  });

  it("clist insertion: exactly one new prepareRenderObject call", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const eff = flatRedEffect();
    const list = clist<RenderTree>([
      RenderTree.leaf(makeRO(eff)),
      RenderTree.leaf(makeRO(eff)),
      RenderTree.leaf(makeRO(eff)),
    ]);
    const tree = RenderTree.orderedFromList(list);

    const scene = new ScenePass(gpu.device, sig, tree, e => e.compile({ target: "wgsl" }));
    expect(scene.stats.prepareCount).toBe(0);   // dynamic — nothing yet

    scene.resolve(AdaptiveToken.top);            // first read drains the reader
    expect(scene.stats.prepareCount).toBe(3);

    // Append one. Only one new prep should happen on the next resolve.
    transact(() => CList.add(list, RenderTree.leaf(makeRO(eff))));
    const before = scene.stats.prepareCount;
    scene.resolve(AdaptiveToken.top);
    expect(scene.stats.prepareCount).toBe(before + 1);

    // No-delta resolve: zero extra preps.
    scene.resolve(AdaptiveToken.top);
    scene.resolve(AdaptiveToken.top);
    expect(scene.stats.prepareCount).toBe(before + 1);

    // Remove one. No NEW prep, but disposed walker means the leaves
    // count in the resolved list drops.
    transact(() => CList.removeAt(list, 0));
    const out = scene.resolve(AdaptiveToken.top);
    expect(out).toHaveLength(3);
    expect(scene.stats.prepareCount).toBe(before + 1);

    scene.dispose();
  });

  it("cset insertion mirrors clist behaviour", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const eff = flatRedEffect();
    const set = cset<RenderTree>([
      RenderTree.leaf(makeRO(eff)),
      RenderTree.leaf(makeRO(eff)),
    ]);
    const scene = new ScenePass(
      gpu.device, sig,
      RenderTree.unorderedFromSet(set),
      e => e.compile({ target: "wgsl" }),
    );
    scene.resolve(AdaptiveToken.top);
    expect(scene.stats.prepareCount).toBe(2);

    transact(() => CSet.add(set, RenderTree.leaf(makeRO(eff))));
    scene.resolve(AdaptiveToken.top);
    expect(scene.stats.prepareCount).toBe(3);

    scene.resolve(AdaptiveToken.top);
    expect(scene.stats.prepareCount).toBe(3);

    scene.dispose();
  });

  it("RenderTask: per-frame run() of static scene incurs zero new preps after the first", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device });
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();

    const eff = flatRedEffect();
    const tree = RenderTree.ordered(
      RenderTree.leaf(makeRO(eff)),
      RenderTree.leaf(makeRO(eff)),
      RenderTree.leaf(makeRO(eff)),
    );
    const cmds = AList.ofArray<Command>([{ kind: "Render", output: fbo, tree }]);
    const task = runtime.compile(cmds);

    task.run(AdaptiveToken.top);
    const pipelinesAfterFirstFrame = gpu.pipelines.length;
    task.run(AdaptiveToken.top);
    task.run(AdaptiveToken.top);
    task.run(AdaptiveToken.top);
    // Three flat-red leaves with the same effect+sig+state share
    // a pipeline; the cache held it stable across frames.
    expect(gpu.pipelines.length).toBe(pipelinesAfterFirstFrame);

    task.dispose();
    fbo.release();
  });
});
