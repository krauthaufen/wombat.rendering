// GPU transform propagation — CPU-side structural tests for the suffix trie:
// sibling chains share ancestor trie nodes (prefix sharing), and nodes are
// refcounted/freed on add/remove. The df32 math + level dispatch are verified
// on a real GPU in tests-browser/transform-chain-real.test.ts.

import { describe, it, expect } from "vitest";
import { AVal, cval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d } from "@aardworx/wombat.base";
import { MockGPU } from "./_mockGpu.js";
import {
  DerivedUniformsScene, registerRoDerivations, deregisterRoDerivations,
  type RoDerivedRequest,
} from "../packages/rendering/src/runtime/derivedUniforms/sceneIntegration.js";

const dev = (): GPUDevice => new MockGPU().device;
const tr = (x: number, y = 0, z = 0): AVal<Trafo3d> => AVal.constant(Trafo3d.translation(new V3d(x, y, z)));

function chainReq(modelChain: AVal<Trafo3d>[]): RoDerivedRequest {
  return {
    rules: new Map(),
    modelChain,
    trafoAvals: new Map(),
    hostUniformOffset: () => undefined,
    outputOffset: () => undefined,
    drawHeaderBaseByte: 0,
    chunkIdx: 0,
  };
}

describe("transform-propagation suffix trie", () => {
  it("siblings sharing a root share the ancestor trie node (prefix sharing)", () => {
    const scene = new DerivedUniformsScene(dev());
    const root = cval(Trafo3d.translation(new V3d(5, 0, 0)));
    // chains are [leaf, …, root]; siblings share the `root` suffix.
    const r1 = registerRoDerivations(scene, {}, chainReq([tr(0, 1, 0), root]));
    const r2 = registerRoDerivations(scene, {}, chainReq([tr(0, 0, 3), root]));
    // root node shared + 2 distinct leaf nodes = 3 trie nodes (NOT 4).
    expect(scene.trafoTree.nodeCount).toBe(3);
    expect(r1.modelLeaf).toBeDefined();
    expect(r1.modelLeaf!.parent).toBe(r2.modelLeaf!.parent); // same shared root node
    expect(r1.modelLeaf!.parent!.level).toBe(0);             // root is level 0
    expect(r1.modelLeaf!.level).toBe(1);                     // leaf is level 1
  });

  it("a deeper shared ancestor path shares all its trie nodes", () => {
    const scene = new DerivedUniformsScene(dev());
    const root = cval(Trafo3d.translation(new V3d(1, 0, 0)));
    const mid = cval(Trafo3d.translation(new V3d(0, 2, 0)));
    registerRoDerivations(scene, {}, chainReq([tr(7), mid, root]));
    registerRoDerivations(scene, {}, chainReq([tr(8), mid, root]));
    // root + mid shared, 2 leaves = 4 nodes (not 6).
    expect(scene.trafoTree.nodeCount).toBe(4);
  });

  it("add/remove: trie nodes are refcounted and freed", () => {
    const scene = new DerivedUniformsScene(dev());
    const root = cval(Trafo3d.translation(new V3d(1, 1, 1)));
    const r1 = registerRoDerivations(scene, {}, chainReq([tr(1), root]));
    const r2 = registerRoDerivations(scene, {}, chainReq([tr(2), root]));
    const r3 = registerRoDerivations(scene, {}, chainReq([tr(3), root]));
    expect(scene.trafoTree.nodeCount).toBe(4); // root + 3 leaves

    deregisterRoDerivations(scene, r2);          // leaf2 freed; root kept (r1,r3)
    expect(scene.trafoTree.nodeCount).toBe(3);
    deregisterRoDerivations(scene, r1);
    deregisterRoDerivations(scene, r3);          // last refs → root + leaves freed
    expect(scene.trafoTree.nodeCount).toBe(0);

    // Freed constituent slots are reused on the next register (no growth).
    const before = scene.constituents.slotCount;
    registerRoDerivations(scene, {}, chainReq([tr(9), root]));
    expect(scene.constituents.slotCount).toBe(before);
  });

  it("distinct roots do not share", () => {
    const scene = new DerivedUniformsScene(dev());
    registerRoDerivations(scene, {}, chainReq([tr(1), cval(Trafo3d.translation(new V3d(1, 0, 0)))]));
    registerRoDerivations(scene, {}, chainReq([tr(1), cval(Trafo3d.translation(new V3d(2, 0, 0)))]));
    // Different root avals → different root nodes → 4 nodes, no sharing.
    expect(scene.trafoTree.nodeCount).toBe(4);
  });
});
