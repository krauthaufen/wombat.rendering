// GPU transform propagation — CPU-side structural tests for the CHAIN record
// path (registration, the shared-constituent fan-out property, add/remove).
// The df32 math itself is verified on a real GPU in
// tests-browser/transform-chain-real.test.ts.

import { describe, it, expect } from "vitest";
import { AVal, cval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d } from "@aardworx/wombat.base";
import { MockGPU } from "./_mockGpu.js";
import {
  DerivedUniformsScene, registerRoDerivations, deregisterRoDerivations,
  type RoDerivedRequest,
} from "../packages/rendering/src/runtime/derivedUniforms/sceneIntegration.js";
import { handleTag, handlePayload, SlotTag } from "../packages/rendering/src/runtime/derivedUniforms/records.js";
import { CHAIN_RULE_ID } from "../packages/rendering/src/runtime/derivedUniforms/codegen.js";

const dev = (): GPUDevice => new MockGPU().device;

function chainReq(chain: ReturnType<typeof cval<Trafo3d>>[] | AVal<Trafo3d>[], outOff = 0): RoDerivedRequest {
  return {
    rules: new Map(),
    chains: new Map([["ModelTrafo", chain as unknown as AVal<Trafo3d>[]]]),
    trafoAvals: new Map(),
    hostUniformOffset: () => undefined,
    outputOffset: () => outOff,
    drawHeaderBaseByte: 0,
    chunkIdx: 0,
  };
}

describe("transform-propagation CHAIN records", () => {
  it("lays out a chain record as [CHAIN_RULE_ID, outHandle, count, links…]", () => {
    const scene = new DerivedUniformsScene(dev());
    const root = cval(Trafo3d.translation(new V3d(1, 0, 0)));
    const a = AVal.constant(Trafo3d.translation(new V3d(0, 2, 0)));
    const ro = {};
    registerRoDerivations(scene, ro, chainReq([root, a], 64));

    const rec = scene.recordsFor(0);
    expect(rec.recordCount).toBe(1);
    const d = rec.data;
    expect(d[0]).toBe(CHAIN_RULE_ID >>> 0);            // rule id
    expect(handleTag(d[1]!)).toBe(SlotTag.HostHeap);   // output → drawHeader
    expect(handlePayload(d[1]!)).toBe(64);             // at outOff
    expect(d[2]).toBe(2);                               // count
    expect(handleTag(d[3]!)).toBe(SlotTag.Constituent); // link 0 (root)
    expect(handleTag(d[4]!)).toBe(SlotTag.Constituent); // link 1 (a)
  });

  it("shares one constituent slot for a root trafo across N ROs (no fan-out)", () => {
    const scene = new DerivedUniformsScene(dev());
    const root = cval(Trafo3d.translation(new V3d(5, 0, 0)));   // the shared root
    const a = AVal.constant(Trafo3d.scale?.(new V3d(2, 2, 2)) ?? Trafo3d.translation(new V3d(0, 1, 0)));
    const b = AVal.constant(Trafo3d.translation(new V3d(0, 0, 3)));
    registerRoDerivations(scene, {}, chainReq([root, a]));
    registerRoDerivations(scene, {}, chainReq([root, b]));

    const d = scene.recordsFor(0).data;
    const stride = scene.recordsFor(0).strideWords;
    // link 0 of both records is `root` → identical constituent handle.
    const root0 = d[3]!;
    const root1 = d[stride + 3]!;
    expect(root0).toBe(root1);
    // link 1 differs (a vs b).
    expect(d[4]).not.toBe(d[stride + 4]);
    // 3 distinct avals × (fwd+inv) = 6 constituent slots — NOT 4 (no per-RO root copy).
    expect(scene.constituents.slotCount).toBe(6);
  });

  it("add/remove ROs (structural): records appear and swap-remove cleanly", () => {
    const scene = new DerivedUniformsScene(dev());
    const root = cval(Trafo3d.translation(new V3d(1, 1, 1)));
    const ro1 = {}, ro2 = {}, ro3 = {};
    const r1 = registerRoDerivations(scene, ro1, chainReq([root, AVal.constant(Trafo3d.translation(new V3d(1, 0, 0)))]));
    const r2 = registerRoDerivations(scene, ro2, chainReq([root, AVal.constant(Trafo3d.translation(new V3d(2, 0, 0)))]));
    const r3 = registerRoDerivations(scene, ro3, chainReq([root, AVal.constant(Trafo3d.translation(new V3d(3, 0, 0)))]));
    expect(scene.recordsFor(0).recordCount).toBe(3);

    // Remove the middle one — swap-remove must keep the other two intact.
    deregisterRoDerivations(scene, r2);
    expect(scene.recordsFor(0).recordCount).toBe(2);

    deregisterRoDerivations(scene, r1);
    deregisterRoDerivations(scene, r3);
    expect(scene.recordsFor(0).recordCount).toBe(0);
    // All constituents released back to the pool (root refcount hit 0).
    // slotCount is a high-water mark; the pool frees indices for reuse, so a
    // fresh acquire reuses them — assert reuse rather than shrink.
    const before = scene.constituents.slotCount;
    registerRoDerivations(scene, {}, chainReq([root]));
    expect(scene.constituents.slotCount).toBe(before); // reused freed slots, no growth
  });
});
