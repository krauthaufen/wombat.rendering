// GPU transform propagation — CPU-side structural tests for the modelChain
// path: each RO's ancestor chain becomes a per-RO Model constituent computed by
// the chain pass (two CHAIN records: fwd + reversed-inv), with the §7 `Model`
// leaf pointed at that output pair. The df32 math is verified on a real GPU in
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

describe("transform-propagation modelChain → Model constituent", () => {
  it("emits fwd + reversed-inv CHAIN records writing the Model constituent pair", () => {
    const scene = new DerivedUniformsScene(dev());
    const root = cval(Trafo3d.translation(new V3d(1, 0, 0)));
    const a = tr(0, 2, 0);
    const reg = registerRoDerivations(scene, {}, chainReq([root, a]));

    const cr = scene.chainRecords;
    expect(cr.recordCount).toBe(2);          // fwd + inv
    expect(reg.modelPair).toBeDefined();
    const d = cr.data, stride = cr.strideWords;

    // Record 0 = fwd: writes Model.fwd, links forward (root, a).
    expect(d[0]).toBe(CHAIN_RULE_ID >>> 0);
    expect(handleTag(d[1]!)).toBe(SlotTag.Constituent);
    expect(handlePayload(d[1]!)).toBe(reg.modelPair!.fwd);
    expect(d[2]).toBe(2);                     // count
    // Record 1 = inv: writes Model.inv, links reversed.
    expect(d[stride + 0]).toBe(CHAIN_RULE_ID >>> 0);
    expect(handlePayload(d[stride + 1]!)).toBe(reg.modelPair!.inv);
    expect(d[stride + 2]).toBe(2);
    // fwd record link0 == a-then... no: fwd is [root,a]; inv is reversed [a,root].
    // So fwd.link0 (root.fwd) != inv.link0 (a.inv).
    expect(d[3]).not.toBe(d[stride + 3]);
  });

  it("shares one root constituent across N ROs (no fan-out)", () => {
    const scene = new DerivedUniformsScene(dev());
    const root = cval(Trafo3d.translation(new V3d(5, 0, 0)));
    registerRoDerivations(scene, {}, chainReq([root, tr(0, 1, 0)]));
    registerRoDerivations(scene, {}, chainReq([root, tr(0, 0, 3)]));

    const d = scene.chainRecords.data, stride = scene.chainRecords.strideWords;
    expect(scene.chainRecords.recordCount).toBe(4); // 2 ROs × (fwd+inv)
    // Both ROs' fwd records start with the SAME root constituent handle.
    // Record order: ro1.fwd, ro1.inv, ro2.fwd, ro2.inv.
    const ro1FwdRoot = d[3]!;            // ro1 fwd link0 = root.fwd
    const ro2FwdRoot = d[2 * stride + 3]!; // ro2 fwd link0 = root.fwd
    expect(ro1FwdRoot).toBe(ro2FwdRoot);
  });

  it("add/remove ROs (structural): chain records + Model pairs reclaimed", () => {
    const scene = new DerivedUniformsScene(dev());
    const root = cval(Trafo3d.translation(new V3d(1, 1, 1)));
    const r1 = registerRoDerivations(scene, {}, chainReq([root, tr(1)]));
    const r2 = registerRoDerivations(scene, {}, chainReq([root, tr(2)]));
    const r3 = registerRoDerivations(scene, {}, chainReq([root, tr(3)]));
    expect(scene.chainRecords.recordCount).toBe(6);

    deregisterRoDerivations(scene, r2); // swap-remove middle
    expect(scene.chainRecords.recordCount).toBe(4);
    deregisterRoDerivations(scene, r1);
    deregisterRoDerivations(scene, r3);
    expect(scene.chainRecords.recordCount).toBe(0);

    // Freed slots (link + Model-pair) are reused on the next register — no growth.
    const before = scene.constituents.slotCount;
    registerRoDerivations(scene, {}, chainReq([root, tr(9)]));
    expect(scene.constituents.slotCount).toBe(before);
  });
});
