import { describe, it, expect } from "vitest";
import type { Expr, Type } from "@aardworx/wombat.shader/ir";
import type { aval } from "@aardworx/wombat.adaptive";
import type { Trafo3d } from "@aardworx/wombat.base";
import { MockGPU } from "./_mockGpu.js";
import { ruleFromIR, uniformRef, type DerivedRule } from "../packages/rendering/src/runtime/derivedUniforms/rule.js";
import {
  DerivedUniformsScene, registerRoDerivations, deregisterRoDerivations,
  type RoDerivedRequest,
} from "../packages/rendering/src/runtime/derivedUniforms/sceneIntegration.js";
import { SlotTag, handleTag, handlePayload } from "../packages/rendering/src/runtime/derivedUniforms/records.js";

const Tf32: Type = { kind: "Float", width: 32 };
const Tmat4: Type = { kind: "Matrix", element: Tf32, rows: 4, cols: 4 };
const mul = (l: Expr, r: Expr): Expr => ({ kind: "MulMatMat", lhs: l, rhs: r, type: Tmat4 });
const inv = (v: Expr): Expr => ({ kind: "Inverse", value: v, type: Tmat4 });
const u = (n: string) => uniformRef(n, Tmat4);

const fakeDevice = (): GPUDevice => new MockGPU().device;
const fakeBuf = (): GPUBuffer => new MockGPU().device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
const fakeAval = (): aval<Trafo3d> => ({} as aval<Trafo3d>);

function req(overrides: Partial<RoDerivedRequest> & Pick<RoDerivedRequest, "rules">): RoDerivedRequest {
  return {
    trafoAvals: new Map(),
    hostUniformOffset: () => undefined,
    outputOffset: (n) => OUT.get(n),
    drawHeaderBaseByte: 1024,
    chunkIdx: 0,
    ...overrides,
  };
}
const OUT = new Map<string, number>([
  ["ModelTrafo", 0], ["ModelTrafoInv", 64], ["ViewProjTrafo", 128], ["ModelViewProjTrafo", 192],
]);

describe("derived-uniforms: sceneIntegration", () => {
  it("registers a 2-matmul rule: one record, two constituent input slots, output to drawHeader", () => {
    const scene = new DerivedUniformsScene(fakeDevice());
    const view = fakeAval(), model = fakeAval();
    const rules = new Map<string, DerivedRule>([["ViewProjTrafo", ruleFromIR(mul(u("View"), u("Model")))]]);
    const ro = {};
    registerRoDerivations(scene, ro, req({
      rules,
      trafoAvals: new Map([["View", view], ["Model", model]]),
    }));
    expect(scene.recordsFor(0).recordCount).toBe(1);
    const s = scene.recordsFor(0).strideWords;
    const rec = Array.from(scene.recordsFor(0).data.subarray(0, s));
    // [rule_id, out_handle, in0, in1, pad]
    expect(handleTag(rec[1]!)).toBe(SlotTag.HostHeap);
    expect(handlePayload(rec[1]!)).toBe(1024 + 128); // drawHeaderBaseByte + outputOffset("ViewProjTrafo")
    expect(handleTag(rec[2]!)).toBe(SlotTag.Constituent);
    expect(handleTag(rec[3]!)).toBe(SlotTag.Constituent);
    expect(rec[2]).not.toBe(rec[3]); // distinct constituent slots for View vs Model
    expect(scene.registry.size).toBe(1);
  });

  it("Inverse(trafo) leaf resolves to the trafo's backward (.inv) constituent slot", () => {
    const scene = new DerivedUniformsScene(fakeDevice());
    const model = fakeAval();
    const rules = new Map<string, DerivedRule>([["ModelTrafoInv", ruleFromIR(inv(u("Model")))]]);
    registerRoDerivations(scene, {}, req({ rules, trafoAvals: new Map([["Model", model]]) }));
    const s = scene.recordsFor(0).strideWords;
    const rec = Array.from(scene.recordsFor(0).data.subarray(0, s));
    expect(handleTag(rec[2]!)).toBe(SlotTag.Constituent);
    // ConstituentSlots allocates fwd then inv adjacently; inv === fwd + 1.
    const fwd = handlePayload(rec[2]!) - 1; // the inv slot we got back
    expect(handlePayload(rec[2]!)).toBe(fwd + 1);
  });

  it("flattens a chained rule (MVP from ViewProj·Model) before registering", () => {
    const scene = new DerivedUniformsScene(fakeDevice());
    const rules = new Map<string, DerivedRule>([
      ["ViewProjTrafo", ruleFromIR(mul(u("Proj"), u("View")))],
      ["ModelViewProjTrafo", ruleFromIR(mul(u("ViewProjTrafo"), u("Model")))],
    ]);
    registerRoDerivations(scene, {}, req({
      rules,
      trafoAvals: new Map([["Proj", fakeAval()], ["View", fakeAval()], ["Model", fakeAval()]]),
    }));
    expect(scene.recordsFor(0).recordCount).toBe(2);
    expect(scene.registry.size).toBe(2);
    // The MVP record must reference a 3-input flattened rule (Proj, View, Model), all constituents.
    const s = scene.recordsFor(0).strideWords;
    let mvp: number[] | undefined;
    for (let i = 0; i < 2; i++) {
      const r = Array.from(scene.recordsFor(0).data.subarray(i * s, i * s + s));
      if (scene.registry.get(r[0]!)!.inputs.length === 3) mvp = r;
    }
    expect(mvp).toBeDefined();
    expect(handleTag(mvp![2]!)).toBe(SlotTag.Constituent);
    expect(handleTag(mvp![3]!)).toBe(SlotTag.Constituent);
    expect(handleTag(mvp![4]!)).toBe(SlotTag.Constituent);
  });

  it("resolves a non-trafo leaf to a host drawHeader byte offset", () => {
    const scene = new DerivedUniformsScene(fakeDevice());
    const rules = new Map<string, DerivedRule>([["ModelViewProjTrafo", ruleFromIR(mul(u("SomeHostMat"), u("Model")))]]);
    registerRoDerivations(scene, {}, req({
      rules,
      trafoAvals: new Map([["Model", fakeAval()]]),
      hostUniformOffset: (n) => (n === "SomeHostMat" ? 32 : undefined),
    }));
    const s = scene.recordsFor(0).strideWords;
    const rec = Array.from(scene.recordsFor(0).data.subarray(0, s));
    // in0 = SomeHostMat (host heap), in1 = ModelTrafo (constituent)
    expect(handleTag(rec[2]!)).toBe(SlotTag.HostHeap);
    expect(handlePayload(rec[2]!)).toBe(1024 + 32);
    expect(handleTag(rec[3]!)).toBe(SlotTag.Constituent);
  });

  it("deregister removes all of an RO's records and releases its constituents", () => {
    const scene = new DerivedUniformsScene(fakeDevice());
    const ro = {};
    const reg = registerRoDerivations(scene, ro, req({
      rules: new Map([
        ["ModelTrafo", ruleFromIR(u("Model"))],
        ["ViewProjTrafo", ruleFromIR(mul(u("Proj"), u("View")))],
      ]),
      trafoAvals: new Map([["Model", fakeAval()], ["Proj", fakeAval()], ["View", fakeAval()]]),
    }));
    expect(scene.recordsFor(0).recordCount).toBe(2);
    deregisterRoDerivations(scene, reg);
    expect(scene.recordsFor(0).recordCount).toBe(0);
  });

  it("throws when a rule references an input the RO can't supply", () => {
    const scene = new DerivedUniformsScene(fakeDevice());
    expect(() =>
      registerRoDerivations(scene, {}, req({
        rules: new Map([["ModelTrafo", ruleFromIR(u("Mystery"))]]),
      })),
    ).toThrow(/cannot be resolved|not available/);
  });
});
