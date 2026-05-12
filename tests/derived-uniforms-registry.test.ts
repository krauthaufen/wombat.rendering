import { describe, it, expect } from "vitest";
import type { Expr, Type } from "@aardworx/wombat.shader/ir";
import { ruleFromIR, uniformRef } from "../packages/rendering/src/runtime/derivedUniforms/rule.js";
import { DerivedUniformRegistry } from "../packages/rendering/src/runtime/derivedUniforms/registry.js";

const Tf32: Type = { kind: "Float", width: 32 };
const Tmat4: Type = { kind: "Matrix", element: Tf32, rows: 4, cols: 4 };
const mul = (lhs: Expr, rhs: Expr): Expr => ({ kind: "MulMatMat", lhs, rhs, type: Tmat4 });

describe("derived-uniforms: registry", () => {
  it("dedups by content hash and bumps version only on new rules", () => {
    const reg = new DerivedUniformRegistry();
    const a1 = ruleFromIR(mul(uniformRef("View", Tmat4), uniformRef("Model", Tmat4)));
    const a2 = ruleFromIR(mul(uniformRef("View", Tmat4), uniformRef("Model", Tmat4)));
    const b = ruleFromIR(mul(uniformRef("Proj", Tmat4), uniformRef("View", Tmat4)));

    expect(reg.version).toBe(0);
    const idA = reg.register(a1);
    expect(reg.version).toBe(1);
    expect(reg.register(a2)).toBe(idA); // same content ⇒ same id
    expect(reg.version).toBe(1); // no new rule
    const idB = reg.register(b);
    expect(idB).not.toBe(idA);
    expect(reg.version).toBe(2);
    expect(reg.size).toBe(2);
  });

  it("entry records inputs in first-appearance order", () => {
    const reg = new DerivedUniformRegistry();
    const id = reg.register(ruleFromIR(mul(uniformRef("View", Tmat4), uniformRef("Model", Tmat4))));
    expect(reg.get(id)!.inputs.map((i) => i.name)).toEqual(["View", "Model"]);
    expect(reg.get(id)!.outputType).toEqual(Tmat4);
  });

  it("ids are monotonic and never reused after release", () => {
    const reg = new DerivedUniformRegistry();
    const a = ruleFromIR(uniformRef("View", Tmat4));
    const idA = reg.register(a);
    reg.release(a.hash);
    const idB = reg.register(ruleFromIR(uniformRef("Proj", Tmat4)));
    expect(idB).toBeGreaterThan(idA);
    // re-registering the released rule keeps its original id (v0: never sweeps)
    expect(reg.register(a)).toBe(idA);
  });
});
