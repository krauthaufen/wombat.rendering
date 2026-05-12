import { describe, it, expect } from "vitest";
import type { Expr, Type } from "@aardworx/wombat.shader/ir";
import { ruleFromIR, uniformRef, hashIR } from "../packages/rendering/src/runtime/derivedUniforms/rule.js";
import { flatten, inputsOf } from "../packages/rendering/src/runtime/derivedUniforms/flatten.js";

const Tf32: Type = { kind: "Float", width: 32 };
const Tmat4: Type = { kind: "Matrix", element: Tf32, rows: 4, cols: 4 };
const mul = (lhs: Expr, rhs: Expr): Expr => ({ kind: "MulMatMat", lhs, rhs, type: Tmat4 });
const inv = (v: Expr): Expr => ({ kind: "Inverse", value: v, type: Tmat4 });

describe("derived-uniforms: rule.ts", () => {
  it("ruleFromIR derives outputType from ir and hashes it", () => {
    const r = ruleFromIR(mul(uniformRef("View", Tmat4), uniformRef("Model", Tmat4)));
    expect(r.outputType).toEqual(Tmat4);
    expect(r.hash).toBe(hashIR(r.ir));
  });

  it("ruleFromIR rejects a mismatched declared outputType", () => {
    expect(() => ruleFromIR(uniformRef("Model", Tmat4), Tf32)).toThrow();
  });
});

describe("derived-uniforms: flatten.ts", () => {
  it("inputsOf collects leaves in first-appearance order with the inverse flag", () => {
    const ir = mul(uniformRef("View", Tmat4), inv(uniformRef("Model", Tmat4)));
    expect(inputsOf(ir)).toEqual([
      { name: "View", type: Tmat4, inverse: false },
      { name: "Model", type: Tmat4, inverse: true },
    ]);
  });

  it("inputsOf dedups by (name, inverse) but keeps both halves distinct", () => {
    const ir = mul(uniformRef("M", Tmat4), mul(inv(uniformRef("M", Tmat4)), uniformRef("M", Tmat4)));
    expect(inputsOf(ir)).toEqual([
      { name: "M", type: Tmat4, inverse: false },
      { name: "M", type: Tmat4, inverse: true },
    ]);
  });

  it("inputsOf rejects a name reused at a different type", () => {
    const ir = mul(uniformRef("X", Tmat4), uniformRef("X", Tf32) as Expr);
    expect(() => inputsOf(ir)).toThrow(/conflicting types/);
  });

  it("flatten substitutes a derived producer", () => {
    const viewProj = ruleFromIR(mul(uniformRef("Proj", Tmat4), uniformRef("View", Tmat4)));
    const mvp = mul(uniformRef("ViewProj", Tmat4), uniformRef("Model", Tmat4));
    const flat = flatten("MVP", mvp, new Map([["ViewProj", viewProj]]));
    expect(inputsOf(flat).map((i) => i.name)).toEqual(["Proj", "View", "Model"]);
    expect(flat).toEqual(
      mul(mul(uniformRef("Proj", Tmat4), uniformRef("View", Tmat4)), uniformRef("Model", Tmat4)),
    );
  });

  it("flatten recurses through chains", () => {
    const vp = ruleFromIR(mul(uniformRef("Proj", Tmat4), uniformRef("View", Tmat4)));
    const mvp = ruleFromIR(mul(uniformRef("ViewProj", Tmat4), uniformRef("Model", Tmat4)));
    const x = mul(uniformRef("MVP", Tmat4), uniformRef("Extra", Tmat4));
    const flat = flatten("X", x, new Map([["ViewProj", vp], ["MVP", mvp]]));
    expect(inputsOf(flat).map((i) => i.name)).toEqual(["Proj", "View", "Model", "Extra"]);
  });

  it("flatten leaves non-derived names untouched", () => {
    const ir = mul(uniformRef("View", Tmat4), uniformRef("Model", Tmat4));
    expect(flatten("Whatever", ir, new Map())).toEqual(ir);
  });

  it("flatten detects a 2-cycle", () => {
    const a = ruleFromIR(uniformRef("B", Tmat4));
    const b = ruleFromIR(uniformRef("A", Tmat4));
    expect(() => flatten("A", a.ir, new Map([["A", a], ["B", b]]))).toThrow(/cycle/);
  });

  it("flatten detects a self-cycle", () => {
    const a = ruleFromIR(uniformRef("A", Tmat4));
    expect(() => flatten("A", a.ir, new Map([["A", a]]))).toThrow(/cycle/);
  });
});
