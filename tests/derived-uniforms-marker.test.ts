import { describe, it, expect } from "vitest";
import type { Type } from "@aardworx/wombat.shader/ir";
import { derivedUniform } from "../packages/rendering/src/runtime/derivedUniforms/marker.js";
import { isDerivedRule } from "../packages/rendering/src/runtime/derivedUniforms/rule.js";
import { inputsOf } from "../packages/rendering/src/runtime/derivedUniforms/flatten.js";

const Tf32: Type = { kind: "Float", width: 32 };
const Tmat4: Type = { kind: "Matrix", element: Tf32, rows: 4, cols: 4 };
const Tmat3: Type = { kind: "Matrix", element: Tf32, rows: 3, cols: 3 };
const Tvec3: Type = { kind: "Vector", element: Tf32, dim: 3 };
const leaf = (i: ReturnType<typeof inputsOf>[number]) => `${i.inverse ? "i:" : "v:"}${i.name}`;

describe("derivedUniform marker", () => {
  it("builds a mat4 composition rule and brands it", () => {
    const r = derivedUniform((u) => u.ViewTrafo.mul(u.ModelTrafo));
    expect(isDerivedRule(r)).toBe(true);
    expect(r.outputType).toEqual(Tmat4);
    expect(r.ir.kind).toBe("MulMatMat");
    expect(inputsOf(r.ir).map(leaf)).toEqual(["v:ViewTrafo", "v:ModelTrafo"]);
  });

  it("u.<Name> defaults to a mat4 trafo leaf; chains compose", () => {
    const r = derivedUniform((u) => u.ProjTrafo.mul(u.ViewTrafo).mul(u.ModelTrafo));
    expect(r.outputType).toEqual(Tmat4);
    expect(inputsOf(r.ir).map((i) => i.name)).toEqual(["ProjTrafo", "ViewTrafo", "ModelTrafo"]);
  });

  it(".inverse() records the leaf with inverse:true", () => {
    const r = derivedUniform((u) => u.ModelTrafo.inverse());
    expect(inputsOf(r.ir)).toEqual([{ name: "ModelTrafo", type: Tmat4, inverse: true }]);
  });

  it("normal-matrix shape: inverse → transpose → upperLeft3x3 yields a mat3 rule with one inverse leaf", () => {
    const r = derivedUniform((u) => u.ModelTrafo.inverse().transpose().upperLeft3x3());
    expect(r.outputType).toEqual(Tmat3);
    expect(inputsOf(r.ir).map(leaf)).toEqual(["i:ModelTrafo"]);
  });

  it("transformOrigin yields a vec3 rule (translation of the matrix)", () => {
    const r = derivedUniform((u) => u.ViewTrafo.inverse().transformOrigin());
    expect(r.outputType).toEqual(Tvec3);
    expect(inputsOf(r.ir).map(leaf)).toEqual(["i:ViewTrafo"]);
  });

  it("a derivedUniform-built rule equals the corresponding standard recipe", async () => {
    const { STANDARD_DERIVED_RULES } = await import("../packages/rendering/src/runtime/derivedUniforms/recipes.js");
    // The standard `ModelViewProjTrafo` recipe is Proj·View·Model over the base trafo leaves
    // (named Model/View/Proj). A marker rule over the same leaf names produces the same hash.
    const built = derivedUniform((u) => u.Proj.mul(u.View).mul(u.Model));
    expect(built.hash).toBe(STANDARD_DERIVED_RULES.get("ModelViewProjTrafo")!.hash);
  });

  it("rejects a builder that doesn't return a derived expression", () => {
    // @ts-expect-error intentionally wrong return
    expect(() => derivedUniform(() => 42)).toThrow();
  });
});

describe("derivedUniform: leafTypes hint (the build-time marker's output)", () => {
  it("u.<Name> mints a leaf of the hinted type", () => {
    const r = derivedUniform((u) => u.Tint.swizzle("xyz"), { Tint: "vec4" });
    expect(r.outputType).toEqual({ kind: "Vector", element: { kind: "Float", width: 32 }, dim: 3 });
    expect(inputsOf(r.ir)).toEqual([{ name: "Tint", type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 }, inverse: false }]);
  });
  it("un-hinted names still default to mat4", () => {
    const r = derivedUniform((u) => u.A.mul(u.B), { A: "mat4" });
    expect(r.outputType).toEqual({ kind: "Matrix", element: { kind: "Float", width: 32 }, rows: 4, cols: 4 });
  });
});
