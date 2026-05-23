// CPU interpreter for derived-uniform rules (cpuEval) — the legacy per-RO
// path's answer to the heap §7 GPU compute pass. These pin the f64
// interpretation of the common rule shapes against direct wombat.base math.

import { describe, expect, it } from "vitest";
import { V3d, M44d, Trafo3d } from "@aardworx/wombat.base";
import { derivedUniform } from "../packages/rendering/src/runtime/derivedUniforms/marker.js";
import { interpretExpr } from "../packages/rendering/src/runtime/derivedUniforms/cpuEval.js";

const model = Trafo3d.translation(new V3d(1, 2, 3)).mul(Trafo3d.rotation(new V3d(0, 0, 1), 0.7));
const view  = Trafo3d.translation(new V3d(-4, 0, 2)).mul(Trafo3d.rotation(new V3d(1, 0, 0), -0.4));
const proj  = Trafo3d.translation(new V3d(0, 0, -5)).mul(Trafo3d.rotation(new V3d(0, 1, 0), 0.3));

const leaves: Record<string, unknown> = { ModelTrafo: model, ViewTrafo: view, ProjTrafo: proj };
const readLeaf = (name: string): unknown => leaves[name];

const dataOf = (v: unknown): Float64Array => (v as { _data: Float64Array })._data;
function expectMatClose(got: unknown, want: M44d): void {
  const g = dataOf(got), w = dataOf(want);
  expect(g.length).toBe(16);
  for (let i = 0; i < 16; i++) expect(g[i]!).toBeCloseTo(w[i]!, 9);
}
function expectVecClose(got: unknown, want: V3d): void {
  const g = dataOf(got), w = dataOf(want);
  expect(g.length).toBe(3);
  for (let i = 0; i < 3; i++) expect(g[i]!).toBeCloseTo(w[i]!, 9);
}

describe("derived-uniform CPU interpreter", () => {
  it("ModelViewTrafo = View · Model", () => {
    const rule = derivedUniform((u) => u.ViewTrafo.mul(u.ModelTrafo));
    expectMatClose(interpretExpr(rule.ir, readLeaf), view.forward.mul(model.forward));
  });

  it("inverse() of a leaf trafo (f64, no df32 backward trick)", () => {
    const rule = derivedUniform((u) => u.ModelTrafo.inverse());
    expectMatClose(interpretExpr(rule.ir, readLeaf), model.forward.inverse());
  });

  it("ModelViewProjTrafo = Proj · View · Model", () => {
    const rule = derivedUniform((u) => u.ProjTrafo.mul(u.ViewTrafo).mul(u.ModelTrafo));
    expectMatClose(interpretExpr(rule.ir, readLeaf), proj.forward.mul(view.forward).mul(model.forward));
  });

  it("custom rule: WorldUpInModel = normalize((Model⁻¹) · (0,1,0,0)).xyz", () => {
    const rule = derivedUniform((u) => u.ModelTrafo.inverse().transformDir(0, 1, 0).normalize());
    const want = model.forward.inverse().transformDir(new V3d(0, 1, 0)).normalize();
    expectVecClose(interpretExpr(rule.ir, readLeaf), want);
  });

  it("throws clearly on a missing leaf", () => {
    const rule = derivedUniform((u) => u.ViewTrafo.mul(u.ModelTrafo));
    expect(() => interpretExpr(rule.ir, () => undefined)).toThrow(/missing uniform leaf/);
  });
});
