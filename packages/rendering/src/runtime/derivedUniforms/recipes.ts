// §7 v2 — the standard trafo recipes, as DerivedRule constants.
//
// These replace the hardcoded RecipeId table. Each rule reads BASE-trafo leaves
// (`Model` / `View` / `Proj` — the raw `aval<Trafo3d>`s the RO binds as
// `ModelTrafo` / `ViewTrafo` / `ProjTrafo` uniforms) and produces the derived
// uniform the shader reads (`ModelViewProjTrafo`, `NormalMatrix`, …). A leaf
// wrapped in `Inverse(...)` selects that trafo's stored backward half (free).
//
// `Inverse`-of-a-product (e.g. `ModelViewProjTrafoInv`) is written as the product
// of the inverted factors in reverse order — `(P·V·M)⁻¹ = M⁻¹·V⁻¹·P⁻¹` — so it
// stays a constituent matmul chain the codegen recognizes.
//
// `NormalMatrix` is `(Model⁻¹)ᵀ` truncated to its upper-3×3 — declared as a mat3
// output; the codegen's normal-matrix arm does the transpose + truncation.

import type { Expr, Type } from "@aardworx/wombat.shader/ir";
import { ruleFromIR, uniformRef, type DerivedRule } from "./rule.js";

const Tf32: Type = { kind: "Float", width: 32 };
const Tmat4: Type = { kind: "Matrix", element: Tf32, rows: 4, cols: 4 };
const Tmat3: Type = { kind: "Matrix", element: Tf32, rows: 3, cols: 3 };

const mat4 = (name: string): Expr => uniformRef(name, Tmat4);
const inv = (v: Expr): Expr => ({ kind: "Inverse", value: v, type: Tmat4 });
const mul = (lhs: Expr, rhs: Expr): Expr => ({ kind: "MulMatMat", lhs, rhs, type: Tmat4 });
/** Left-fold `A·B·C·…` (multiplication order). */
const chain = (...fs: Expr[]): Expr => fs.reduce((acc, f) => mul(acc, f));
/** mat4 → mat3 truncation (upper-left 3×3). */
const upper3 = (v: Expr): Expr => ({ kind: "ConvertMatrix", value: v, type: Tmat3 });
const transpose = (v: Expr): Expr => ({ kind: "Transpose", value: v, type: Tmat4 });

const Model = mat4("Model");
const View = mat4("View");
const Proj = mat4("Proj");

/** Standard derived rules, keyed by the drawHeader uniform name each produces. */
export const STANDARD_DERIVED_RULES: ReadonlyMap<string, DerivedRule> = new Map<string, DerivedRule>([
  // Trafo passthroughs (collapse the df32 fwd/bwd half to an f32 mat4).
  ["ModelTrafo", ruleFromIR(Model)],
  ["ModelTrafoInv", ruleFromIR(inv(Model))],
  ["ViewTrafo", ruleFromIR(View)],
  ["ViewTrafoInv", ruleFromIR(inv(View))],
  ["ProjTrafo", ruleFromIR(Proj)],
  ["ProjTrafoInv", ruleFromIR(inv(Proj))],

  // Composites. `mul(a,b)` ⇒ matrix product `a · b`.
  ["ModelViewTrafo", ruleFromIR(chain(View, Model))],
  ["ModelViewTrafoInv", ruleFromIR(chain(inv(Model), inv(View)))],
  ["ModelViewProjTrafo", ruleFromIR(chain(Proj, View, Model))],
  ["ModelViewProjTrafoInv", ruleFromIR(chain(inv(Model), inv(View), inv(Proj)))],
  ["ViewProjTrafo", ruleFromIR(chain(Proj, View))],
  ["ViewProjTrafoInv", ruleFromIR(chain(inv(View), inv(Proj)))],

  // NormalMatrix = (Model⁻¹)ᵀ, upper-3×3.
  ["NormalMatrix", ruleFromIR(upper3(transpose(inv(Model))), Tmat3)],
]);

/** Leaf name → the RO uniform (`spec.inputs[…]`) that supplies its `aval<Trafo3d>`. */
export const STANDARD_TRAFO_LEAVES: ReadonlyMap<string, string> = new Map([
  ["Model", "ModelTrafo"],
  ["View", "ViewTrafo"],
  ["Proj", "ProjTrafo"],
]);

export function isStandardDerivedName(name: string): boolean {
  return STANDARD_DERIVED_RULES.has(name);
}
