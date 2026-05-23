// Legacy CPU evaluation of a derived-MODE rule (derivedModes/cpuEval).
// Hand-builds the rule IR the build-time `rule(...)` marker would emit:
//
//   const det = determinant(mat3(u.ModelTrafo));
//   return det < 0 ? flipCull(declared) : declared;
//
// Exercises matrix coercion, ConvertMatrix (mat4→mat3), Determinant, the
// `det < 0` comparison + Conditional, a `declared` leaf, AND a user-function
// Call (flipCull, a separate Function in the module — not inlined).

import { describe, expect, it } from "vitest";
import { AdaptiveToken } from "@aardworx/wombat.adaptive";
import { M44d } from "@aardworx/wombat.base";
import { evaluateModeRule } from "../packages/rendering/src/runtime/derivedModes/cpuEval.js";
import type { DerivedModeRule } from "../packages/rendering/src/runtime/derivedModes/rule.js";

const Tf32 = { kind: "Float", width: 32 } as const;
const Ti32 = { kind: "Int", signed: true, width: 32 } as const;
const Tbool = { kind: "Bool" } as const;
const Tmat4 = { kind: "Matrix", element: Tf32, rows: 4, cols: 4 } as const;
const Tmat3 = { kind: "Matrix", element: Tf32, rows: 3, cols: 3 } as const;

const ci = (value: number): any => ({ kind: "Const", value: { kind: "Int", signed: true, value }, type: Ti32 });
const u = (name: string, type: unknown): any => ({ kind: "ReadInput", scope: "Uniform", name, type });
const cVar: any = { kind: "Var", var: { name: "c", type: Ti32, mutable: false }, type: Ti32 };
const detVar: any = { kind: "Var", var: { name: "det", type: Tf32, mutable: false }, type: Tf32 };

// flipCull(c) = c==2 ? 1 : (c==1 ? 2 : c)  — swaps front/back, leaves none.
const flipSig = { name: "flipCull", returnType: Ti32, parameters: [{ name: "c", type: Ti32, modifier: "in" }] };
const flipFn: any = {
  kind: "Function",
  signature: flipSig,
  body: { kind: "ReturnValue", value: {
    kind: "Conditional", type: Ti32,
    cond: { kind: "Eq", lhs: cVar, rhs: ci(2), type: Tbool },
    ifTrue: ci(1),
    ifFalse: { kind: "Conditional", type: Ti32, cond: { kind: "Eq", lhs: cVar, rhs: ci(1), type: Tbool }, ifTrue: ci(2), ifFalse: cVar },
  } },
};

const entryBody: any = { kind: "Sequential", body: [
  { kind: "Declare", var: { name: "det", type: Tf32, mutable: false }, init: { kind: "Expr", value:
    { kind: "Determinant", type: Tf32, value: { kind: "ConvertMatrix", type: Tmat3, value: u("ModelTrafo", Tmat4) } } } },
  { kind: "ReturnValue", value: {
    kind: "Conditional", type: Ti32,
    cond: { kind: "Lt", lhs: detVar, rhs: { kind: "Const", value: { kind: "Float", value: 0 }, type: Tf32 }, type: Tbool },
    ifTrue: { kind: "Call", fn: { id: "flipCull", signature: flipSig, pure: true }, args: [u("declared", Ti32)], type: Ti32 },
    ifFalse: u("declared", Ti32),
  } },
] };

const entry: any = { kind: "Entry", entry: { name: "rule", stage: "compute", inputs: [], outputs: [], arguments: [], returnType: Ti32, body: entryBody, decorations: [] } };

function cullRule(declared: "none" | "front" | "back"): DerivedModeRule<"cull"> {
  return {
    __derivedModeRule: true,
    axis: "cull",
    expr: { id: "r", template: { types: [], values: [flipFn, entry] } as any, holes: {} as any, avalHoles: {} as any, dumpIR: () => "" } as any,
    declared,
  };
}

const identity = M44d.fromArray([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
const mirrorX   = M44d.fromArray([-1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]); // det(upper3) = -1
const reader = (m: M44d) => (name: string): unknown => (name === "ModelTrafo" ? m : undefined);

describe("derived-mode CPU evaluator", () => {
  it("non-mirrored model → declared cull unchanged", () => {
    const r = cullRule("back");
    expect(evaluateModeRule(r, reader(identity), AdaptiveToken.top)).toBe("back");
  });

  it("mirrored model (det<0) → flipCull(declared): back→front", () => {
    const r = cullRule("back");
    expect(evaluateModeRule(r, reader(mirrorX), AdaptiveToken.top)).toBe("front");
  });

  it("mirrored model with declared front → back", () => {
    const r = cullRule("front");
    expect(evaluateModeRule(r, reader(mirrorX), AdaptiveToken.top)).toBe("back");
  });

  it("mirrored model with declared none → none (flipCull leaves 0)", () => {
    const r = cullRule("none");
    expect(evaluateModeRule(r, reader(mirrorX), AdaptiveToken.top)).toBe("none");
  });
});
