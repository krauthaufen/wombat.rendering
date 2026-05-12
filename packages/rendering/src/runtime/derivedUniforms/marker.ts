// §7 v2 — `derivedUniform((u) => …)` author-facing marker.
//
// A uniform binding is either a *value* (an `aval` / constant) or a *rule*
// (`derivedUniform((u) => u.ViewTrafo.mul(u.ModelTrafo))`). A rule reads other
// uniforms — `u.<Name>` is a leaf, and is resolved per RenderObject at registration
// time (a `Trafo3d` aval ⇒ a df32 constituent slot; anything else ⇒ a host uniform).
//
// This builds the rule's IR by tracing: `u.<Name>` is a mat4 leaf (the trafo case —
// the vast majority of derived uniforms), and the `DerivedExpr` methods compose them
// (the result type follows from the ops, so `.upperLeft3x3()` yields a mat3 rule, etc.).
// For non-mat4-leaf rules use `ruleFromIR` with hand-built IR, or — once it lands — the
// build-time `derivedUniform(...)` vite marker, which reads leaf types from the program.

import type { Expr, Type } from "@aardworx/wombat.shader/ir";
import { ruleFromIR, uniformRef, type DerivedRule } from "./rule.js";

const Tf32: Type = { kind: "Float", width: 32 };
const mat = (n: 3 | 4): Type => ({ kind: "Matrix", element: Tf32, rows: n, cols: n });
const vec = (n: 2 | 3 | 4): Type => ({ kind: "Vector", element: Tf32, dim: n });

function transposed(t: Type): Type {
  return t.kind === "Matrix" ? { kind: "Matrix", element: t.element, rows: t.cols, cols: t.rows } : t;
}
function constVec4(x: number, y: number, z: number, w: number): Expr {
  const c = (v: number): Expr => ({ kind: "Const", type: Tf32, value: { kind: "Float", value: v } });
  return { kind: "NewVector", components: [c(x), c(y), c(z), c(w)], type: vec(4) };
}

/** A traced expression in a `derivedUniform` builder. */
export class DerivedExpr {
  constructor(readonly ir: Expr) {}
  get type(): Type { return this.ir.type; }

  /** Matrix product `this · other` (also matrix·vector / vector·matrix by operand types). */
  mul(other: DerivedExpr): DerivedExpr {
    const a = this.ir.type, b = other.ir.type;
    if (a.kind === "Matrix" && b.kind === "Matrix") {
      return new DerivedExpr({ kind: "MulMatMat", lhs: this.ir, rhs: other.ir, type: { kind: "Matrix", element: a.element, rows: a.rows, cols: b.cols } });
    }
    if (a.kind === "Matrix" && b.kind === "Vector") {
      return new DerivedExpr({ kind: "MulMatVec", lhs: this.ir, rhs: other.ir, type: { kind: "Vector", element: b.element, dim: a.rows } });
    }
    if (a.kind === "Vector" && b.kind === "Matrix") {
      return new DerivedExpr({ kind: "MulVecMat", lhs: this.ir, rhs: other.ir, type: { kind: "Vector", element: a.element, dim: b.cols } });
    }
    return new DerivedExpr({ kind: "Mul", lhs: this.ir, rhs: other.ir, type: a.kind === "Float" ? b : a });
  }
  add(other: DerivedExpr): DerivedExpr { return new DerivedExpr({ kind: "Add", lhs: this.ir, rhs: other.ir, type: this.ir.type }); }
  sub(other: DerivedExpr): DerivedExpr { return new DerivedExpr({ kind: "Sub", lhs: this.ir, rhs: other.ir, type: this.ir.type }); }
  neg(): DerivedExpr { return new DerivedExpr({ kind: "Neg", value: this.ir, type: this.ir.type }); }

  /** Inverse (of a constituent trafo this reads its stored backward half; otherwise unsupported in codegen). */
  inverse(): DerivedExpr { return new DerivedExpr({ kind: "Inverse", value: this.ir, type: this.ir.type }); }
  transpose(): DerivedExpr { return new DerivedExpr({ kind: "Transpose", value: this.ir, type: transposed(this.ir.type) }); }
  /** Upper-left 3×3 (mat → mat3). */
  upperLeft3x3(): DerivedExpr { return new DerivedExpr({ kind: "ConvertMatrix", value: this.ir, type: mat(3) }); }
  /** This mat4 applied to the origin `(0,0,0,1)`, returning the resulting position (xyz). */
  transformOrigin(): DerivedExpr {
    const v4: Expr = { kind: "MulMatVec", lhs: this.ir, rhs: constVec4(0, 0, 0, 1), type: vec(4) };
    return new DerivedExpr({ kind: "VecSwizzle", value: v4, comps: ["x", "y", "z"], type: vec(3) });
  }
  /** This mat4 applied to the direction `(x,y,z,0)` (no translation), returning the result (xyz). */
  transformDir(x: number, y: number, z: number): DerivedExpr {
    const v4: Expr = { kind: "MulMatVec", lhs: this.ir, rhs: constVec4(x, y, z, 0), type: vec(4) };
    return new DerivedExpr({ kind: "VecSwizzle", value: v4, comps: ["x", "y", "z"], type: vec(3) });
  }
  /** vecN swizzle, e.g. `.swizzle("xyz")`. */
  swizzle(comps: string): DerivedExpr {
    const c = comps.split("") as ("x" | "y" | "z" | "w")[];
    return new DerivedExpr({ kind: "VecSwizzle", value: this.ir, comps: c, type: c.length === 1 ? Tf32 : vec(c.length as 2 | 3 | 4) });
  }
}

/** The `u` passed to a `derivedUniform` builder. `u.<Name>` is a mat4 trafo leaf. */
export type DerivedScope = { readonly [name: string]: DerivedExpr };

function makeScope(): DerivedScope {
  return new Proxy({} as DerivedScope, {
    get(_t, key): DerivedExpr {
      if (typeof key !== "string") throw new Error("derivedUniform: leaf names must be strings");
      return new DerivedExpr(uniformRef(key, mat(4)));
    },
  });
}

/**
 * Define a derived uniform. `derivedUniform(u => u.ViewTrafo.mul(u.ModelTrafo))` makes a
 * rule that slots in wherever a uniform value goes — see this module's header.
 */
export function derivedUniform<T = unknown>(build: (u: DerivedScope) => DerivedExpr): DerivedRule<T> {
  const result = build(makeScope());
  if (!(result instanceof DerivedExpr)) {
    throw new Error("derivedUniform: the builder must return a derived expression, e.g. u.ViewTrafo.mul(u.ModelTrafo)");
  }
  return ruleFromIR<T>(result.ir, result.ir.type);
}
