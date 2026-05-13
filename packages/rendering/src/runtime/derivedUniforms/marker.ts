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

const Tu32: Type = { kind: "Int", signed: false, width: 32 };
const Ti32: Type = { kind: "Int", signed: true, width: 32 };

/** Leaf type names accepted by `DerivedExpr.as(...)`. */
export type LeafTypeName = "f32" | "u32" | "i32" | "vec2" | "vec3" | "vec4" | "mat3" | "mat4";
const LEAF_TYPES: Record<LeafTypeName, Type> = {
  f32: Tf32, u32: Tu32, i32: Ti32,
  vec2: vec(2), vec3: vec(3), vec4: vec(4), mat3: mat(3), mat4: mat(4),
};

function transposed(t: Type): Type {
  return t.kind === "Matrix" ? { kind: "Matrix", element: t.element, rows: t.cols, cols: t.rows } : t;
}
function constVec4(x: number, y: number, z: number, w: number): Expr {
  const c = (v: number): Expr => ({ kind: "Const", type: Tf32, value: { kind: "Float", value: v } });
  return { kind: "NewVector", components: [c(x), c(y), c(z), c(w)], type: vec(4) };
}
function elemOf(t: Type): Type {
  return t.kind === "Vector" || t.kind === "Matrix" ? t.element : t;
}
/** A `CallIntrinsic` node — `op` is a minimal stub (this IR is only ever printed by the
 *  derived-uniform codegen, never run through the shader compiler). */
function intrinsicExpr(name: string, type: Type, args: readonly Expr[]): Expr {
  return {
    kind: "CallIntrinsic",
    op: { name, returnTypeOf: () => type, pure: true, emit: { glsl: name, wgsl: name } },
    args,
    type,
  };
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
  /** vecN swizzle, e.g. `.swizzle("xyz")`. Single component ⇒ scalar. */
  swizzle(comps: string): DerivedExpr {
    const c = comps.split("") as ("x" | "y" | "z" | "w")[];
    return new DerivedExpr({ kind: "VecSwizzle", value: this.ir, comps: c, type: c.length === 1 ? Tf32 : vec(c.length as 2 | 3 | 4) });
  }
  get x(): DerivedExpr { return this.swizzle("x"); }
  get y(): DerivedExpr { return this.swizzle("y"); }
  get z(): DerivedExpr { return this.swizzle("z"); }
  get w(): DerivedExpr { return this.swizzle("w"); }

  /**
   * Re-type a leaf — `u.<Name>` defaults to a mat4 trafo leaf; `.as("vec4")` (etc.) gives the
   * leaf its real type so you can read *any* uniform: `u.Tint.as("vec4").swizzle("xyz")`.
   * Only valid directly on a `u.<Name>` leaf.
   */
  as(typeName: LeafTypeName): DerivedExpr {
    if (this.ir.kind !== "ReadInput" || this.ir.scope !== "Uniform") {
      throw new Error("DerivedExpr.as(...): only valid on a `u.<Name>` uniform leaf");
    }
    return new DerivedExpr({ kind: "ReadInput", scope: "Uniform", name: this.ir.name, type: LEAF_TYPES[typeName] });
  }

  // ── Intrinsics — every WGSL builtin (name passed straight through). ──
  /** Escape hatch: an arbitrary builtin call `name(this, ...others)` producing `resultType`. */
  call(name: string, resultType: Type, ...others: DerivedExpr[]): DerivedExpr {
    return new DerivedExpr(intrinsicExpr(name, resultType, [this.ir, ...others.map((o) => o.ir)]));
  }
  private u1(name: string): DerivedExpr { return new DerivedExpr(intrinsicExpr(name, this.ir.type, [this.ir])); }
  private bin(name: string, o: DerivedExpr): DerivedExpr { return new DerivedExpr(intrinsicExpr(name, this.ir.type, [this.ir, o.ir])); }
  sin(): DerivedExpr { return this.u1("sin"); }
  cos(): DerivedExpr { return this.u1("cos"); }
  tan(): DerivedExpr { return this.u1("tan"); }
  abs(): DerivedExpr { return this.u1("abs"); }
  floor(): DerivedExpr { return this.u1("floor"); }
  ceil(): DerivedExpr { return this.u1("ceil"); }
  fract(): DerivedExpr { return this.u1("fract"); }
  sqrt(): DerivedExpr { return this.u1("sqrt"); }
  exp(): DerivedExpr { return this.u1("exp"); }
  log(): DerivedExpr { return this.u1("log"); }
  exp2(): DerivedExpr { return this.u1("exp2"); }
  log2(): DerivedExpr { return this.u1("log2"); }
  sign(): DerivedExpr { return this.u1("sign"); }
  normalize(): DerivedExpr { return this.u1("normalize"); }
  min(o: DerivedExpr): DerivedExpr { return this.bin("min", o); }
  max(o: DerivedExpr): DerivedExpr { return this.bin("max", o); }
  pow(o: DerivedExpr): DerivedExpr { return this.bin("pow", o); }
  atan2(o: DerivedExpr): DerivedExpr { return this.bin("atan2", o); }
  step(edge: DerivedExpr): DerivedExpr { return new DerivedExpr(intrinsicExpr("step", this.ir.type, [edge.ir, this.ir])); }
  clamp(lo: DerivedExpr, hi: DerivedExpr): DerivedExpr { return new DerivedExpr(intrinsicExpr("clamp", this.ir.type, [this.ir, lo.ir, hi.ir])); }
  mix(o: DerivedExpr, t: DerivedExpr): DerivedExpr { return new DerivedExpr(intrinsicExpr("mix", this.ir.type, [this.ir, o.ir, t.ir])); }
  dot(o: DerivedExpr): DerivedExpr { return new DerivedExpr(intrinsicExpr("dot", elemOf(this.ir.type), [this.ir, o.ir])); }
  cross(o: DerivedExpr): DerivedExpr { return new DerivedExpr(intrinsicExpr("cross", this.ir.type, [this.ir, o.ir])); }
  length(): DerivedExpr { return new DerivedExpr(intrinsicExpr("length", elemOf(this.ir.type), [this.ir])); }
  distance(o: DerivedExpr): DerivedExpr { return new DerivedExpr(intrinsicExpr("distance", elemOf(this.ir.type), [this.ir, o.ir])); }
  reflect(n: DerivedExpr): DerivedExpr { return new DerivedExpr(intrinsicExpr("reflect", this.ir.type, [this.ir, n.ir])); }

  // ── Comparisons (produce a bool DerivedExpr). Use these to feed `select`. ──
  private cmp(kind: "Eq" | "Neq" | "Lt" | "Le" | "Gt" | "Ge", other: DerivedExpr): DerivedExpr {
    const ir = { kind, lhs: this.ir, rhs: other.ir, type: { kind: "Bool" } as Type } as Expr;
    return new DerivedExpr(ir);
  }
  eq(o: DerivedExpr): DerivedExpr { return this.cmp("Eq", o); }
  ne(o: DerivedExpr): DerivedExpr { return this.cmp("Neq", o); }
  lt(o: DerivedExpr): DerivedExpr { return this.cmp("Lt", o); }
  le(o: DerivedExpr): DerivedExpr { return this.cmp("Le", o); }
  gt(o: DerivedExpr): DerivedExpr { return this.cmp("Gt", o); }
  ge(o: DerivedExpr): DerivedExpr { return this.cmp("Ge", o); }

  /** `select(this_when_true, else_when_false, cond)` — WGSL ternary. The
   *  receiver supplies the "true" branch; pass the "false" branch + a
   *  bool DerivedExpr as the condition. */
  select(ifFalse: DerivedExpr, cond: DerivedExpr): DerivedExpr {
    const ir = { kind: "Conditional", cond: cond.ir, ifTrue: this.ir, ifFalse: ifFalse.ir, type: this.ir.type } as Expr;
    return new DerivedExpr(ir);
  }

  /** Determinant of a square matrix expression — yields a scalar f32. */
  determinant(): DerivedExpr {
    if (this.ir.type.kind !== "Matrix") {
      throw new Error("DerivedExpr.determinant(): receiver must be a matrix expression");
    }
    return new DerivedExpr({ kind: "Determinant", value: this.ir, type: Tf32 } as Expr);
  }

  // ── Static literal builders. Mode rules in particular need u32 constants for enum
  //    indices (slot lookup). Floats for thresholds (e.g. determinant against zero).
  static f32(value: number): DerivedExpr {
    return new DerivedExpr({ kind: "Const", type: Tf32, value: { kind: "Float", value } } as Expr);
  }
  static u32(value: number): DerivedExpr {
    return new DerivedExpr({ kind: "Const", type: Tu32, value: { kind: "Int", value: value >>> 0, signed: false } } as Expr);
  }
  static i32(value: number): DerivedExpr {
    return new DerivedExpr({ kind: "Const", type: Ti32, value: { kind: "Int", value: value | 0, signed: true } } as Expr);
  }
}

/** The `u` passed to a `derivedUniform` builder. `u.<Name>` is a uniform leaf — a mat4
 *  trafo by default; its real type when the wombat-shader-vite plugin supplies one (or
 *  when you write `u.<Name>.as("vec4")`). */
export type DerivedScope = { readonly [name: string]: DerivedExpr };

/** Per-leaf type hints, keyed by uniform name (the build-time marker fills these in). */
export type DerivedLeafTypes = Readonly<Record<string, LeafTypeName>>;

function makeScope(leafTypes?: DerivedLeafTypes): DerivedScope {
  return new Proxy({} as DerivedScope, {
    get(_t, key): DerivedExpr {
      if (typeof key !== "string") throw new Error("derivedUniform: leaf names must be strings");
      const hinted = leafTypes?.[key];
      return new DerivedExpr(uniformRef(key, hinted ? LEAF_TYPES[hinted] : mat(4)));
    },
  });
}

/**
 * Define a derived uniform. `derivedUniform(u => u.ViewTrafo.mul(u.ModelTrafo))` makes a
 * rule that slots in wherever a uniform value goes — see this module's header. The optional
 * `leafTypes` map gives `u.<Name>` leaves their real WGSL types (the wombat-shader-vite
 * `derivedUniform(...)` marker fills it in from your `UniformScope` declarations; without
 * the plugin, `u.<Name>` defaults to mat4 and you use `.as("vec4")` etc. when you need
 * another type).
 */
export function derivedUniform<T = unknown>(
  build: (u: DerivedScope) => DerivedExpr,
  leafTypes?: DerivedLeafTypes,
): DerivedRule<T> {
  const result = build(makeScope(leafTypes));
  if (!(result instanceof DerivedExpr)) {
    throw new Error("derivedUniform: the builder must return a derived expression, e.g. u.ViewTrafo.mul(u.ModelTrafo)");
  }
  return ruleFromIR<T>(result.ir, result.ir.type);
}
