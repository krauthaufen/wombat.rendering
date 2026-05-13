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

/** Convertible-to-DerivedExpr operand. `aardvark-operators` rewrites
 *  e.g. `det < 0` to `det.lt(0)` — so every binary method accepts a
 *  raw number and lifts it to a Const matching the receiver's
 *  numeric type (f32 / u32 / i32). */
export type DerivedOperand = DerivedExpr | number;

/** True iff `t` is a numeric scalar IR type (Float / Int). */
function isNumericScalar(t: Type): boolean {
  return t.kind === "Float" || t.kind === "Int";
}
/** Element type for scalar arithmetic: scalar types pass through,
 *  vector/matrix types unwrap to their element type, anything else
 *  defaults to f32 (the natural numeric on the GPU side). */
function scalarLiftType(t: Type): Type {
  if (isNumericScalar(t)) return t;
  if (t.kind === "Vector" || t.kind === "Matrix") return t.element;
  return Tf32;
}
function liftConst(value: number, target: Type): Expr {
  if (target.kind === "Int") {
    if (target.signed) {
      return { kind: "Const", type: target, value: { kind: "Int", value: value | 0, signed: true } } as Expr;
    }
    return { kind: "Const", type: target, value: { kind: "Int", value: value >>> 0, signed: false } } as Expr;
  }
  // Default: f32 const.
  return { kind: "Const", type: Tf32, value: { kind: "Float", value } } as Expr;
}
/** Coerce `DerivedOperand` to a DerivedExpr, lifting raw numbers to a
 *  Const expression whose type matches the receiver's scalar type. */
function lift(other: DerivedOperand, receiverType: Type): DerivedExpr {
  if (other instanceof DerivedExpr) return other;
  return new DerivedExpr(liftConst(other, scalarLiftType(receiverType)));
}

/**
 * A traced expression in a `derivedUniform` / `derivedMode` builder.
 *
 * Plays nicely with `boperators` (https://npmjs.com/package/boperators):
 * the named-static operator methods (`static ["+"]`, `static ["*"]`,
 * `static ["<"]`, …) make `a + b`, `a * b`, `a < b`, etc. on
 * DerivedExprs rewrite to those statics at build time. Without the
 * plugin, the same logic is reachable via the instance methods
 * (`.add`, `.mul`, `.lt`, …).
 *
 * Ternary `a < b ? x : y` isn't rewritten — call `.select` on the
 * resulting bool DerivedExpr:
 *
 *     (det < 0).select(flipped, declared)
 */
export class DerivedExpr {
  constructor(readonly ir: Expr) {}
  get type(): Type { return this.ir.type; }

  /** Matrix product `this · other` (also matrix·vector / vector·matrix by operand types). */
  mul(other: DerivedOperand): DerivedExpr {
    const o = lift(other, this.ir.type);
    const a = this.ir.type, b = o.ir.type;
    if (a.kind === "Matrix" && b.kind === "Matrix") {
      return new DerivedExpr({ kind: "MulMatMat", lhs: this.ir, rhs: o.ir, type: { kind: "Matrix", element: a.element, rows: a.rows, cols: b.cols } });
    }
    if (a.kind === "Matrix" && b.kind === "Vector") {
      return new DerivedExpr({ kind: "MulMatVec", lhs: this.ir, rhs: o.ir, type: { kind: "Vector", element: b.element, dim: a.rows } });
    }
    if (a.kind === "Vector" && b.kind === "Matrix") {
      return new DerivedExpr({ kind: "MulVecMat", lhs: this.ir, rhs: o.ir, type: { kind: "Vector", element: a.element, dim: b.cols } });
    }
    return new DerivedExpr({ kind: "Mul", lhs: this.ir, rhs: o.ir, type: a.kind === "Float" || a.kind === "Int" ? b : a });
  }
  add(other: DerivedOperand): DerivedExpr { const o = lift(other, this.ir.type); return new DerivedExpr({ kind: "Add", lhs: this.ir, rhs: o.ir, type: this.ir.type }); }
  sub(other: DerivedOperand): DerivedExpr { const o = lift(other, this.ir.type); return new DerivedExpr({ kind: "Sub", lhs: this.ir, rhs: o.ir, type: this.ir.type }); }
  div(other: DerivedOperand): DerivedExpr { const o = lift(other, this.ir.type); return new DerivedExpr({ kind: "Div", lhs: this.ir, rhs: o.ir, type: this.ir.type }); }
  mod(other: DerivedOperand): DerivedExpr { const o = lift(other, this.ir.type); return new DerivedExpr({ kind: "Mod", lhs: this.ir, rhs: o.ir, type: this.ir.type }); }
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
  min(o: DerivedOperand): DerivedExpr { return this.bin("min", lift(o, this.ir.type)); }
  max(o: DerivedOperand): DerivedExpr { return this.bin("max", lift(o, this.ir.type)); }
  pow(o: DerivedOperand): DerivedExpr { return this.bin("pow", lift(o, this.ir.type)); }
  atan2(o: DerivedOperand): DerivedExpr { return this.bin("atan2", lift(o, this.ir.type)); }
  step(edge: DerivedOperand): DerivedExpr { const e = lift(edge, this.ir.type); return new DerivedExpr(intrinsicExpr("step", this.ir.type, [e.ir, this.ir])); }
  clamp(lo: DerivedOperand, hi: DerivedOperand): DerivedExpr { const l = lift(lo, this.ir.type); const h = lift(hi, this.ir.type); return new DerivedExpr(intrinsicExpr("clamp", this.ir.type, [this.ir, l.ir, h.ir])); }
  mix(o: DerivedOperand, t: DerivedOperand): DerivedExpr { const oo = lift(o, this.ir.type); const tt = lift(t, this.ir.type); return new DerivedExpr(intrinsicExpr("mix", this.ir.type, [this.ir, oo.ir, tt.ir])); }
  dot(o: DerivedExpr): DerivedExpr { return new DerivedExpr(intrinsicExpr("dot", elemOf(this.ir.type), [this.ir, o.ir])); }
  cross(o: DerivedExpr): DerivedExpr { return new DerivedExpr(intrinsicExpr("cross", this.ir.type, [this.ir, o.ir])); }
  length(): DerivedExpr { return new DerivedExpr(intrinsicExpr("length", elemOf(this.ir.type), [this.ir])); }
  distance(o: DerivedExpr): DerivedExpr { return new DerivedExpr(intrinsicExpr("distance", elemOf(this.ir.type), [this.ir, o.ir])); }
  reflect(n: DerivedExpr): DerivedExpr { return new DerivedExpr(intrinsicExpr("reflect", this.ir.type, [this.ir, n.ir])); }

  // ── Comparisons (produce a bool DerivedExpr). Receivers + arg can be
  //    raw numbers — `aardvark-operators` lifts `a < 2` to `a.lt(2)`,
  //    and we lift `2` to a const matching the receiver's scalar type. ──
  private cmp(kind: "Eq" | "Neq" | "Lt" | "Le" | "Gt" | "Ge", other: DerivedOperand): DerivedExpr {
    const o = lift(other, this.ir.type);
    const ir = { kind, lhs: this.ir, rhs: o.ir, type: { kind: "Bool" } as Type } as Expr;
    return new DerivedExpr(ir);
  }
  eq(o: DerivedOperand): DerivedExpr { return this.cmp("Eq", o); }
  ne(o: DerivedOperand): DerivedExpr { return this.cmp("Neq", o); }
  lt(o: DerivedOperand): DerivedExpr { return this.cmp("Lt", o); }
  le(o: DerivedOperand): DerivedExpr { return this.cmp("Le", o); }
  gt(o: DerivedOperand): DerivedExpr { return this.cmp("Gt", o); }
  ge(o: DerivedOperand): DerivedExpr { return this.cmp("Ge", o); }

  /**
   * Ternary select. The RECEIVER must be a bool — typically the
   * result of a comparison — and the args are the then / else
   * branches:
   *
   *     (det < 0).select(flipped, declared)   // det<0 ? flipped : declared
   *
   * The result type follows `then` (which must be widening-
   * compatible with `else`). Raw numbers are lifted using `then`'s
   * type when `then` is a DerivedExpr, else f32.
   */
  select(thenE: DerivedOperand, elseE: DerivedOperand): DerivedExpr {
    if (this.ir.type.kind !== "Bool") {
      throw new Error("DerivedExpr.select(then, else): receiver must be a bool (typically a comparison)");
    }
    const t = thenE instanceof DerivedExpr ? thenE : lift(thenE, elseE instanceof DerivedExpr ? elseE.ir.type : Tf32);
    const e = elseE instanceof DerivedExpr ? elseE : lift(elseE, t.ir.type);
    const ir = { kind: "Conditional", cond: this.ir, ifTrue: t.ir, ifFalse: e.ir, type: t.ir.type } as Expr;
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

  // ─── boperators static dispatch ──────────────────────────────────────
  // `a OP b` (where at least one operand is a DerivedExpr) is rewritten
  // by the boperators plugin to `DerivedExpr["OP"](a, b)`. Either-order
  // overloads (`number OP expr`, `expr OP number`) are handled by lifting
  // the raw number to a Const matching the other operand's scalar type.

  static "+"(a: DerivedExpr, b: DerivedExpr): DerivedExpr;
  static "+"(a: DerivedExpr, b: number): DerivedExpr;
  static "+"(a: number, b: DerivedExpr): DerivedExpr;
  static "+"(a: DerivedOperand, b: DerivedOperand): DerivedExpr {
    const x = a instanceof DerivedExpr ? a : lift(a, (b as DerivedExpr).ir.type);
    return x.add(b);
  }
  static "-"(a: DerivedExpr, b: DerivedExpr): DerivedExpr;
  static "-"(a: DerivedExpr, b: number): DerivedExpr;
  static "-"(a: number, b: DerivedExpr): DerivedExpr;
  static "-"(a: DerivedExpr): DerivedExpr; // unary minus
  static "-"(a: DerivedOperand, b?: DerivedOperand): DerivedExpr {
    if (b === undefined) return (a as DerivedExpr).neg();
    const x = a instanceof DerivedExpr ? a : lift(a, (b as DerivedExpr).ir.type);
    return x.sub(b);
  }
  static "*"(a: DerivedExpr, b: DerivedExpr): DerivedExpr;
  static "*"(a: DerivedExpr, b: number): DerivedExpr;
  static "*"(a: number, b: DerivedExpr): DerivedExpr;
  static "*"(a: DerivedOperand, b: DerivedOperand): DerivedExpr {
    const x = a instanceof DerivedExpr ? a : lift(a, (b as DerivedExpr).ir.type);
    return x.mul(b);
  }
  static "/"(a: DerivedExpr, b: DerivedExpr): DerivedExpr;
  static "/"(a: DerivedExpr, b: number): DerivedExpr;
  static "/"(a: number, b: DerivedExpr): DerivedExpr;
  static "/"(a: DerivedOperand, b: DerivedOperand): DerivedExpr {
    const x = a instanceof DerivedExpr ? a : lift(a, (b as DerivedExpr).ir.type);
    return x.div(b);
  }
  static "%"(a: DerivedExpr, b: DerivedExpr): DerivedExpr;
  static "%"(a: DerivedExpr, b: number): DerivedExpr;
  static "%"(a: number, b: DerivedExpr): DerivedExpr;
  static "%"(a: DerivedOperand, b: DerivedOperand): DerivedExpr {
    const x = a instanceof DerivedExpr ? a : lift(a, (b as DerivedExpr).ir.type);
    return x.mod(b);
  }
  static "<"(a: DerivedExpr, b: DerivedExpr): DerivedExpr;
  static "<"(a: DerivedExpr, b: number): DerivedExpr;
  static "<"(a: number, b: DerivedExpr): DerivedExpr;
  static "<"(a: DerivedOperand, b: DerivedOperand): DerivedExpr {
    const x = a instanceof DerivedExpr ? a : lift(a, (b as DerivedExpr).ir.type);
    return x.lt(b);
  }
  static "<="(a: DerivedExpr, b: DerivedExpr): DerivedExpr;
  static "<="(a: DerivedExpr, b: number): DerivedExpr;
  static "<="(a: number, b: DerivedExpr): DerivedExpr;
  static "<="(a: DerivedOperand, b: DerivedOperand): DerivedExpr {
    const x = a instanceof DerivedExpr ? a : lift(a, (b as DerivedExpr).ir.type);
    return x.le(b);
  }
  static ">"(a: DerivedExpr, b: DerivedExpr): DerivedExpr;
  static ">"(a: DerivedExpr, b: number): DerivedExpr;
  static ">"(a: number, b: DerivedExpr): DerivedExpr;
  static ">"(a: DerivedOperand, b: DerivedOperand): DerivedExpr {
    const x = a instanceof DerivedExpr ? a : lift(a, (b as DerivedExpr).ir.type);
    return x.gt(b);
  }
  static ">="(a: DerivedExpr, b: DerivedExpr): DerivedExpr;
  static ">="(a: DerivedExpr, b: number): DerivedExpr;
  static ">="(a: number, b: DerivedExpr): DerivedExpr;
  static ">="(a: DerivedOperand, b: DerivedOperand): DerivedExpr {
    const x = a instanceof DerivedExpr ? a : lift(a, (b as DerivedExpr).ir.type);
    return x.ge(b);
  }
  static "=="(a: DerivedExpr, b: DerivedExpr): DerivedExpr;
  static "=="(a: DerivedExpr, b: number): DerivedExpr;
  static "=="(a: number, b: DerivedExpr): DerivedExpr;
  static "=="(a: DerivedOperand, b: DerivedOperand): DerivedExpr {
    const x = a instanceof DerivedExpr ? a : lift(a, (b as DerivedExpr).ir.type);
    return x.eq(b);
  }
  static "!="(a: DerivedExpr, b: DerivedExpr): DerivedExpr;
  static "!="(a: DerivedExpr, b: number): DerivedExpr;
  static "!="(a: number, b: DerivedExpr): DerivedExpr;
  static "!="(a: DerivedOperand, b: DerivedOperand): DerivedExpr {
    const x = a instanceof DerivedExpr ? a : lift(a, (b as DerivedExpr).ir.type);
    return x.ne(b);
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
