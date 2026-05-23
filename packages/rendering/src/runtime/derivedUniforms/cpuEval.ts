// CPU interpreter for derived-uniform rule IR (a wombat.shader `Expr`).
//
// Evaluates a `derivedUniform(u => …)` rule in DOUBLE precision (M44d / V3d /
// …) — no df32 dual-float trick, no GPU constituent slots. The heap path
// compiles rules into a §7 GPU compute pass (f32, df32 for the inverse trick);
// here we just interpret the IR with f64 host math (`M44d.inverse()` is exact
// enough that the backward-storage trick is unnecessary), and `writeField`
// narrows the result to f32 at pack time.
//
// Used by the legacy per-RO uniform path so derived uniforms work on ROs that
// aren't heap-eligible (a storage buffer, 2+ textures, …). The evaluator is a
// standalone `Expr → value` function so other CPU rule paths can reuse it.

import type { Expr, Type } from "@aardworx/wombat.shader/ir";
import { V2d, V3d, V4d, M22d, M33d, M44d } from "@aardworx/wombat.base";

/** A value produced by the interpreter (host doubles). */
export type EvalValue = number | boolean | V2d | V3d | V4d | M22d | M33d | M44d;

/** Resolve a `u.<name>` leaf to its current host value (Trafo3d | M44f | M44d |
 *  V*f | number | …) — or undefined if unbound. */
export type LeafReader = (name: string) => unknown;

// ── runtime classification ───────────────────────────────────────────────
function isVec(v: unknown): v is V2d | V3d | V4d {
  return v instanceof V2d || v instanceof V3d || v instanceof V4d;
}
function isMat(v: unknown): v is M22d | M33d | M44d {
  return v instanceof M22d || v instanceof M33d || v instanceof M44d;
}
function data(v: EvalValue): Float64Array { return (v as unknown as { _data: Float64Array })._data; }
function num(v: EvalValue): number {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  throw new Error("derived eval: expected scalar, got " + describe(v));
}
function describe(v: unknown): string {
  if (isVec(v)) return "vec" + data(v as EvalValue).length;
  if (isMat(v)) return "mat";
  return typeof v;
}
function vecOf(d: ArrayLike<number>): V2d | V3d | V4d {
  switch (d.length) {
    case 2: return new V2d(d[0]!, d[1]!);
    case 3: return new V3d(d[0]!, d[1]!, d[2]!);
    case 4: return new V4d(d[0]!, d[1]!, d[2]!, d[3]!);
    default: throw new Error("derived eval: bad vector arity " + d.length);
  }
}

// ── leaf coercion (by the IR leaf type) ──────────────────────────────────
function toM44(raw: unknown): M44d {
  if (raw instanceof M44d) return raw;
  const o = raw as { forward?: unknown; _data?: ArrayLike<number> };
  if (o && o.forward !== undefined) return toM44(o.forward);          // Trafo3d → forward
  if (o && o._data !== undefined && o._data.length >= 16) return M44d.fromArray(o._data); // M44f
  throw new Error("derived eval: leaf is not a mat4: " + describe(raw));
}
function upperLeft3(m: M44d): M33d {
  const d = data(m); // row-major r*4+c
  return M33d.fromArray([d[0]!, d[1]!, d[2]!, d[4]!, d[5]!, d[6]!, d[8]!, d[9]!, d[10]!]);
}
function toM33(raw: unknown): M33d {
  if (raw instanceof M33d) return raw;
  const o = raw as { forward?: unknown; _data?: ArrayLike<number> };
  if (raw instanceof M44d) return upperLeft3(raw);
  if (o && o.forward !== undefined) return upperLeft3(toM44(o.forward));
  if (o && o._data !== undefined && o._data.length === 9) return M33d.fromArray(o._data);
  if (o && o._data !== undefined && o._data.length >= 16) return upperLeft3(toM44(raw));
  throw new Error("derived eval: leaf is not a mat3: " + describe(raw));
}
function toM22(raw: unknown): M22d {
  if (raw instanceof M22d) return raw;
  const o = raw as { _data?: ArrayLike<number> };
  if (o && o._data !== undefined && o._data.length === 4) return M22d.fromArray(o._data);
  throw new Error("derived eval: leaf is not a mat2: " + describe(raw));
}
function toVec(raw: unknown, n: 2 | 3 | 4): V2d | V3d | V4d {
  if (isVec(raw)) return raw;
  const o = raw as { _data?: ArrayLike<number> };
  if (o && o._data !== undefined) {
    const d = o._data;
    return vecOf(n === 2 ? [d[0]!, d[1]!] : n === 3 ? [d[0]!, d[1]!, d[2]!] : [d[0]!, d[1]!, d[2]!, d[3]!]);
  }
  throw new Error("derived eval: leaf is not a vec" + n + ": " + describe(raw));
}
function coerce(raw: unknown, t: Type): EvalValue {
  switch (t.kind) {
    case "Matrix": return t.rows === 4 ? toM44(raw) : t.rows === 3 ? toM33(raw) : toM22(raw);
    case "Vector": return toVec(raw, t.dim);
    case "Float":  return Number(raw);
    case "Int":    return Math.trunc(Number(raw));
    case "Bool":   return Boolean(raw);
    default: throw new Error("derived eval: unsupported leaf type " + t.kind);
  }
}

// ── arithmetic over scalar / vector / matrix ─────────────────────────────
function negate(x: EvalValue): EvalValue {
  if (typeof x === "number") return -x;
  if (isVec(x) || isMat(x)) return remap(x, (a) => -a);
  throw new Error("derived eval: cannot negate " + describe(x));
}
/** Map a vector/matrix's components through `f`, returning the same shape. */
function remap(v: V2d | V3d | V4d | M22d | M33d | M44d, f: (x: number) => number): EvalValue {
  const d = data(v); const out = new Array<number>(d.length);
  for (let i = 0; i < d.length; i++) out[i] = f(d[i]!);
  return isMat(v) ? matOf(out) : vecOf(out);
}
function matOf(d: ArrayLike<number>): M22d | M33d | M44d {
  switch (d.length) {
    case 4:  return M22d.fromArray(d);
    case 9:  return M33d.fromArray(d);
    case 16: return M44d.fromArray(d);
    default: throw new Error("derived eval: bad matrix arity " + d.length);
  }
}
/** Element-wise binary op, broadcasting a scalar against a vector/matrix. */
function elementwise(a: EvalValue, b: EvalValue, f: (x: number, y: number) => number): EvalValue {
  if (typeof a === "number" && typeof b === "number") return f(a, b);
  if (isVec(a) || isMat(a)) {
    const da = data(a);
    const out = new Array<number>(da.length);
    if (typeof b === "number") { for (let i = 0; i < da.length; i++) out[i] = f(da[i]!, b); }
    else { const db = data(b as EvalValue); for (let i = 0; i < da.length; i++) out[i] = f(da[i]!, db[i]!); }
    return isMat(a) ? matOf(out) : vecOf(out);
  }
  if (typeof a === "number" && (isVec(b) || isMat(b))) {
    const db = data(b); const out = new Array<number>(db.length);
    for (let i = 0; i < db.length; i++) out[i] = f(a, db[i]!);
    return isMat(b) ? matOf(out) : vecOf(out);
  }
  throw new Error(`derived eval: cannot combine ${describe(a)} and ${describe(b)}`);
}
/** `Mul` node — scalar·scalar, scalar·(vec|mat) broadcast, or component-wise
 *  vec·vec (mat·mat / mat·vec / vec·mat have their own node kinds). */
function mulNode(a: EvalValue, b: EvalValue): EvalValue {
  if (typeof a === "number" && isMat(b)) return (b as M44d).mul(a) as EvalValue;
  if (typeof b === "number" && isMat(a)) return (a as M44d).mul(b) as EvalValue;
  return elementwise(a, b, (x, y) => x * y);
}
function vlen(v: EvalValue): number {
  if (typeof v === "number") return Math.abs(v);
  const d = data(v); let s = 0; for (let i = 0; i < d.length; i++) s += d[i]! * d[i]!;
  return Math.sqrt(s);
}
function dot(a: EvalValue, b: EvalValue): number {
  const da = data(a), db = data(b); let s = 0;
  for (let i = 0; i < da.length; i++) s += da[i]! * db[i]!;
  return s;
}
function normalize(v: EvalValue): EvalValue {
  const len = vlen(v); return len === 0 ? v : remap(v as V3d, (x) => x / len);
}
function swizzle(v: EvalValue, comps: readonly string[]): EvalValue {
  const d = data(v); const idx = (c: string): number => ({ x: 0, y: 1, z: 2, w: 3 }[c] ?? 0);
  if (comps.length === 1) return d[idx(comps[0]!)]!;
  return vecOf(comps.map((c) => d[idx(c)]!));
}
function convertMatrix(v: EvalValue, t: Type): EvalValue {
  if (t.kind === "Matrix" && t.rows === 3 && v instanceof M44d) return upperLeft3(v);
  if (isMat(v)) return v; // identity-ish convert
  throw new Error("derived eval: ConvertMatrix on " + describe(v));
}

// ── intrinsics (vec-componentwise where natural) ─────────────────────────
const UNARY: Record<string, (x: number) => number> = {
  sin: Math.sin, cos: Math.cos, tan: Math.tan, abs: Math.abs, floor: Math.floor,
  ceil: Math.ceil, sqrt: Math.sqrt, exp: Math.exp, log: Math.log,
  exp2: (x) => 2 ** x, log2: Math.log2, sign: Math.sign, fract: (x) => x - Math.floor(x),
};
const BINARY: Record<string, (x: number, y: number) => number> = {
  min: Math.min, max: Math.max, pow: (x, y) => x ** y, atan2: Math.atan2,
};
function intrinsic(name: string, a: EvalValue[]): EvalValue {
  if (UNARY[name]) return typeof a[0] === "number" ? UNARY[name]!(a[0]) : remap(a[0] as V3d, UNARY[name]!);
  if (BINARY[name]) return elementwise(a[0]!, a[1]!, BINARY[name]!);
  switch (name) {
    case "normalize": return normalize(a[0]!);
    case "length":    return vlen(a[0]!);
    case "dot":       return dot(a[0]!, a[1]!);
    case "cross":     return (a[0] as V3d).cross(a[1] as V3d);
    case "distance":  return vlen(elementwise(a[0]!, a[1]!, (x, y) => x - y));
    case "step":      return elementwise(a[1]!, a[0]!, (x, edge) => (x < edge ? 0 : 1));
    case "clamp":     return elementwise(elementwise(a[0]!, a[1]!, Math.max), a[2]!, Math.min);
    case "mix": {     // a + (b - a) * t
      const diff = elementwise(a[1]!, a[0]!, (y, x) => y - x);
      const scaled = elementwise(diff, a[2]!, (d, t) => d * t);
      return elementwise(a[0]!, scaled, (x, s) => x + s);
    }
    case "reflect": { const d = dot(a[0]!, a[1]!); return elementwise(a[0]!, elementwise(a[1]!, 2 * d, (n, k) => n * k), (i, s) => i - s); }
    default: throw new Error("derived eval: unsupported intrinsic '" + name + "'");
  }
}

/**
 * Interpret a derived-uniform rule IR, reading `u.<name>` leaves via
 * `readLeaf`. Returns a host double value (`M44d` / `V3d` / number / …) that
 * `writeField` narrows to f32 when packing the UBO.
 */
export function interpretExpr(ir: Expr, readLeaf: LeafReader): EvalValue {
  const ev = (e: Expr): EvalValue => {
    switch (e.kind) {
      case "Const": {
        const v = e.value;
        if (v.kind === "Bool") return v.value;
        if (v.kind === "Int" || v.kind === "Float") return v.value;
        throw new Error("derived eval: null const");
      }
      case "ReadInput": {
        if (e.scope !== "Uniform") throw new Error("derived eval: non-uniform leaf scope " + e.scope);
        const raw = readLeaf(e.name);
        if (raw === undefined) throw new Error("derived eval: missing uniform leaf '" + e.name + "'");
        return coerce(raw, e.type);
      }
      case "Neg": return negate(ev(e.value));
      case "Not": return !(ev(e.value) as boolean);
      case "Add": return elementwise(ev(e.lhs), ev(e.rhs), (x, y) => x + y);
      case "Sub": return elementwise(ev(e.lhs), ev(e.rhs), (x, y) => x - y);
      case "Div": return elementwise(ev(e.lhs), ev(e.rhs), (x, y) => x / y);
      case "Mod": return elementwise(ev(e.lhs), ev(e.rhs), (x, y) => x % y);
      case "Mul": return mulNode(ev(e.lhs), ev(e.rhs));
      case "MulMatMat": return (ev(e.lhs) as M44d).mul(ev(e.rhs) as M44d) as EvalValue;
      case "MulMatVec": return (ev(e.lhs) as M44d).mul(ev(e.rhs) as V4d) as EvalValue;
      case "MulVecMat": return (ev(e.rhs) as M44d).transpose().mul(ev(e.lhs) as V4d) as EvalValue;
      case "Transpose": return (ev(e.value) as M44d).transpose();
      case "Inverse": return (ev(e.value) as M44d).inverse();
      case "Determinant": return (ev(e.value) as M44d).determinant();
      case "Dot": return dot(ev(e.lhs), ev(e.rhs));
      case "Cross": return (ev(e.lhs) as V3d).cross(ev(e.rhs) as V3d);
      case "Length": return vlen(ev(e.value));
      case "ConvertMatrix": return convertMatrix(ev(e.value), e.type);
      case "VecSwizzle": return swizzle(ev(e.value), e.comps);
      case "NewVector": return vecOf(e.components.map((c) => num(ev(c))));
      case "And": return (ev(e.lhs) as boolean) && (ev(e.rhs) as boolean);
      case "Or": return (ev(e.lhs) as boolean) || (ev(e.rhs) as boolean);
      case "Eq":  return num(ev(e.lhs)) === num(ev(e.rhs));
      case "Neq": return num(ev(e.lhs)) !== num(ev(e.rhs));
      case "Lt":  return num(ev(e.lhs)) <   num(ev(e.rhs));
      case "Le":  return num(ev(e.lhs)) <=  num(ev(e.rhs));
      case "Gt":  return num(ev(e.lhs)) >   num(ev(e.rhs));
      case "Ge":  return num(ev(e.lhs)) >=  num(ev(e.rhs));
      case "Conditional": return (ev(e.cond) as boolean) ? ev(e.ifTrue) : ev(e.ifFalse);
      case "CallIntrinsic": return intrinsic(e.op.name, e.args.map(ev));
      default: throw new Error("derived eval: unsupported IR node '" + e.kind + "'");
    }
  };
  return ev(ir);
}
