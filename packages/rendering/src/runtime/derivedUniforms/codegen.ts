// §7 v2 — uber-kernel codegen.
//
// Emits the single compute kernel from the rule registry. One thread per
// record; each thread switches on `rule_id` and runs the matching arm:
// load the inputs (df32 from a constituent slot, or plain f32 from the
// drawHeader), do the math, write the result to MainHeap.
//
// v0 supports the rule shapes the standard trafo recipes need:
//   • a single mat4 input, output mat4  → collapse
//   • a left-fold MulMatMat over N mat4 inputs, output mat4 → dfmulN + collapse
//   • a single mat4 input, output mat3  → upper-3×3 of the transposed input (normal matrix)
// Arbitrary IR rules (the wgsl `expr()` lowering path) land in a follow-up;
// codegen throws on a shape it doesn't recognise. See docs/derived-uniforms-extensible.md.

import type { Expr, Type, Literal } from "@aardworx/wombat.shader/ir";
import { visitExprChildren } from "@aardworx/wombat.shader/ir";
import { mapExpr } from "@aardworx/wombat.shader/passes";
import type { DerivedUniformRegistry, RuleEntry } from "./registry.js";

export interface UberKernel {
  readonly wgsl: string;
  /** Record stride (u32 words) the kernel was generated for — must match RecordsBuffer.strideWords. */
  readonly strideU32: number;
}

// ─── Rule-shape classification ────────────────────────────────────────

type Shape =
  | { kind: "collapse"; arity: 1 }                 // mat4 in → mat4 out (df32-precise)
  | { kind: "matmulChain"; arity: number }         // L1·…·LN (all mat4) → mat4 out (df32-precise)
  | { kind: "normalMatrix"; arity: 1 }             // mat4 in → mat3 out (upper-3×3 of transpose, df32)
  | { kind: "generic"; arity: number };            // arbitrary IR, lowered via the wgsl printer (f32)

function isMat4(e: Expr): boolean {
  const t = e.type;
  return t.kind === "Matrix" && t.rows === 4 && t.cols === 4 && t.element.kind === "Float";
}
function isMat3(t: Expr["type"]): boolean {
  return t.kind === "Matrix" && t.rows === 3 && t.cols === 3 && t.element.kind === "Float";
}

/** A mat4 "leaf" is `ReadInput("Uniform", …)` or `Inverse(ReadInput("Uniform", …))`. */
function isMat4Leaf(e: Expr): boolean {
  if (e.kind === "ReadInput" && e.scope === "Uniform" && isMat4(e)) return true;
  if (e.kind === "Inverse" && isMat4(e) && e.value.kind === "ReadInput" && e.value.scope === "Uniform") return true;
  return false;
}

/** Flatten a left-assoc `MulMatMat` tree into its leaf list (left→right), or undefined if it isn't one. */
function matmulChainLeaves(e: Expr): Expr[] | undefined {
  if (isMat4Leaf(e)) return [e];
  if (e.kind === "MulMatMat" && isMat4(e)) {
    const left = matmulChainLeaves(e.lhs);
    const right = matmulChainLeaves(e.rhs);
    if (left && right) return [...left, ...right];
  }
  return undefined;
}

function classify(entry: RuleEntry): Shape {
  const ir = entry.ir;
  // Preferred (df32-precise) shapes for the standard trafo recipes.
  if (isMat3(entry.outputType) && entry.inputs.length === 1) return { kind: "normalMatrix", arity: 1 };
  if (!isMat3(entry.outputType)) {
    const chain = matmulChainLeaves(ir);
    if (chain !== undefined) {
      if (chain.length === 1) return { kind: "collapse", arity: 1 };
      // The arm consumes inputs positionally in chain order; that matches `entry.inputs`
      // (first-appearance order) only when no leaf is repeated — otherwise fall to generic.
      if (chain.length === entry.inputs.length) return { kind: "matmulChain", arity: chain.length };
    }
  }
  // Anything else: lower the IR via the wgsl printer (single precision).
  return { kind: "generic", arity: entry.inputs.length };
}

// ─── WGSL fragments ───────────────────────────────────────────────────

const DF32_LIB = /* wgsl */ `
fn split12(a: f32) -> vec2<f32> {
  let hi = bitcast<f32>(bitcast<u32>(a) & 0xFFFFE000u);
  return vec2<f32>(hi, a - hi);
}
fn two_sum(a: f32, b: f32) -> vec2<f32> {
  let s  = a + b;
  let bb = fma(1.0, s, -a);
  let t1 = fma(1.0, s, -bb);
  let t2 = fma(1.0, a, -t1);
  let t3 = fma(1.0, b, -bb);
  return vec2<f32>(s, t2 + t3);
}
fn quick_two_sum(a: f32, b: f32) -> vec2<f32> {
  let s = a + b;
  let t = fma(1.0, s, -a);
  return vec2<f32>(s, fma(1.0, b, -t));
}
fn two_prod(a: f32, b: f32) -> vec2<f32> {
  let p = a * b;
  let A = split12(a);
  let B = split12(b);
  let err = ((A.x * B.x - p) + A.x * B.y + A.y * B.x) + A.y * B.y;
  return vec2<f32>(p, err);
}
fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let s = two_sum(a.x, b.x);
  let t = two_sum(a.y, b.y);
  let s3 = quick_two_sum(s.x, s.y + t.x);
  return quick_two_sum(s3.x, s3.y + t.y);
}
fn df_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let p      = two_prod(a.x, b.x);
  let cross1 = fma(a.x, b.y, p.y);
  let cross  = fma(a.y, b.x, cross1);
  return quick_two_sum(p.x, cross);
}
`;

// Tagged-handle decode (must mirror records.ts: tag in top 3 bits, payload in low 29).
const HANDLE_HELPERS = /* wgsl */ `
const SLOT_PAYLOAD_MASK: u32 = 0x1FFFFFFFu;
fn slot_tag(h: u32) -> u32 { return h >> 29u; }
fn slot_payload(h: u32) -> u32 { return h & SLOT_PAYLOAD_MASK; }

// One df32 mat4 entry (r,c) of the matrix referenced by handle h.
//   tag 0 = constituent slot: Constituents[idx*16 + r*4 + c]   (df32 hi/lo)
//   tag 1 = host drawHeader:  MainHeap[(byte>>2) + r*4 + c]     (plain f32 → (v, 0))
fn load_entry_mat4(h: u32, r: u32, c: u32) -> vec2<f32> {
  let tag = slot_tag(h);
  let p   = slot_payload(h);
  if (tag == 0u) {
    return Constituents[p * 16u + r * 4u + c];
  }
  return vec2<f32>(MainHeap[(p >> 2u) + r * 4u + c], 0.0);
}

fn write_mat4_entry(out_byte: u32, r: u32, c: u32, v: f32) {
  MainHeap[(out_byte >> 2u) + r * 4u + c] = v;
}
fn write_mat3_entry(out_byte: u32, r: u32, c: u32, v: f32) {
  // std140 mat3<f32>: 3 rows × 4-float stride; the 4th column per row is left untouched.
  MainHeap[(out_byte >> 2u) + r * 4u + c] = v;
}
`;

function bindings(strideU32: number): string {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read>       Constituents: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> MainHeap:     array<f32>;
@group(0) @binding(2) var<storage, read>       RecordData:   array<u32>;
struct CountUniform { count: u32 }
@group(0) @binding(3) var<uniform>             Count: CountUniform;

const RECORD_STRIDE: u32 = ${strideU32 >>> 0}u;
`;
}

/** `collapse`: copy a single mat4 input (collapsing df32 hi+lo) to the output. */
function emitCollapseArm(id: number): string {
  return /* wgsl */ `
fn arm_${id}(in0: u32, out_byte: u32) {
  for (var r: u32 = 0u; r < 4u; r = r + 1u) {
    for (var c: u32 = 0u; c < 4u; c = c + 1u) {
      let e = load_entry_mat4(in0, r, c);
      write_mat4_entry(out_byte, r, c, e.x + e.y);
    }
  }
}
`;
}

/** `normalMatrix`: mat3 = upper-3×3 of the transpose of the single mat4 input. NM[i,j] = M[j,i]. */
function emitNormalMatrixArm(id: number): string {
  return /* wgsl */ `
fn arm_${id}(in0: u32, out_byte: u32) {
  for (var i: u32 = 0u; i < 3u; i = i + 1u) {
    for (var j: u32 = 0u; j < 3u; j = j + 1u) {
      let e = load_entry_mat4(in0, j, i);
      write_mat3_entry(out_byte, i, j, e.x + e.y);
    }
  }
}
`;
}

/** `matmulChain` of N mat4 inputs L0·L1·…·L(N-1), collapsed to f32. */
function emitMatMulChainArm(id: number, n: number): string {
  const params = Array.from({ length: n }, (_, i) => `in${i}: u32`).join(", ");
  // P holds the running product in df32, row-major (16 entries). Start P = L0, then P = P · Lk.
  let body = `  var P: array<vec2<f32>, 16>;\n`;
  body += `  for (var r: u32 = 0u; r < 4u; r = r + 1u) {\n`;
  body += `    for (var c: u32 = 0u; c < 4u; c = c + 1u) { P[r * 4u + c] = load_entry_mat4(in0, r, c); }\n`;
  body += `  }\n`;
  for (let k = 1; k < n; k++) {
    body += `  {\n`;
    body += `    var Q: array<vec2<f32>, 16>;\n`;
    body += `    for (var r: u32 = 0u; r < 4u; r = r + 1u) {\n`;
    body += `      for (var c: u32 = 0u; c < 4u; c = c + 1u) {\n`;
    body += `        var acc = vec2<f32>(0.0, 0.0);\n`;
    body += `        for (var t: u32 = 0u; t < 4u; t = t + 1u) {\n`;
    body += `          acc = df_add(acc, df_mul(P[r * 4u + t], load_entry_mat4(in${k}, t, c)));\n`;
    body += `        }\n`;
    body += `        Q[r * 4u + c] = acc;\n`;
    body += `      }\n`;
    body += `    }\n`;
    body += `    P = Q;\n`;
    body += `  }\n`;
  }
  body += `  for (var r: u32 = 0u; r < 4u; r = r + 1u) {\n`;
  body += `    for (var c: u32 = 0u; c < 4u; c = c + 1u) {\n`;
  body += `      let e = P[r * 4u + c];\n`;
  body += `      write_mat4_entry(out_byte, r, c, e.x + e.y);\n`;
  body += `    }\n`;
  body += `  }\n`;
  return `\nfn arm_${id}(${params}, out_byte: u32) {\n${body}}\n`;
}

// ─── Generic path: arbitrary IR lowered via the wgsl printer (f32) ────
//
// The rule's IR is rewritten so each input leaf — `ReadInput("Uniform", x)` or
// `Inverse(ReadInput("Uniform", x))` (the constituent's stored backward half, already
// inverted) — becomes a `Var("in<i>")`, then the wgsl `expr()` printer prints the body.
// Inputs/outputs are f32 (constituents are loaded collapsed); a leftover `Inverse` of a
// non-leaf throws (WGSL has no matrix inverse). Matrices are stored row-major in the heap
// and df32-row-major in the constituents, so the loader transposes into a WGSL (column-major)
// matrix value and the storer transposes back.

interface WgslT { readonly wgsl: string; readonly sym: string; readonly isMat: boolean; readonly dim: number }

function wgslTypeOf(t: Type, where: string): WgslT {
  switch (t.kind) {
    case "Float": return { wgsl: "f32", sym: "f32", isMat: false, dim: 1 };
    case "Bool": throw new Error(`derived rule (${where}): bool is not a heap-storable type`);
    case "Int": return t.signed
      ? { wgsl: "i32", sym: "i32", isMat: false, dim: 1 }
      : { wgsl: "u32", sym: "u32", isMat: false, dim: 1 };
    case "Vector": {
      if (t.element.kind !== "Float") throw new Error(`derived rule (${where}): only vecN<f32> supported`);
      return { wgsl: `vec${t.dim}<f32>`, sym: `vec${t.dim}_f32`, isMat: false, dim: t.dim };
    }
    case "Matrix": {
      if (t.element.kind !== "Float") throw new Error(`derived rule (${where}): only matNxM<f32> supported`);
      if (t.rows !== t.cols || (t.rows !== 3 && t.rows !== 4)) {
        throw new Error(`derived rule (${where}): only mat3x3<f32> / mat4x4<f32> supported`);
      }
      return { wgsl: `mat${t.cols}x${t.rows}<f32>`, sym: `mat${t.cols}x${t.rows}_f32`, isMat: true, dim: t.rows };
    }
    default: throw new Error(`derived rule (${where}): unsupported type kind '${t.kind}'`);
  }
}

/** `load_<sym>(h: u32) -> <wgsl>` — from a constituent slot (df32, collapsed) or the main heap. */
function emitLoader(t: WgslT): string {
  if (t.isMat) {
    const n = t.dim;
    return `
fn load_${t.sym}(h: u32) -> ${t.wgsl} {
  var m: ${t.wgsl};
  for (var r: u32 = 0u; r < ${n}u; r = r + 1u) {
    for (var c: u32 = 0u; c < ${n}u; c = c + 1u) {
      let e = load_entry_mat4(h, r, c);
      m[c][r] = e.x + e.y;
    }
  }
  return m;
}
`;
  }
  // Scalar / vector: from the main heap only (constituents are mat4 trafos).
  const b = "let b = slot_payload(h) >> 2u;";
  if (t.dim === 1) {
    const conv = t.wgsl === "f32" ? "MainHeap[b]" : `bitcast<${t.wgsl}>(MainHeap[b])`;
    return `\nfn load_${t.sym}(h: u32) -> ${t.wgsl} { ${b} return ${conv}; }\n`;
  }
  const comps = Array.from({ length: t.dim }, (_, i) => (i === 0 ? "MainHeap[b]" : `MainHeap[b + ${i}u]`)).join(", ");
  return `\nfn load_${t.sym}(h: u32) -> ${t.wgsl} { ${b} return ${t.wgsl}(${comps}); }\n`;
}

/** `store_<sym>(out_byte: u32, v: <wgsl>)` — into the main heap. */
function emitStorer(t: WgslT): string {
  if (t.isMat) {
    const n = t.dim;
    const writeFn = n === 3 ? "write_mat3_entry" : "write_mat4_entry";
    return `
fn store_${t.sym}(out_byte: u32, v: ${t.wgsl}) {
  for (var r: u32 = 0u; r < ${n}u; r = r + 1u) {
    for (var c: u32 = 0u; c < ${n}u; c = c + 1u) { ${writeFn}(out_byte, r, c, v[c][r]); }
  }
}
`;
  }
  const b = "let b = out_byte >> 2u;";
  if (t.dim === 1) {
    const v = t.wgsl === "f32" ? "v" : "bitcast<f32>(v)";
    return `\nfn store_${t.sym}(out_byte: u32, v: ${t.wgsl}) { ${b} MainHeap[b] = ${v}; }\n`;
  }
  const comp = ["x", "y", "z", "w"];
  const writes = Array.from({ length: t.dim }, (_, i) => `MainHeap[b + ${i}u] = v.${comp[i]};`).join(" ");
  return `\nfn store_${t.sym}(out_byte: u32, v: ${t.wgsl}) { ${b} ${writes} }\n`;
}

function varExpr(idx: number, type: Type): Expr {
  return { kind: "Var", var: { name: `in${idx}`, type, mutable: false }, type };
}

// ── Minimal IR → WGSL expression printer (self-contained; covers the subset a
//    derived-rule body can be). Throws on anything not lowerable to plain WGSL. ──

function wgslScalar(t: Type): string {
  switch (t.kind) {
    case "Float": return "f32";
    case "Bool": return "bool";
    case "Int": return t.signed ? "i32" : "u32";
    default: throw new Error(`derived rule: non-scalar type where scalar expected`);
  }
}
function wgslTypeName(t: Type): string {
  switch (t.kind) {
    case "Float": return "f32";
    case "Bool": return "bool";
    case "Int": return t.signed ? "i32" : "u32";
    case "Vector": return `vec${t.dim}<${wgslScalar(t.element)}>`;
    case "Matrix": return `mat${t.cols}x${t.rows}<${wgslScalar(t.element)}>`;
    default: throw new Error(`derived rule: unsupported type '${t.kind}' in rule body`);
  }
}
function wgslLiteral(l: Literal): string {
  switch (l.kind) {
    case "Bool": return l.value ? "true" : "false";
    case "Int": return l.signed ? `${l.value}i` : `${l.value >>> 0}u`;
    case "Float": return Number.isFinite(l.value) ? (Number.isInteger(l.value) ? `${l.value}.0` : `${l.value}`) : "0.0";
    case "Null": return "0";
  }
}
const BIN: Partial<Record<Expr["kind"], string>> = {
  Add: "+", Sub: "-", Mul: "*", Div: "/", Mod: "%",
  MulMatMat: "*", MulMatVec: "*", MulVecMat: "*",
  And: "&&", Or: "||", BitAnd: "&", BitOr: "|", BitXor: "^",
  Eq: "==", Neq: "!=", Lt: "<", Le: "<=", Gt: ">", Ge: ">=",
};
function printExpr(e: Expr): string {
  switch (e.kind) {
    case "Var": return e.var.name;
    case "Const": return wgslLiteral(e.value);
    case "Neg": return `(-${printExpr(e.value)})`;
    case "Not": return `(!${printExpr(e.value)})`;
    case "BitNot": return `(~${printExpr(e.value)})`;
    case "Add": case "Sub": case "Mul": case "Div": case "Mod":
    case "MulMatMat": case "MulMatVec": case "MulVecMat":
    case "And": case "Or": case "BitAnd": case "BitOr": case "BitXor":
    case "Eq": case "Neq": case "Lt": case "Le": case "Gt": case "Ge":
      return `(${printExpr(e.lhs)} ${BIN[e.kind]} ${printExpr(e.rhs)})`;
    case "ShiftLeft": case "ShiftRight": {
      const op = e.kind === "ShiftLeft" ? "<<" : ">>";
      const r = e.rhs.type.kind === "Int" && !e.rhs.type.signed ? printExpr(e.rhs) : `u32(${printExpr(e.rhs)})`;
      return `(${printExpr(e.lhs)} ${op} ${r})`;
    }
    case "Transpose": return `transpose(${printExpr(e.value)})`;
    case "Determinant": return `determinant(${printExpr(e.value)})`;
    case "Dot": return `dot(${printExpr(e.lhs)}, ${printExpr(e.rhs)})`;
    case "Cross": return `cross(${printExpr(e.lhs)}, ${printExpr(e.rhs)})`;
    case "Length": return `length(${printExpr(e.value)})`;
    case "VecSwizzle": return `${printExpr(e.value)}.${e.comps.join("")}`;
    case "VecItem": return `${printExpr(e.value)}[${printExpr(e.index)}]`;
    case "MatrixCol": return `${printExpr(e.matrix)}[${printExpr(e.col)}]`;
    case "MatrixElement": return `${printExpr(e.matrix)}[${printExpr(e.col)}][${printExpr(e.row)}]`;
    case "NewVector": return `${wgslTypeName(e.type)}(${e.components.map(printExpr).join(", ")})`;
    case "Conditional": return `select(${printExpr(e.ifFalse)}, ${printExpr(e.ifTrue)}, ${printExpr(e.cond)})`;
    case "Convert": case "ConvertMatrix": return `${wgslTypeName(e.type)}(${printExpr(e.value)})`;
    case "Field": return `${printExpr(e.target)}.${e.name}`;
    case "Item": return `${printExpr(e.target)}[${printExpr(e.index)}]`;
    case "CallIntrinsic": return `${e.op.emit.wgsl}(${e.args.map(printExpr).join(", ")})`;
    default:
      throw new Error(`derived rule: IR node '${e.kind}' is not supported in a rule body`);
  }
}

const INV_SUFFIX = "inv"; // names a synthetic "inverted constituent" leaf during rewriting

/** Rewrite `entry.ir` so input leaves become `Var("in<i>")`, then print the body via wgsl `expr()`. */
function emitGenericRuleFn(entry: RuleEntry): string {
  const byName = new Map<string, number>();
  entry.inputs.forEach((inp, i) => byName.set(inp.inverse ? inp.name + INV_SUFFIX : inp.name, i));
  // Pass 1: collapse `Inverse(ReadInput("Uniform", x))` → `ReadInput("Uniform", xinv)`.
  const collapsed = mapExpr(entry.ir, (e) =>
    e.kind === "Inverse" && e.value.kind === "ReadInput" && e.value.scope === "Uniform"
      ? ({ kind: "ReadInput", scope: "Uniform", name: e.value.name + INV_SUFFIX, type: e.type } as Expr)
      : e,
  );
  // Pass 2: every `ReadInput("Uniform", n)` → `Var("in<i>")`.
  const body = mapExpr(collapsed, (e) => {
    if (e.kind === "ReadInput" && e.scope === "Uniform") {
      const idx = byName.get(e.name);
      if (idx === undefined) throw new Error(`derived rule ${entry.id}: input leaf '${e.name}' not in the rule's input list`);
      return varExpr(idx, e.type);
    }
    return e;
  });
  // Anything not lowerable to plain WGSL?
  const walk = (e: Expr): void => {
    if (e.kind === "Inverse") {
      throw new Error(`derived rule ${entry.id}: a matrix inverse in the rule body is not supported (WGSL has no inverse — only Inverse of a constituent input is, which reads its stored backward half)`);
    }
    if (e.kind === "ReadInput") throw new Error(`derived rule ${entry.id}: unresolved input leaf '${e.name}'`);
    visitExprChildren(e, walk);
  };
  walk(body);
  const params = entry.inputs.map((inp, i) => `in${i}: ${wgslTypeOf(inp.type, `rule ${entry.id} input ${i}`).wgsl}`).join(", ");
  const ret = wgslTypeOf(entry.outputType, `rule ${entry.id} output`).wgsl;
  return `\nfn rule_${entry.id}(${params}) -> ${ret} {\n  return ${printExpr(body)};\n}\n`;
}

function emitGenericArm(entry: RuleEntry): string {
  const inTs = entry.inputs.map((inp, i) => wgslTypeOf(inp.type, `rule ${entry.id} input ${i}`));
  const outT = wgslTypeOf(entry.outputType, `rule ${entry.id} output`);
  const params = inTs.map((_, i) => `in${i}: u32`).join(", ");
  let body = "";
  inTs.forEach((t, i) => { body += `  let a${i} = load_${t.sym}(in${i});\n`; });
  const callArgs = inTs.map((_, i) => `a${i}`).join(", ");
  body += `  store_${outT.sym}(out_byte, rule_${entry.id}(${callArgs}));\n`;
  return `${emitGenericRuleFn(entry)}\nfn arm_${entry.id}(${params}, out_byte: u32) {\n${body}}\n`;
}

function genericTypes(entry: RuleEntry): WgslT[] {
  return [...entry.inputs.map((inp, i) => wgslTypeOf(inp.type, `rule ${entry.id} input ${i}`)), wgslTypeOf(entry.outputType, `rule ${entry.id} output`)];
}

function emitArm(entry: RuleEntry, shape: Shape): string {
  switch (shape.kind) {
    case "collapse": return emitCollapseArm(entry.id);
    case "normalMatrix": return emitNormalMatrixArm(entry.id);
    case "matmulChain": return emitMatMulChainArm(entry.id, shape.arity);
    case "generic": return emitGenericArm(entry);
  }
}

function emitCase(entry: RuleEntry, shape: Shape): string {
  const ins = Array.from({ length: shape.arity }, (_, i) => `RecordData[base + ${2 + i}u]`).join(", ");
  const args = shape.arity === 0 ? "out_byte" : `${ins}, out_byte`;
  return `    case ${entry.id >>> 0}u: { arm_${entry.id}(${args}); }`;
}

// ─── Top-level ────────────────────────────────────────────────────────

export function buildUberKernel(registry: DerivedUniformRegistry, strideU32: number): UberKernel {
  const entries = registry.entries();
  const shapes = entries.map((e) => classify(e));

  // Loaders / storers needed by the generic-path rules, deduped by WGSL type symbol.
  const ioTypes = new Map<string, WgslT>();
  entries.forEach((e, i) => {
    if (shapes[i]!.kind === "generic") for (const t of genericTypes(e)) ioTypes.set(t.sym, t);
  });
  const loadersStorers = [...ioTypes.values()].map((t) => emitLoader(t) + emitStorer(t)).join("");

  const arms = entries.map((e, i) => emitArm(e, shapes[i]!)).join("");
  const cases = entries.map((e, i) => emitCase(e, shapes[i]!)).join("\n");
  const needsDf32 = shapes.some((s) => s.kind === "matmulChain");

  const wgsl = `${bindings(strideU32)}
${needsDf32 ? DF32_LIB : ""}
${HANDLE_HELPERS}
${loadersStorers}
${arms}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= Count.count) { return; }
  let base = gid.x * RECORD_STRIDE;
  let rule_id = RecordData[base];
  let out_byte = RecordData[base + 1u] & SLOT_PAYLOAD_MASK;
  switch rule_id {
${cases}
    default: { return; }
  }
}
`;
  return { wgsl, strideU32 };
}
