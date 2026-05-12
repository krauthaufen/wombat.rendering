// §7 v2 — uber-kernel codegen.
//
// Emits the single compute kernel from the rule registry. One thread per
// record; each thread switches on `rule_id` and runs the matching arm:
// load the inputs (df32 from a constituent slot, or plain f32 from the
// drawHeader / globals), do the math, write the result to MainHeap.
//
// v0 supports the rule shapes the standard trafo recipes need:
//   • a single mat4 input, output mat4  → collapse
//   • a left-fold MulMatMat over N mat4 inputs, output mat4 → dfmulN + collapse
//   • a single mat4 input, output mat3  → upper-3×3 of the transposed input (normal matrix)
// Arbitrary IR rules (the wgsl `expr()` lowering path) land in a follow-up;
// codegen throws on a shape it doesn't recognise. See docs/derived-uniforms-extensible.md.

import type { Expr } from "@aardworx/wombat.shader/ir";
import type { DerivedUniformRegistry, RuleEntry } from "./registry.js";

export interface UberKernel {
  readonly wgsl: string;
  /** Record stride (u32 words) the kernel was generated for — must match RecordsBuffer.strideWords. */
  readonly strideU32: number;
}

// ─── Rule-shape classification ────────────────────────────────────────

type Shape =
  | { kind: "collapse"; arity: 1 }                 // mat4 in → mat4 out
  | { kind: "matmulChain"; arity: number }         // L1·…·LN (all mat4) → mat4 out
  | { kind: "normalMatrix"; arity: 1 };            // mat4 in → mat3 out (upper-3×3 of transpose)

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
  if (isMat3(entry.outputType)) {
    // Only mat4→mat3 rule we support: the normal matrix (upper-3×3 of the transposed input).
    if (entry.inputs.length === 1) return { kind: "normalMatrix", arity: 1 };
    throw new Error(`derived rule ${entry.id}: unsupported mat3 output shape`);
  }
  const chain = matmulChainLeaves(ir);
  if (chain !== undefined) {
    if (chain.length === 1) return { kind: "collapse", arity: 1 };
    // The arm consumes inputs positionally in chain order; that matches `entry.inputs`
    // (first-appearance order) only when no leaf is repeated.
    if (chain.length !== entry.inputs.length) {
      throw new Error(`derived rule ${entry.id}: a repeated input in a matmul chain is not supported in v0`);
    }
    return { kind: "matmulChain", arity: chain.length };
  }
  throw new Error(
    `derived rule ${entry.id}: unsupported IR shape (v0 supports constituent matmul chains and the normal matrix; ` +
      `arbitrary-expression rules land in a follow-up)`,
  );
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

function emitArm(entry: RuleEntry, shape: Shape): string {
  switch (shape.kind) {
    case "collapse": return emitCollapseArm(entry.id);
    case "normalMatrix": return emitNormalMatrixArm(entry.id);
    case "matmulChain": return emitMatMulChainArm(entry.id, shape.arity);
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

  const arms = entries.map((e, i) => emitArm(e, shapes[i]!)).join("");
  const cases = entries.map((e, i) => emitCase(e, shapes[i]!)).join("\n");
  const needsDf32 = shapes.some((s) => s.kind === "matmulChain");

  const wgsl = `${bindings(strideU32)}
${needsDf32 ? DF32_LIB : ""}
${HANDLE_HELPERS}
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
