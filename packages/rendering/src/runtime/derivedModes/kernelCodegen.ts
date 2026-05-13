// Codegen the partition kernel from a derived-mode rule's shader IR.
//
// Input: one or more `RuleSpec` entries (axis + rule body Stmt +
// `declared` substitution value + intrinsic eval table + the
// resolved output set under that declared). Output: a WGSL kernel
// string with `clear` + `partitionRecords` entry points.
//
// The body of each axis-rule's evaluator function is emitted by
// walking the shader IR's Stmt/Expr tree directly (no shader
// `compileModule` call — we want a tight self-contained kernel,
// not a full pipeline). `declared` reads collapse to a Const at
// codegen time (one kernel per declared value, cached by
// `(axis, declaredU32, ruleHash)` upstream).
//
// Resolved-set → slot index: the rule body returns an axis-enum u32
// (e.g. CullMode 0/1/2). The bucket's slot ordering follows the
// resolved set sorted ascending. We emit a small inline lookup
// (`SLOT_LOOKUP_<axis>: array<u32, MAX_ENUM>`) so the kernel can
// translate "rule output 1" → "slot 0", "rule output 2" → "slot 1",
// etc. in O(1).

import type { Expr, Stmt, Type, Literal, IntrinsicEvalTable } from "@aardworx/wombat.shader/ir";
import { stableStringify } from "@aardworx/wombat.shader/ir";
import type { ModeAxis } from "./rule.js";

/** One traced+rewritten axis-rule body, ready for codegen. */
export interface RuleCodegenSpec {
  readonly axis: ModeAxis;
  /** Unique within `axis` within the bucket. */
  readonly axisRuleId: number;
  /**
   * The rule body — already rewritten so every `ReturnValue` returns
   * a `Const u32` PER-AXIS index into this axis's union value table
   * (`axisCardinality` below). The codegen never sees the original
   * `__record:*` intrinsics.
   */
  readonly body: Stmt;
  readonly intrinsics: IntrinsicEvalTable;
  /** Per-rule WGSL helpers (concatenated into the kernel). */
  readonly helpersWGSL: string;
  readonly inputUniforms: ReadonlyArray<string>;
}

/** Per-active-axis entry inside a single combo: either picks a rule
 *  (axis index computed at runtime by `rule_<axis>_<id>`) or a fixed
 *  per-axis index (a compile-time constant for axes the combo has no
 *  rule on — the index of baseDescriptor's value in this axis's
 *  union table). */
export type ComboAxisSource =
  | { readonly kind: "rule"; readonly axisRuleId: number }
  | { readonly kind: "const"; readonly idx: number };

export interface ComboAxis {
  readonly axis: ModeAxis;
  /** |axisValues[axis]| — the stride for mixed-radix encoding is the
   *  product of cardinalities of axes after this one in `axes`. */
  readonly cardinality: number;
  readonly source: ComboAxisSource;
}

export interface ComboCodegenSpec {
  readonly comboId: number;
  /** Active axes (those with at least one rule in the bucket), in
   *  canonical order. Every combo in the bucket carries the same
   *  axis list — combos differ only in which axes have rules. */
  readonly axes: ReadonlyArray<ComboAxis>;
}

export interface KernelCodegenSpec {
  /** Distinct axis-rules registered on the bucket. */
  readonly rules: ReadonlyArray<RuleCodegenSpec>;
  /** Distinct combos appearing on ROs in this bucket. */
  readonly combos: ReadonlyArray<ComboCodegenSpec>;
  /** Total slot count = ∏ cardinality across active axes. */
  readonly totalSlots: number;
}

/** Emit the WGSL partition kernel for the given rules + combos.
 *  Per-record dispatch:
 *    `switch (r.comboId) { case 0u: { return combo_0(r); } … }`
 *  Each `combo_<id>(r)` computes per-axis indices (calling
 *  `rule_<axis>_<axisRuleId>(r)` for axes the combo has a rule on,
 *  or using a const for the others) and composes them into a
 *  global slot index via mixed-radix encoding over `axes`. */
export function emitPartitionKernel(spec: KernelCodegenSpec): string {
  if (spec.combos.length === 0) {
    throw new Error("kernelCodegen: at least one combo is required");
  }
  if (spec.totalSlots > 16) {
    throw new Error(`kernelCodegen: totalSlots ${spec.totalSlots} exceeds the v1 limit of 16`);
  }
  const N = spec.totalSlots;

  // Emit one `rule_<axis>_<axisRuleId>(r) -> u32` per axis-rule.
  // Helpers are deduped by string equality so two rules sharing the
  // same WGSL helper don't double-emit (rule_<axis>_X and
  // rule_<axis>_Y can both use a user-defined `flipCull` etc. that
  // the shader frontend lifted into module scope).
  const helperSet = new Set<string>();
  const ruleFns: string[] = [];
  for (const r of spec.rules) {
    if (r.helpersWGSL.trim().length > 0) helperSet.add(r.helpersWGSL);
    ruleFns.push(emitStmtAsWgslFn(r));
  }
  const helpersWGSL = [...helperSet].join("\n");
  const ruleFnsWGSL = ruleFns.join("\n");

  // Emit one combo fn per combo: per-axis index via rule call or
  // const, then mixed-radix encode into the global slot index.
  const comboFns: string[] = [];
  const dispatchCases: string[] = [];
  for (const c of spec.combos) {
    comboFns.push(emitComboFn(c));
    dispatchCases.push(`    case ${c.comboId}u: { return combo_${c.comboId}(r); }`);
  }
  const comboFnsWGSL = comboFns.join("\n");
  const dispatcherWGSL = `
fn dispatch(r: Record) -> u32 {
  switch (r.comboId) {
${dispatchCases.join("\n")}
    default: { return ${N}u; }
  }
}
`;

  const slotCases = new Array(N).fill(0).map((_, slotIdx) =>
    `    case ${slotIdx}u: { let off = atomicAdd(&slot${slotIdx}Count[0], 1u); let base = off * SCAN_REC_U32;` +
    ` slot${slotIdx}DrawTable[base + 0u] = 0u;` +
    ` slot${slotIdx}DrawTable[base + 1u] = r.drawIdx;` +
    ` slot${slotIdx}DrawTable[base + 2u] = r.indexStart;` +
    ` slot${slotIdx}DrawTable[base + 3u] = r.indexCount;` +
    ` slot${slotIdx}DrawTable[base + 4u] = r.instanceCount; }`,
  ).join("\n");

  const slotCountClears = new Array(N).fill(0).map((_, i) =>
    `  atomicStore(&slot${i}Count[0], 0u);`,
  ).join("\n");

  let nextBinding = 3;
  const countBindings = new Array(N).fill(0).map((_, i) =>
    `@group(0) @binding(${nextBinding + i}) var<storage, read_write> slot${i}Count: array<atomic<u32>>;`,
  ).join("\n");
  nextBinding += N;
  const drawBindings = new Array(N).fill(0).map((_, i) =>
    `@group(0) @binding(${nextBinding + i}) var<storage, read_write> slot${i}DrawTable: array<u32>;`,
  ).join("\n");

  return `
struct Record {
  firstEmit:     u32,
  drawIdx:       u32,
  indexStart:    u32,
  indexCount:    u32,
  instanceCount: u32,
  modelRef:      u32,
  comboId:       u32,
};
struct PartitionParams { numRecords: u32 };

@group(0) @binding(0) var<storage, read>       arena:          array<u32>;
@group(0) @binding(1) var<storage, read>       masterRecords:  array<Record>;
@group(0) @binding(2) var<uniform>             params:         PartitionParams;
${countBindings}
${drawBindings}

const SCAN_REC_U32: u32 = 5u;

${LOADERS}

${helpersWGSL}

${ruleFnsWGSL}

${comboFnsWGSL}

${dispatcherWGSL}

@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) { return; }
${slotCountClears}
}

@compute @workgroup_size(64)
fn partitionRecords(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.numRecords) { return; }
  let r = masterRecords[i];
  let slotIdx = dispatch(r);
  if (slotIdx >= ${N}u) { return; }
  switch (slotIdx) {
${slotCases}
    default: { }
  }
}
`;
}

/** Emit one combo fn: per-axis index via rule call or const, then
 *  mixed-radix encode into a global slot index. Strides are
 *  precomputed (product of cardinalities of axes *after* the
 *  current one in `axes`). */
function emitComboFn(c: ComboCodegenSpec): string {
  if (c.axes.length === 0) {
    // Trivial combo (no active axes): always slot 0.
    return `fn combo_${c.comboId}(r: Record) -> u32 {\n  return 0u;\n}\n`;
  }
  // Mixed-radix strides: stride[i] = ∏_{j>i} cardinality[j].
  const strides: number[] = new Array(c.axes.length).fill(1);
  for (let i = c.axes.length - 2; i >= 0; i--) {
    strides[i] = strides[i + 1]! * c.axes[i + 1]!.cardinality;
  }
  const idxLines: string[] = [];
  const sumTerms: string[] = [];
  for (let i = 0; i < c.axes.length; i++) {
    const a = c.axes[i]!;
    const varName = `i_${a.axis}`;
    if (a.source.kind === "rule") {
      idxLines.push(`  let ${varName} = rule_${a.axis}_${a.source.axisRuleId}(r);`);
    } else {
      idxLines.push(`  let ${varName} = ${a.source.idx >>> 0}u;`);
    }
    sumTerms.push(strides[i] === 1 ? `${varName}` : `${varName} * ${strides[i]}u`);
  }
  return `fn combo_${c.comboId}(r: Record) -> u32 {\n${idxLines.join("\n")}\n  return ${sumTerms.join(" + ")};\n}\n`;
}

const LOADERS = `
fn load_mat3_upper(refBytes: u32) -> mat3x3<f32> {
  let baseU32 = (refBytes + 16u) >> 2u;
  return mat3x3<f32>(
    vec3<f32>(bitcast<f32>(arena[baseU32 +  0u]), bitcast<f32>(arena[baseU32 +  4u]), bitcast<f32>(arena[baseU32 +  8u])),
    vec3<f32>(bitcast<f32>(arena[baseU32 +  1u]), bitcast<f32>(arena[baseU32 +  5u]), bitcast<f32>(arena[baseU32 +  9u])),
    vec3<f32>(bitcast<f32>(arena[baseU32 +  2u]), bitcast<f32>(arena[baseU32 +  6u]), bitcast<f32>(arena[baseU32 + 10u])),
  );
}
fn load_mat4(refBytes: u32) -> mat4x4<f32> {
  let baseU32 = (refBytes + 16u) >> 2u;
  return mat4x4<f32>(
    vec4<f32>(bitcast<f32>(arena[baseU32 +  0u]), bitcast<f32>(arena[baseU32 +  4u]), bitcast<f32>(arena[baseU32 +  8u]), bitcast<f32>(arena[baseU32 + 12u])),
    vec4<f32>(bitcast<f32>(arena[baseU32 +  1u]), bitcast<f32>(arena[baseU32 +  5u]), bitcast<f32>(arena[baseU32 +  9u]), bitcast<f32>(arena[baseU32 + 13u])),
    vec4<f32>(bitcast<f32>(arena[baseU32 +  2u]), bitcast<f32>(arena[baseU32 +  6u]), bitcast<f32>(arena[baseU32 + 10u]), bitcast<f32>(arena[baseU32 + 14u])),
    vec4<f32>(bitcast<f32>(arena[baseU32 +  3u]), bitcast<f32>(arena[baseU32 +  7u]), bitcast<f32>(arena[baseU32 + 11u]), bitcast<f32>(arena[baseU32 + 15u])),
  );
}
`;

// ─── Stmt + Expr → WGSL emitter (covers the shader-IR subset rule
//     bodies can express: ReturnValue / If / Sequential / Declare /
//     Write / Expression / For / While, plus the full Expr lattice). ──

function emitStmtAsWgslFn(rule: RuleCodegenSpec): string {
  const ctx: EmitCtx = {
    axis: rule.axis,
    inputUniforms: rule.inputUniforms,
    localCounter: 0,
  };
  const bodyWgsl = emitStmt(rule.body, ctx, "  ");
  return `fn rule_${rule.axis}_${rule.axisRuleId}(r: Record) -> u32 {\n${bodyWgsl}\n  return 0u;\n}\n`;
}

interface EmitCtx {
  readonly axis: ModeAxis;
  readonly inputUniforms: ReadonlyArray<string>;
  localCounter: number;
}

// ─────────────────────────────────────────────────────────────────────
// IR rewrite helpers (heap scene calls these before emitPartitionKernel)
// ─────────────────────────────────────────────────────────────────────

/**
 * Replace every `ReadInput("<scope>", "<name>")` leaf in `e` with
 * the supplied `replacement` Expr. Used to bake the rule's
 * `declared` ReadInput as a `Const u32` of the current declared
 * value before codegen — the resulting body has no `declared`
 * references, so the kernel doesn't need a per-dispatch uniform.
 */
export function substituteReadInput(
  e: Expr, scope: string, name: string, replacement: Expr,
): Expr {
  return mapExpr(e, (n) => {
    const r = n as { kind: string; scope?: string; name?: string };
    if (r.kind === "ReadInput" && r.scope === scope && r.name === name) {
      return replacement;
    }
    return n;
  });
}
export function substituteReadInputInStmt(
  s: Stmt, scope: string, name: string, replacement: Expr,
): Stmt {
  return mapStmtExprs(s, (e) => substituteReadInput(e, scope, name, replacement));
}

/**
 * Rewrite the body so every `ReturnValue` returns a `Const u32` slot
 * index (resolved by `exprToSlot`). `Conditional` expressions are
 * recursed into — each branch is rewritten independently. Other
 * non-Conditional return expressions are matched as a whole against
 * `exprToSlot` (using stableStringify-content equality upstream).
 *
 * After this pass the kernel emits a u32 directly from
 * `rule_<axis>(r)`; the partition entry switches on it as a slot
 * index, no SLOT_LOOKUP table needed.
 */
export function rewriteOutputsToSlotIndices(
  body: Stmt,
  exprToSlot: (e: Expr) => number | undefined,
): Stmt {
  const Tu32: Type = { kind: "Int", signed: false, width: 32 };
  const constU32 = (v: number): Expr =>
    ({ kind: "Const", type: Tu32, value: { kind: "Int", value: v >>> 0, signed: false } } as Expr);
  const rewriteOutput = (e: Expr): Expr => {
    const k = (e as { kind: string }).kind;
    // Try the whole expression first — `exprToSlot` deals in
    // stableStringify-content matches, so a leaf Expr that
    // appeared in the symbolic output set gets replaced as a
    // unit.
    const direct = exprToSlot(e);
    if (direct !== undefined) return constU32(direct);
    // Conditional: rewrite each branch independently. The
    // condition stays as-is (it's a per-record routing predicate).
    if (k === "Conditional") {
      const c = e as { cond: Expr; ifTrue: Expr; ifFalse: Expr };
      return {
        kind: "Conditional",
        cond: c.cond,
        ifTrue: rewriteOutput(c.ifTrue),
        ifFalse: rewriteOutput(c.ifFalse),
        type: Tu32,
      } as Expr;
    }
    return e;
  };
  const rewriteStmt = (s: Stmt): Stmt => {
    const k = (s as { kind: string }).kind;
    if (k === "ReturnValue") {
      const r = s as { value: Expr };
      return { kind: "ReturnValue", value: rewriteOutput(r.value) } as Stmt;
    }
    // Recurse into structured statements.
    if (k === "Sequential" || k === "Isolated") {
      const b = (s as { body: ReadonlyArray<Stmt> }).body;
      return { kind: k, body: b.map(rewriteStmt) } as Stmt;
    }
    if (k === "If") {
      const i = s as { cond: Expr; then: Stmt; else?: Stmt };
      return {
        kind: "If",
        cond: i.cond,
        then: rewriteStmt(i.then),
        ...(i.else !== undefined ? { else: rewriteStmt(i.else) } : {}),
      } as Stmt;
    }
    if (k === "For" || k === "While" || k === "DoWhile" || k === "Loop") {
      const w = s as { cond?: Expr; body: Stmt; init?: Stmt; step?: Stmt };
      return { ...(s as object), body: rewriteStmt(w.body) } as Stmt;
    }
    return s;
  };
  return rewriteStmt(body);
}

// Generic Expr/Stmt mapper used by the substitution helpers above.
function mapExpr(e: Expr, fn: (e: Expr) => Expr): Expr {
  const mapped = fn(e);
  if (mapped !== e) return mapped;
  const obj = e as unknown as Record<string, unknown>;
  let changed = false;
  const out: Record<string, unknown> = { ...obj };
  for (const k of Object.keys(out)) {
    const v = out[k];
    if (v !== null && typeof v === "object") {
      if ("kind" in (v as object) && !Array.isArray(v)) {
        const sub = mapExpr(v as Expr, fn);
        if (sub !== v) { out[k] = sub; changed = true; }
      } else if (Array.isArray(v)) {
        let ac = false;
        const m = v.map((c: unknown) => {
          if (c !== null && typeof c === "object" && "kind" in (c as object)) {
            const sub = mapExpr(c as Expr, fn);
            if (sub !== c) ac = true;
            return sub;
          }
          return c;
        });
        if (ac) { out[k] = m; changed = true; }
      }
    }
  }
  return changed ? (out as unknown as Expr) : e;
}
function mapStmtExprs(s: Stmt, fn: (e: Expr) => Expr): Stmt {
  const obj = s as unknown as Record<string, unknown>;
  let changed = false;
  const out: Record<string, unknown> = { ...obj };
  for (const k of Object.keys(out)) {
    const v = out[k];
    if (v !== null && typeof v === "object") {
      if ("kind" in (v as object) && !Array.isArray(v)) {
        const isStmt = /^(Nop|Expression|Declare|Write|Sequential|Isolated|Return|ReturnValue|Break|Continue|If|For|While|DoWhile|Loop|Switch|Discard|Barrier|WriteOutput|Increment|Decrement)$/.test((v as { kind: string }).kind);
        const sub = isStmt ? mapStmtExprs(v as Stmt, fn) : mapExpr(v as Expr, fn);
        if (sub !== v) { out[k] = sub; changed = true; }
      } else if (Array.isArray(v)) {
        let ac = false;
        const m = v.map((c: unknown) => {
          if (c !== null && typeof c === "object" && "kind" in (c as object)) {
            const kind = (c as { kind: string }).kind;
            const isStmt = /^(Nop|Expression|Declare|Write|Sequential|Isolated|Return|ReturnValue|Break|Continue|If|For|While|DoWhile|Loop|Switch|Discard|Barrier|WriteOutput|Increment|Decrement)$/.test(kind);
            const sub = isStmt ? mapStmtExprs(c as Stmt, fn) : mapExpr(c as Expr, fn);
            if (sub !== c) ac = true;
            return sub;
          }
          return c;
        });
        if (ac) { out[k] = m; changed = true; }
      }
    }
  }
  return changed ? (out as unknown as Stmt) : s;
}

function emitStmt(s: Stmt, ctx: EmitCtx, indent: string): string {
  const k = (s as { kind: string }).kind;
  switch (k) {
    case "ReturnValue": {
      const v = (s as { value: Expr }).value;
      return `${indent}return ${emitExpr(v, ctx)};`;
    }
    case "Return":
      return `${indent}return 0u;`;
    case "Sequential":
    case "Isolated": {
      const body = (s as { body: ReadonlyArray<Stmt> }).body;
      return body.map(c => emitStmt(c, ctx, indent)).join("\n");
    }
    case "Declare": {
      const d = s as unknown as { var: { name: string; type: Type; mutable: boolean }; init?: Expr };
      const kw = d.var.mutable ? "var" : "let";
      const ty = wgslTypeName(d.var.type);
      if (d.init === undefined) return `${indent}${kw} ${d.var.name}: ${ty};`;
      return `${indent}${kw} ${d.var.name}: ${ty} = ${emitExpr(d.init as Expr, ctx)};`;
    }
    case "Write": {
      const w = s as unknown as { target: Expr; value: Expr };
      return `${indent}${emitExpr(w.target, ctx)} = ${emitExpr(w.value, ctx)};`;
    }
    case "Expression": {
      const e = s as { value: Expr };
      return `${indent}${emitExpr(e.value, ctx)};`;
    }
    case "If": {
      const i = s as { cond: Expr; then: Stmt; else?: Stmt };
      const thenWgsl = emitStmt(i.then, ctx, indent + "  ");
      const elseWgsl = i.else !== undefined ? emitStmt(i.else, ctx, indent + "  ") : undefined;
      return elseWgsl !== undefined
        ? `${indent}if (${emitExpr(i.cond, ctx)}) {\n${thenWgsl}\n${indent}} else {\n${elseWgsl}\n${indent}}`
        : `${indent}if (${emitExpr(i.cond, ctx)}) {\n${thenWgsl}\n${indent}}`;
    }
    case "For": {
      const f = s as { init: Stmt; cond: Expr; step: Stmt; body: Stmt };
      const init = emitStmt(f.init, ctx, "").trim().replace(/;$/, "");
      const cond = emitExpr(f.cond, ctx);
      const step = emitStmt(f.step, ctx, "").trim().replace(/;$/, "");
      const body = emitStmt(f.body, ctx, indent + "  ");
      return `${indent}for (${init}; ${cond}; ${step}) {\n${body}\n${indent}}`;
    }
    case "While": {
      const w = s as { cond: Expr; body: Stmt };
      return `${indent}while (${emitExpr(w.cond, ctx)}) {\n${emitStmt(w.body, ctx, indent + "  ")}\n${indent}}`;
    }
    case "Break":    return `${indent}break;`;
    case "Continue": return `${indent}continue;`;
    case "Nop":      return "";
    default:
      throw new Error(`kernelCodegen: unsupported Stmt kind '${k}' in rule body`);
  }
}

function emitExpr(e: Expr, ctx: EmitCtx): string {
  const k = (e as { kind: string }).kind;
  switch (k) {
    case "Const":  return wgslLiteral((e as { value: Literal }).value);
    case "Var":    return (e as { var: { name: string } }).var.name;
    case "ReadInput": {
      const r = e as { scope: string; name: string; type: Type };
      void ctx;
      // `declared` is pre-substituted by heapScene before codegen,
      // so any surviving ReadInput at this point is a `u.<X>` read
      // from arena via the record's modelRef.
      if (r.type.kind === "Matrix" && r.type.rows === 3 && r.type.cols === 3) {
        return `load_mat3_upper(r.modelRef)`;
      }
      if (r.type.kind === "Matrix" && r.type.rows === 4 && r.type.cols === 4) {
        return `load_mat4(r.modelRef)`;
      }
      throw new Error(
        `kernelCodegen: rule body reads uniform '${r.name}' of unsupported arena type ${JSON.stringify(r.type)}`,
      );
    }
    case "Neg":  return `(-${emitExpr((e as { value: Expr }).value, ctx)})`;
    case "Not":  return `(!${emitExpr((e as { value: Expr }).value, ctx)})`;
    case "Add": case "Sub": case "Mul": case "Div": case "Mod":
    case "MulMatMat": case "MulMatVec": case "MulVecMat":
    case "And": case "Or": case "BitAnd": case "BitOr": case "BitXor":
    case "Eq": case "Neq": case "Lt": case "Le": case "Gt": case "Ge": {
      const b = e as { lhs: Expr; rhs: Expr };
      return `(${emitExpr(b.lhs, ctx)} ${BIN[k]} ${emitExpr(b.rhs, ctx)})`;
    }
    case "Transpose":   return `transpose(${emitExpr((e as { value: Expr }).value, ctx)})`;
    case "Determinant": return `determinant(${emitExpr((e as { value: Expr }).value, ctx)})`;
    case "Conditional": {
      const c = e as { cond: Expr; ifTrue: Expr; ifFalse: Expr };
      return `select(${emitExpr(c.ifFalse, ctx)}, ${emitExpr(c.ifTrue, ctx)}, ${emitExpr(c.cond, ctx)})`;
    }
    case "Convert":
    case "ConvertMatrix":
      return `${wgslTypeName(e.type)}(${emitExpr((e as { value: Expr }).value, ctx)})`;
    case "CallIntrinsic": {
      const ci = e as { op: { name: string; emit: { wgsl: string } }; args: ReadonlyArray<Expr> };
      const fnName = ci.op.emit.wgsl ?? ci.op.name;
      return `${fnName}(${ci.args.map(a => emitExpr(a, ctx)).join(", ")})`;
    }
    case "VecSwizzle": {
      const v = e as { value: Expr; comps: ReadonlyArray<string> };
      return `${emitExpr(v.value, ctx)}.${v.comps.join("")}`;
    }
    case "VecItem":      return `${emitExpr((e as { value: Expr }).value, ctx)}[${emitExpr((e as { index: Expr }).index, ctx)}]`;
    case "MatrixCol":    return `${emitExpr((e as { matrix: Expr }).matrix, ctx)}[${emitExpr((e as { col: Expr }).col, ctx)}]`;
    case "MatrixElement":
      return `${emitExpr((e as { matrix: Expr }).matrix, ctx)}[${emitExpr((e as { col: Expr }).col, ctx)}][${emitExpr((e as { row: Expr }).row, ctx)}]`;
    case "NewVector":
      return `${wgslTypeName(e.type)}(${(e as { components: ReadonlyArray<Expr> }).components.map(a => emitExpr(a, ctx)).join(", ")})`;
    default:
      throw new Error(`kernelCodegen: unsupported Expr kind '${k}'`);
  }
}

const BIN: Record<string, string> = {
  Add: "+", Sub: "-", Mul: "*", Div: "/", Mod: "%",
  MulMatMat: "*", MulMatVec: "*", MulVecMat: "*",
  And: "&&", Or: "||", BitAnd: "&", BitOr: "|", BitXor: "^",
  Eq: "==", Neq: "!=", Lt: "<", Le: "<=", Gt: ">", Ge: ">=",
};

function wgslLiteral(l: Literal): string {
  switch (l.kind) {
    case "Bool":  return l.value ? "true" : "false";
    case "Int":   return l.signed ? `${l.value}i` : `${l.value >>> 0}u`;
    case "Float": return Number.isFinite(l.value)
      ? (Number.isInteger(l.value) ? `${l.value}.0` : `${l.value}`)
      : "0.0";
    case "Null":  return "0";
  }
}
function wgslScalar(t: Type): string {
  switch (t.kind) {
    case "Float": return "f32";
    case "Bool":  return "bool";
    case "Int":   return t.signed ? "i32" : "u32";
    default: throw new Error("kernelCodegen: non-scalar type where scalar expected");
  }
}
function wgslTypeName(t: Type): string {
  switch (t.kind) {
    case "Float":  return "f32";
    case "Bool":   return "bool";
    case "Int":    return t.signed ? "i32" : "u32";
    case "Vector": return `vec${t.dim}<${wgslScalar(t.element)}>`;
    case "Matrix": return `mat${t.cols}x${t.rows}<${wgslScalar(t.element)}>`;
    default: throw new Error(`kernelCodegen: unsupported type '${t.kind}'`);
  }
}
