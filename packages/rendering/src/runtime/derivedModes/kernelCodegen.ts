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
import type { ModeAxis } from "./rule.js";

/** Per-axis rule spec consumed by the codegen. */
export interface RuleCodegenSpec {
  readonly axis: ModeAxis;
  /** The rule body (`RuleExpr.template.values[entry].entry.body`). */
  readonly body: Stmt;
  /** Current declared value substituted as a Const before codegen. */
  readonly declaredU32: number;
  /**
   * The resolved set (output of `evaluateSet` for the rule's symbolic
   * outputs under the current declared), sorted ascending. Drives
   * slot count, slot enum ordering, and the SLOT_LOOKUP table.
   */
  readonly resolved: ReadonlyArray<number>;
  /**
   * Axis-specific intrinsic table. The codegen emits direct
   * `intrinsicName(args)` calls in WGSL — the caller must ensure
   * the intrinsic is available in WGSL OR provide it as a helper
   * (see `helperWGSL`). The eval table is used only for sanity
   * checks during pre-emission constant folding.
   */
  readonly intrinsics: IntrinsicEvalTable;
  /**
   * Pre-emitted WGSL helper functions referenced by the rule body
   * (e.g. `fn flipCull(c: u32) -> u32 { ... }`). One block per axis;
   * concatenated into the kernel.
   */
  readonly helpersWGSL: string;
  /** Names of `u.<X>` uniform leaves the body reads, in declaration
   *  order. Maps to per-RO arena byte offsets at codegen time —
   *  currently fixed to one input (modelRef) until the dispatcher
   *  generalises to multi-input rules. */
  readonly inputUniforms: ReadonlyArray<string>;
}

export interface KernelCodegenSpec {
  readonly rules: ReadonlyArray<RuleCodegenSpec>;
  /** Total slot count across the bucket — product of per-axis
   *  resolved-set sizes (for multi-axis rules) or just the single
   *  axis's `resolved.length` (for one-axis rules). */
  readonly totalSlots: number;
}

/** Emit the WGSL partition kernel for the given rules. */
export function emitPartitionKernel(spec: KernelCodegenSpec): string {
  if (spec.rules.length === 0) {
    throw new Error("kernelCodegen: at least one rule is required");
  }
  if (spec.rules.length > 1) {
    throw new Error("kernelCodegen: multi-axis rules not yet supported (single-axis only for now)");
  }
  if (spec.totalSlots > 16) {
    throw new Error(`kernelCodegen: totalSlots ${spec.totalSlots} exceeds the v1 limit of 16`);
  }

  const rule = spec.rules[0]!;
  const N = spec.totalSlots;
  if (N !== rule.resolved.length) {
    throw new Error(
      `kernelCodegen: totalSlots (${N}) != resolved.length (${rule.resolved.length})`,
    );
  }

  // SLOT_LOOKUP[enumValue] = slotIdx (or 0xFFFFFFFFu when invalid).
  // We size the array to the max enum value + 1 the resolved set
  // can produce (small — at most a few entries per axis).
  const maxEnum = rule.resolved.length > 0 ? Math.max(...rule.resolved) : 0;
  const lookupLen = maxEnum + 1;
  const lookup = new Array<number>(lookupLen).fill(0xFFFFFFFF >>> 0);
  for (let i = 0; i < rule.resolved.length; i++) lookup[rule.resolved[i]!] = i;
  const lookupArr = lookup.map(v => `${v >>> 0}u`).join(", ");

  // Emit the per-axis evaluator function. Walk the Stmt tree.
  // `declared` ReadInput collapses to a Const(declaredU32). `u.<X>`
  // ReadInputs lower to `load_<X>(r.modelRef)` etc.
  const ruleFnBody = emitStmtAsWgslFn(rule);

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
};
struct PartitionParams { numRecords: u32 };

@group(0) @binding(0) var<storage, read>       arena:          array<u32>;
@group(0) @binding(1) var<storage, read>       masterRecords:  array<Record>;
@group(0) @binding(2) var<uniform>             params:         PartitionParams;
${countBindings}
${drawBindings}

const SCAN_REC_U32: u32 = 5u;

const SLOT_LOOKUP_${rule.axis}: array<u32, ${lookupLen}> = array<u32, ${lookupLen}>(${lookupArr});

${LOADERS}

${rule.helpersWGSL}

${ruleFnBody}

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
  let outVal = rule_${rule.axis}(r);
  if (outVal >= ${lookupLen}u) { return; }
  let slotIdx = SLOT_LOOKUP_${rule.axis}[outVal];
  if (slotIdx == 0xFFFFFFFFu) { return; }
  switch (slotIdx) {
${slotCases}
    default: { }
  }
}
`;
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
    declaredU32: rule.declaredU32,
    inputUniforms: rule.inputUniforms,
    localCounter: 0,
  };
  const bodyWgsl = emitStmt(rule.body, ctx, "  ");
  return `fn rule_${rule.axis}(r: Record) -> u32 {\n${bodyWgsl}\n  return 0u;\n}\n`;
}

interface EmitCtx {
  readonly axis: ModeAxis;
  readonly declaredU32: number;
  readonly inputUniforms: ReadonlyArray<string>;
  localCounter: number;
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
      // `declared` collapses to a Const baked at codegen.
      if (r.name === "declared") return `${ctx.declaredU32 >>> 0}u`;
      // `u.<X>` reads from arena via the record's modelRef.
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
