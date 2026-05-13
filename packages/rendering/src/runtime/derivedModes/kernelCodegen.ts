// Phase 5c.3 — codegen the partition kernel from a set of mode-rule IRs.
//
// Replaces the hand-rolled `PARTITION_FLIP_CULL_BY_DET_WGSL` with a
// per-bucket kernel string built from whatever rules the bucket
// actually carries.
//
// Input: one `RuleSlot` per axis-with-rule. Each slot has:
//   - the rule's IR (output type must be u32; the body reads
//     `ReadInput("Uniform", "<X>")` for arena inputs and
//     `ReadInput("Declared", "<axis>")` for the SG-context value).
//   - per-RO arena byte offsets for each uniform leaf (a CPU side-
//     table keyed by leaf name, threaded into the record).
//
// Output: a WGSL kernel with two entry points (`clear` + `partition`)
// matching the GpuPartitionScene's bindings.

import type { Expr, Type, Literal } from "@aardworx/wombat.shader/ir";
import type { ModeAxis } from "./rule.js";
import { MODE_RULE_SCOPES, isDeclaredVarName } from "./rule.js";

export interface RuleCodegenInput {
  readonly axis: ModeAxis;
  /** Body IR — must produce u32. */
  readonly ir: Expr;
  /**
   * Names of `u.<X>` leaves the body reads, in declaration order. The
   * dispatcher places the per-RO arena byte offset for each into the
   * record at `inputUniformIndex[i]` u32 slots after the base record
   * fields (firstEmit, drawIdx, indexStart, indexCount, instanceCount).
   * For now the record is hardcoded to one input (modelRef) — this
   * shape will be generalized in a follow-on.
   */
  readonly inputUniforms: ReadonlyArray<string>;
  /** Number of distinct output values for this axis (= rule.domain.length). */
  readonly domainSize: number;
}

export interface KernelCodegenSpec {
  /** One per axis present on this bucket. */
  readonly rules: ReadonlyArray<RuleCodegenInput>;
  /** Total slot count = product of `rules[i].domainSize`. */
  readonly totalSlots: number;
}

/** Public for tests — combines the per-axis u32 outputs into a single
 *  modeKey via mixed-radix packing. Caller uses the same encoding when
 *  building the CPU-side modeKeyToSlotIdx lookup. */
export function packModeKey(axisIndices: ReadonlyArray<number>, domainSizes: ReadonlyArray<number>): number {
  let key = 0;
  let stride = 1;
  for (let i = 0; i < axisIndices.length; i++) {
    key += axisIndices[i]! * stride;
    stride *= domainSizes[i]!;
  }
  return key;
}

/** Emit the WGSL partition kernel for the given rules. */
export function emitPartitionKernel(spec: KernelCodegenSpec): string {
  if (spec.rules.length === 0) {
    throw new Error("kernelCodegen: at least one rule is required");
  }
  if (spec.rules.length > 4) {
    throw new Error("kernelCodegen: at most 4 axes per bucket are supported in v1");
  }
  if (spec.totalSlots > 16) {
    throw new Error(`kernelCodegen: totalSlots ${spec.totalSlots} exceeds the v1 limit of 16`);
  }

  // For each rule we produce a function `rule_<axis>(r: Record) -> u32`.
  // The body inlines the IR — `ReadInput("Uniform", X)` becomes a
  // call to `load_<X>(arena, r.modelRef)` (currently fixed to the
  // single arena slot the record carries); `ReadInput("Declared", axis)`
  // becomes a field read on the per-bucket `Params` uniform.
  const declaredFields = spec.rules.map(r => `  decl_${r.axis}: u32,`).join("\n");

  const ruleFns = spec.rules.map(rule => emitRuleFn(rule)).join("\n");

  const axisDispatch = spec.rules.map(r => `  let i_${r.axis} = rule_${r.axis}(r);`).join("\n");

  const domainSizes = spec.rules.map(r => r.domainSize);
  const pack = spec.rules.map((r, i) => {
    let stride = 1;
    for (let j = 0; j < i; j++) stride *= domainSizes[j]!;
    return stride === 1 ? `i_${r.axis}` : `${stride}u * i_${r.axis}`;
  }).join(" + ");

  const slotCases = new Array(spec.totalSlots).fill(0).map((_, slotIdx) => {
    return `    case ${slotIdx}u: { let off = atomicAdd(&slot${slotIdx}Count[0], 1u); let base = off * SCAN_REC_U32;` +
      ` slot${slotIdx}DrawTable[base + 0u] = 0u;` +
      ` slot${slotIdx}DrawTable[base + 1u] = r.drawIdx;` +
      ` slot${slotIdx}DrawTable[base + 2u] = r.indexStart;` +
      ` slot${slotIdx}DrawTable[base + 3u] = r.indexCount;` +
      ` slot${slotIdx}DrawTable[base + 4u] = r.instanceCount; }`;
  }).join("\n");

  const slotCountClears = new Array(spec.totalSlots).fill(0).map((_, i) => `  atomicStore(&slot${i}Count[0], 0u);`).join("\n");

  // Bindings: arena + master + params + N atomic counts + N draw tables.
  const N = spec.totalSlots;
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
struct PartitionParams {
  numRecords: u32,
${declaredFields}
};

@group(0) @binding(0) var<storage, read>       arena:           array<u32>;
@group(0) @binding(1) var<storage, read>       masterRecords:   array<Record>;
@group(0) @binding(2) var<uniform>             params:          PartitionParams;
${countBindings}
${drawBindings}

const SCAN_REC_U32: u32 = 5u;

${EMIT_LOADERS}

${ruleFns}

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
${axisDispatch}
  let slotIdx = ${pack};
  switch (slotIdx) {
${slotCases}
    default: { }
  }
}
`;
}

/** Loader helpers — currently only `mat3x3<f32>` reads from arena are
 *  needed (det-flip-by-cull). The matrix lives at `(modelRef + 16)`
 *  bytes (alloc-header pad) in arena, row-major in 4-wide rows. */
const EMIT_LOADERS = `
fn load_mat3_upper(refBytes: u32) -> mat3x3<f32> {
  let baseU32 = (refBytes + 16u) >> 2u;
  return mat3x3<f32>(
    vec3<f32>(bitcast<f32>(arena[baseU32 +  0u]), bitcast<f32>(arena[baseU32 +  4u]), bitcast<f32>(arena[baseU32 +  8u])),
    vec3<f32>(bitcast<f32>(arena[baseU32 +  1u]), bitcast<f32>(arena[baseU32 +  5u]), bitcast<f32>(arena[baseU32 +  9u])),
    vec3<f32>(bitcast<f32>(arena[baseU32 +  2u]), bitcast<f32>(arena[baseU32 +  6u]), bitcast<f32>(arena[baseU32 + 10u])),
  );
}
`;

// ── IR → WGSL printer (mirrors derivedUniforms/codegen.ts; reproduces
//    the subset we need so this module is self-contained). ──

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
function wgslLiteral(l: Literal): string {
  switch (l.kind) {
    case "Bool":  return l.value ? "true" : "false";
    case "Int":   return l.signed ? `${l.value}i` : `${l.value >>> 0}u`;
    case "Float": return Number.isFinite(l.value) ? (Number.isInteger(l.value) ? `${l.value}.0` : `${l.value}`) : "0.0";
    case "Null":  return "0";
  }
}
const BIN: Partial<Record<Expr["kind"], string>> = {
  Add: "+", Sub: "-", Mul: "*", Div: "/", Mod: "%",
  MulMatMat: "*", MulMatVec: "*", MulVecMat: "*",
  And: "&&", Or: "||", BitAnd: "&", BitOr: "|", BitXor: "^",
  Eq: "==", Neq: "!=", Lt: "<", Le: "<=", Gt: ">", Ge: ">=",
};

/** Lower an IR expression to a WGSL expression string, given a way to
 *  resolve `ReadInput` leaves to existing values in scope. */
function printExpr(e: Expr, resolveInput: (scope: string, name: string, type: Type) => string): string {
  const r = (x: Expr): string => printExpr(x, resolveInput);
  switch (e.kind) {
    case "ReadInput": return resolveInput((e as { scope: string }).scope, (e as { name: string }).name, e.type);
    case "Var": {
      const v = (e as { var: { name: string } }).var;
      const decl = isDeclaredVarName(v.name);
      if (decl !== null) return `params.decl_${decl.axis}`;
      return v.name;
    }
    case "Const": return wgslLiteral((e as { value: Literal }).value);
    case "Neg": return `(-${r((e as { value: Expr }).value)})`;
    case "Not": return `(!${r((e as { value: Expr }).value)})`;
    case "Add": case "Sub": case "Mul": case "Div": case "Mod":
    case "MulMatMat": case "MulMatVec": case "MulVecMat":
    case "And": case "Or":
    case "Eq": case "Neq": case "Lt": case "Le": case "Gt": case "Ge":
      return `(${r((e as { lhs: Expr }).lhs)} ${BIN[e.kind]} ${r((e as { rhs: Expr }).rhs)})`;
    case "Transpose": return `transpose(${r((e as { value: Expr }).value)})`;
    case "Determinant": return `determinant(${r((e as { value: Expr }).value)})`;
    case "Dot": return `dot(${r((e as { lhs: Expr }).lhs)}, ${r((e as { rhs: Expr }).rhs)})`;
    case "Cross": return `cross(${r((e as { lhs: Expr }).lhs)}, ${r((e as { rhs: Expr }).rhs)})`;
    case "Length": return `length(${r((e as { value: Expr }).value)})`;
    case "VecSwizzle":
      return `${r((e as { value: Expr }).value)}.${((e as { comps: ReadonlyArray<string> }).comps).join("")}`;
    case "VecItem":
      return `${r((e as { value: Expr }).value)}[${r((e as { index: Expr }).index)}]`;
    case "MatrixCol":
      return `${r((e as { matrix: Expr }).matrix)}[${r((e as { col: Expr }).col)}]`;
    case "MatrixElement":
      return `${r((e as { matrix: Expr }).matrix)}[${r((e as { col: Expr }).col)}][${r((e as { row: Expr }).row)}]`;
    case "NewVector":
      return `${wgslTypeName(e.type)}(${(e as { components: ReadonlyArray<Expr> }).components.map(r).join(", ")})`;
    case "Conditional":
      return `select(${r((e as { ifFalse: Expr }).ifFalse)}, ${r((e as { ifTrue: Expr }).ifTrue)}, ${r((e as { cond: Expr }).cond)})`;
    case "Convert":
    case "ConvertMatrix":
      return `${wgslTypeName(e.type)}(${r((e as { value: Expr }).value)})`;
    case "CallIntrinsic": {
      const ci = e as { op: { emit: { wgsl: string } }; args: ReadonlyArray<Expr> };
      return `${ci.op.emit.wgsl}(${ci.args.map(r).join(", ")})`;
    }
    default:
      throw new Error(`kernelCodegen: IR node '${e.kind}' is not supported`);
  }
}

function emitRuleFn(rule: RuleCodegenInput): string {
  // Resolution policy:
  //   - ReadInput("Uniform", "<X>"): the record carries `r.modelRef`
  //     for the first-and-only uniform input (v1). The leaf's type
  //     determines which loader we call.
  //   - `Var("__declared_<axis>")` is the SG-declared value — handled
  //     in `printExpr` directly (emitted as `params.decl_<axis>`).
  const resolve = (scope: string, name: string, type: Type): string => {
    if (scope === MODE_RULE_SCOPES.UNIFORM) {
      // For v1 the only supported uniform leaf is a matrix read
      // expressed as ConvertMatrix(upperLeft3x3) wrapped around a
      // mat4 leaf — but the rule body may use the leaf directly with
      // .upperLeft3x3()/.determinant() etc. So we just hand back the
      // arena reader for the upper-left 3x3 of the record's
      // ModelTrafo. The rule is expected to use that path; loading
      // a generic mat4 from arena is straightforward but unused in
      // the current demo.
      if (type.kind === "Matrix" && type.rows === 3 && type.cols === 3) {
        return `load_mat3_upper(r.modelRef)`;
      }
      // Fallback: also synthesize a mat4 read (16 floats starting at
      // refBytes + 16). For now this isn't reachable by any rule the
      // demo uses; codegen throws to make the gap obvious.
      throw new Error(
        `kernelCodegen: v1 only supports ConvertMatrix(mat3) loads from arena ` +
        `(via u.<X>.upperLeft3x3()); leaf '${name}' had type ${JSON.stringify(type)}`,
      );
    }
    throw new Error(`kernelCodegen: unknown leaf scope '${scope}'`);
  };
  // Special-case: if the IR's outermost node is `ConvertMatrix(value)`
  // where value is a uniform mat4 leaf, lower it via load_mat3_upper.
  // The resolver above only sees the bottom leaf, so handle this here
  // by rewriting at the top.
  const lowered = rewriteUpperLeftFromMat4(rule.ir);
  const body = printExpr(lowered, resolve);
  return `\nfn rule_${rule.axis}(r: Record) -> u32 {\n  return ${body};\n}\n`;
}

/**
 * Rewrite `ConvertMatrix(value=ReadInput("Uniform", X), type=mat3)`
 * (the .upperLeft3x3() of a mat4 leaf) into a direct read of the
 * arena-stored upper-left 3x3. Lets downstream resolution stay simple.
 */
function rewriteUpperLeftFromMat4(e: Expr): Expr {
  const M = e as Record<string, unknown>;
  const visit = (n: Expr): Expr => {
    const obj = n as Record<string, unknown>;
    if (obj.kind === "ConvertMatrix") {
      const val = obj.value as Expr;
      const valObj = val as Record<string, unknown>;
      const t = obj.type as Type;
      if (
        valObj.kind === "ReadInput"
        && valObj.scope === MODE_RULE_SCOPES.UNIFORM
        && t.kind === "Matrix" && t.rows === 3 && t.cols === 3
      ) {
        // Replace with a synthetic ReadInput typed as mat3 so the
        // resolver returns load_mat3_upper(r.modelRef) directly.
        return { kind: "ReadInput", scope: MODE_RULE_SCOPES.UNIFORM, name: valObj.name, type: t } as Expr;
      }
    }
    // Recurse into children.
    const out: Record<string, unknown> = { ...obj };
    for (const k of Object.keys(out)) {
      const v = out[k];
      if (v !== null && typeof v === "object" && "kind" in (v as object)) {
        out[k] = visit(v as Expr);
      } else if (Array.isArray(v)) {
        out[k] = v.map((c: unknown) =>
          c !== null && typeof c === "object" && "kind" in (c as object) ? visit(c as Expr) : c,
        );
      }
    }
    return out as unknown as Expr;
  };
  void M;
  return visit(e);
}
