// Derived-mode rule — branded combination of a shader-IR rule
// expression (from `rule(...)` in wombat.shader) + an axis + the
// SG-declared aval. The IR is what gets compiled into the partition
// kernel; the declared aval drives per-frame re-specialisation.
//
// Authoring flow (with `@boperators/plugin-vite` + the shader-vite
// plugin both installed):
//
//     declare const u: { ModelTrafo: M44f };
//     declare const declared: number;
//
//     const cullExpr = rule(() => {
//       const m: M33f = mat3(u.ModelTrafo);
//       const det = determinant(m);
//       return det < 0 ? flipCull(declared) : declared;
//     });
//
//     const cullRule = derivedMode("cull", cullExpr, { declared: cullModeC });
//
// At runtime, heapScene reads `cullRule.expr.template`, extracts the
// entry's body Stmt, calls `analyseOutputSet(body)` to obtain the
// symbolic set, and for each declared-aval value resolves the set
// via `evaluateSet({"declared": current}, intrinsics)` to size the
// bucket's slots + codegen the partition kernel.

import type { RuleExpr } from "@aardworx/wombat.shader";
import type { aval } from "@aardworx/wombat.adaptive";
import type { CullMode, FrontFace, Topology, AttachmentBlend } from "../pipelineCache/index.js";

export type ModeAxis =
  | "cull" | "frontFace" | "topology"
  | "depthCompare" | "depthWrite"
  | "alphaToCoverage"
  /** Per-attachment blend state. The rule's u32 output indexes into
   *  the `values` array on `DerivedModeOptions` (or the `resolve`
   *  callback); each entry is a full {@link AttachmentBlend} struct
   *  with srcFactor / dstFactor / operation / writeMask / enabled
   *  fields — letting blend rules specify the entire mode object,
   *  not just an enum index. */
  | "blend";

export type ModeValue<A extends ModeAxis> =
  A extends "cull"           ? CullMode :
  A extends "frontFace"      ? FrontFace :
  A extends "topology"       ? Topology :
  A extends "depthCompare"   ? GPUCompareFunction :
  A extends "depthWrite"     ? boolean :
  A extends "alphaToCoverage" ? boolean :
  A extends "blend"          ? AttachmentBlend :
  never;

export interface DerivedModeRule<A extends ModeAxis = ModeAxis> {
  readonly __derivedModeRule: true;
  readonly axis: A;
  /** Shader-IR rule body. The entry's return Expr is the rule. */
  readonly expr: RuleExpr<number>;
  /** SG-context declared value for this axis. May be a reactive aval
   *  — heapScene re-specialises + swaps cached kernels on mark. */
  readonly declared: aval<ModeValue<A>> | ModeValue<A>;
  /**
   * Optional u32 → axis value map. Required for axes whose
   * `ModeValue<A>` is a structured object ({@link blend}) — the
   * rule's u32 output is an INDEX, not the canonical enum, and the
   * heap renderer needs this to build per-slot pipeline descriptors.
   *
   * For enum axes (cull / frontFace / topology / depthCompare /
   * depthWrite / alphaToCoverage) the runtime falls back to a
   * canonical axis-enum table when `resolve` is omitted.
   *
   * Receives `u32`s in the order they appear in the rule's resolved
   * output set (typically just `0`..`N-1` for an N-output rule).
   */
  readonly resolve: ((u32: number) => ModeValue<A>) | undefined;
}

export function isDerivedModeRule(x: unknown): x is DerivedModeRule {
  return typeof x === "object" && x !== null
      && (x as { __derivedModeRule?: unknown }).__derivedModeRule === true;
}

export interface DerivedModeOptions<A extends ModeAxis> {
  readonly declared: aval<ModeValue<A>> | ModeValue<A>;
  /**
   * For "blend" (and other structured axes), supply a callback that
   * maps the rule's u32 output to the full mode value (a
   * {@link AttachmentBlend} for blend rules):
   *
   *     const blendRule = derivedMode("blend",
   *       rule(() => u.Premultiplied ? 1 : 0), {
   *         declared: NOOP_BLEND_AT_INDEX_0,
   *         resolve: (i) => i === 1
   *           ? { enabled: true, color: { srcFactor: "one",        dstFactor: "one-minus-src-alpha", operation: "add" },
   *               alpha:   { srcFactor: "one",        dstFactor: "one-minus-src-alpha", operation: "add" }, writeMask: 0xF }
   *           : { enabled: true, color: { srcFactor: "src-alpha",  dstFactor: "one-minus-src-alpha", operation: "add" },
   *               alpha:   { srcFactor: "one",        dstFactor: "one-minus-src-alpha", operation: "add" }, writeMask: 0xF },
   *       });
   *
   * Equivalent shorthand via the `values` array:
   *
   *     resolve: undefined,
   *     values:  [STRAIGHT_ALPHA_BLEND, PREMULT_ALPHA_BLEND],
   *
   * (`resolve` and `values` are mutually exclusive; `resolve` wins
   * when both are passed.)
   *
   * For enum axes (cull etc.), omit both — the runtime uses a
   * canonical enum table to map u32s back to axis values.
   */
  readonly resolve?: (u32: number) => ModeValue<A>;
  /** Compact alias for `resolve: i => values[i]`. Same constraints
   *  as `resolve`. */
  readonly values?: ReadonlyArray<ModeValue<A>>;
}

/**
 * Brand a `RuleExpr` (the shader-IR artefact emitted by the
 * `rule(...)` build-time marker) as a derived-mode rule for `axis`.
 *
 * The `declared` option carries the SG-context value the rule reads
 * via `declare const declared: number` in its body. heapScene
 * substitutes it (as a Const) before codegen — when `declared` is a
 * cval, every mark triggers a re-specialise + cached-kernel swap.
 *
 * `domain` is NOT a parameter: the renderer infers the per-declared
 * set of possible outputs from the rule's IR via
 * `analyseOutputSet` + `evaluateSet`.
 */
export function derivedMode<A extends ModeAxis>(
  axis: A,
  expr: RuleExpr<number>,
  options: DerivedModeOptions<A>,
): DerivedModeRule<A> {
  let resolve: ((u32: number) => ModeValue<A>) | undefined;
  if (options.resolve !== undefined) {
    resolve = options.resolve;
  } else if (options.values !== undefined) {
    const arr = options.values;
    resolve = (i: number): ModeValue<A> => {
      const v = arr[i];
      if (v === undefined) {
        throw new Error(
          `derivedMode("${axis}"): rule emitted u32=${i} but \`values\` only has ${arr.length} entries`,
        );
      }
      return v;
    };
  }
  return {
    __derivedModeRule: true,
    axis,
    expr,
    declared: options.declared,
    resolve,
  };
}
