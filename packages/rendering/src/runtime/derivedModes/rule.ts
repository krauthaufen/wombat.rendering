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
  /** Shader-IR rule body — the entry's return Expr is the rule. For
   *  axes whose `ModeValue<A>` is a struct (e.g. blend's
   *  `AttachmentBlend`) the body returns object literals directly;
   *  the renderer fingerprints distinct returns and auto-assigns
   *  slot indices. For enum axes (cull etc.) the body returns u32
   *  enum indices that the renderer maps via the canonical table. */
  readonly expr: RuleExpr<ModeValue<A> | number>;
  /**
   * SG-context value for this axis. Filled in by `<Sg CullMode>` /
   * `<Sg DepthTest>` / etc. at traversal time — users never write
   * it. The rule body's `declared` ambient leaf resolves to this;
   * when the underlying aval marks, heapScene re-specialises +
   * swaps the cached kernel + pipeline set.
   *
   * `undefined` is the initial state right after `derivedMode(...)`;
   * the SG fills it in before the rule reaches the renderer.
   */
  readonly declared: aval<ModeValue<A>> | ModeValue<A> | undefined;
}

export function isDerivedModeRule(x: unknown): x is DerivedModeRule {
  return typeof x === "object" && x !== null
      && (x as { __derivedModeRule?: unknown }).__derivedModeRule === true;
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
  expr: RuleExpr<ModeValue<A> | number>,
): DerivedModeRule<A> {
  return {
    __derivedModeRule: true,
    axis,
    expr,
    declared: undefined,  // SG fills this in at traversal time
  };
}
