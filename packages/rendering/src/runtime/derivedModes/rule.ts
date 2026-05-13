// `derivedMode(axis, (u, declared) => modeValue)` — author-facing
// marker for per-RO pipeline-state values that are functions of the
// RO's uniforms.
//
// v1 is CPU-evaluated: the user's closure runs in JS each time the
// rule's inputs mark. That's a good fit for the typical authoring
// shapes (a uniform-driven enum flip, a determinant-sign cull flip,
// etc.) and pairs naturally with the bucket-level rebuild fast path
// shipped in 0.9.16 — only ROs whose inputs actually changed get
// re-evaluated, and same-key buckets re-bind their pipeline once
// rather than per-RO.
//
// v2 (deferred) lowers rules to a GPU compute kernel via §7's IR
// machinery — needed only when bulk-animated per-RO inputs become a
// CPU-side bottleneck. See docs/derived-mode-rules-plan.md.
//
// What can be derived (the small-enum axes — cull/blend/depth-compare/
// topology/etc.) and what stays static (depth bias, blend constant,
// stencil ref) is covered in docs/derived-modes.md.

import type { CullMode, FrontFace, Topology } from "../pipelineCache/index.js";

/** Discrete pipeline-state axes a `derivedMode` rule can target. v1
 *  supports cull / frontFace / topology / depthCompare / depthWrite /
 *  blendEnabled. Stencil + per-attachment blend factor/operation are
 *  deferred — same carve-outs as `derived-modes.md`. */
export type ModeAxis =
  | "cull" | "frontFace" | "topology"
  | "depthCompare" | "depthWrite"
  | "alphaToCoverage";

/** Output value type per axis. */
export type ModeValue<A extends ModeAxis> =
  A extends "cull"           ? CullMode :
  A extends "frontFace"      ? FrontFace :
  A extends "topology"       ? Topology :
  A extends "depthCompare"   ? GPUCompareFunction :
  A extends "depthWrite"     ? boolean :
  A extends "alphaToCoverage" ? boolean :
  never;

/**
 * The closure signature a `derivedMode` rule body takes:
 *   - `u`: a proxy whose properties resolve to the RO's uniform values
 *     at evaluation time. The proxy is type-erased in v1 — declare the
 *     uniforms you'll read via `UniformScope` augmentation on the
 *     wombat.shader side for compile-time safety, or annotate locally.
 *   - `declared`: the SG-declared value for this axis at this leaf.
 *     Identity rules `(u, d) => d` are valid and useful for axes that
 *     want to pass declared through unchanged.
 */
export type DerivedModeBuilder<A extends ModeAxis> = (
  u: Record<string, unknown>,
  declared: ModeValue<A>,
) => ModeValue<A>;

export interface DerivedModeRule<A extends ModeAxis = ModeAxis> {
  readonly __derivedModeRule: true;
  readonly axis:    A;
  readonly evaluate: DerivedModeBuilder<A>;
  /**
   * Conservative over-approximation of the rule's output set. Used by
   * the build-time pre-warm in v2 to enumerate the pipelines a scene
   * could realize. v1 doesn't use it at runtime, but recording it
   * here keeps the API forward-compatible.
   */
  readonly domain?: ReadonlyArray<ModeValue<A>>;
}

/**
 * Author a derived mode rule.
 *
 *     derivedMode("cull", (u, declared) =>
 *       u.WindingFlipped ? flipCull(declared) : declared);
 *
 *     derivedMode("blend", (u) => u.Premultiplied ? "premul" : "straight");
 *
 *     // Determinant-flip-cull (the motivating case):
 *     derivedMode("cull", (u, declared) => {
 *       const m = u.ModelTrafo as M44d;
 *       const det =
 *         m.M00 * (m.M11 * m.M22 - m.M12 * m.M21) -
 *         m.M01 * (m.M10 * m.M22 - m.M12 * m.M20) +
 *         m.M02 * (m.M10 * m.M21 - m.M11 * m.M20);
 *       return det < 0 ? flipCull(declared) : declared;
 *     });
 */
export function derivedMode<A extends ModeAxis>(
  axis: A,
  evaluate: DerivedModeBuilder<A>,
  options: { readonly domain?: ReadonlyArray<ModeValue<A>> } = {},
): DerivedModeRule<A> {
  const out: DerivedModeRule<A> = options.domain !== undefined
    ? { __derivedModeRule: true, axis, evaluate, domain: options.domain }
    : { __derivedModeRule: true, axis, evaluate };
  return out;
}

/** Brand check — distinguishes a DerivedModeRule from an aval/raw value
 *  at runtime so PipelineState consumers can route accordingly. */
export function isDerivedModeRule(x: unknown): x is DerivedModeRule {
  return typeof x === "object" && x !== null
      && (x as { __derivedModeRule?: unknown }).__derivedModeRule === true;
}

/** Convenience: flip 'back' ↔ 'front' (and pass through 'none').
 *  Useful in `(u, declared) => mirrored ? flipCull(declared) : declared`. */
export function flipCull(c: CullMode): CullMode {
  return c === "back" ? "front" : c === "front" ? "back" : "none";
}
