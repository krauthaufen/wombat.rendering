// `derivedMode(axis, (u, declared) => DerivedExpr)` — author-facing
// marker for per-RO pipeline-state values that are functions of the
// RO's uniforms.
//
// The body is a *traced* expression — same IR machinery §7 derived
// uniforms uses. `u.<Name>` is a uniform leaf (read from arena per
// RO), `declared` is the SG-context value (per-dispatch kernel
// uniform). The output is a u32 ENUM INDEX into the rule's `domain`,
// e.g. for cull with `domain: ["none", "front", "back"]` the rule
// returns `0` to route a record to the "none" slot, `2` for "back".
//
// At bucket-init time the heap runtime codegens a partition kernel
// from the rule's IR: per record evaluate every axis-rule, pack the
// indices into a modeKey, look up the slot index, atomic-scatter.
// No hand-rolled WGSL anywhere.

import type { Expr, Type } from "@aardworx/wombat.shader/ir";
import { DerivedExpr } from "../derivedUniforms/marker.js";
import type { aval } from "@aardworx/wombat.adaptive";
import type { CullMode, FrontFace, Topology } from "../pipelineCache/index.js";

/** Discrete pipeline-state axes a `derivedMode` rule can target. */
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

const Tu32: Type = { kind: "Int", signed: false, width: 32 };

/** Scope for `u` (per-RO uniform reads, resolved at arena byte offsets). */
const SCOPE_UNIFORM = "Uniform";
/** Prefix for `declared` leaves: surfaced as IR `Var` whose name is
 *  `__declared_<axis>` so the partition-kernel codegen can pattern-
 *  match it and emit `params.decl_<axis>`. (The shader-IR `InputScope`
 *  enum is closed; using `Var` avoids forking it.) */
const DECLARED_VAR_PREFIX = "__declared_";

/**
 * Trace proxy for `u`: `u.<Name>` returns a DerivedExpr wrapping
 * `ReadInput("Uniform", "<Name>")`. Defaults to mat4 (the trafo case);
 * use `.as(...)` to re-type a leaf inline.
 *
 * Side-effect: collects each accessed leaf name into `out` so the
 * runtime can resolve arena byte offsets for them at addRO time.
 */
function makeUniformProxy(out: Set<string>): Record<string, DerivedExpr> {
  const mat4: Type = { kind: "Matrix", element: { kind: "Float", width: 32 }, rows: 4, cols: 4 };
  return new Proxy({} as Record<string, DerivedExpr>, {
    get(_t, key): DerivedExpr {
      if (typeof key !== "string") throw new Error("derivedMode: leaf names must be strings");
      out.add(key);
      return new DerivedExpr({ kind: "ReadInput", scope: SCOPE_UNIFORM, name: key, type: mat4 });
    },
  });
}

/**
 * `declared` is a single u32 leaf — the SG-context value for this
 * axis, supplied by the dispatcher each frame from the rule's
 * `declared` aval. (Codegen reads it from a per-dispatch kernel
 * uniform; there's no arena offset.)
 */
function declaredLeaf(axis: ModeAxis): DerivedExpr {
  const name = `${DECLARED_VAR_PREFIX}${axis}`;
  return new DerivedExpr({ kind: "Var", var: { name, type: Tu32, mutable: false }, type: Tu32 });
}

/** True iff `varName` is a `declared` leaf produced by `declaredLeaf`. */
export function isDeclaredVarName(varName: string): { axis: ModeAxis } | null {
  if (!varName.startsWith(DECLARED_VAR_PREFIX)) return null;
  const axis = varName.slice(DECLARED_VAR_PREFIX.length) as ModeAxis;
  return { axis };
}

/** A traced derived-mode rule. The IR's output type must be `u32`
 *  (the index into `domain`). The runtime checks this at codegen. */
export interface DerivedModeRule<A extends ModeAxis = ModeAxis> {
  readonly __derivedModeRule: true;
  readonly axis: A;
  /** Pure IR producing a u32 slot index. Built by `derivedMode(...)`. */
  readonly ir: Expr;
  /**
   * Names of `u.<Name>` leaves the body read. The bucket resolves
   * each to its arena byte offset at addRO time and packs them into
   * the partition kernel record.
   */
  readonly inputUniforms: ReadonlyArray<string>;
  /**
   * Domain of possible output values, in the order they appear in
   * the rule's u32 output. `domain[0]` corresponds to the rule
   * returning `0`, `domain[1]` to `1`, etc. The bucket creates one
   * slot per domain entry.
   */
  readonly domain: ReadonlyArray<ModeValue<A>>;
  /** The SG-context declared value for this axis (may be reactive). */
  readonly declared: aval<ModeValue<A>> | ModeValue<A>;
}

/** Brand check. */
export function isDerivedModeRule(x: unknown): x is DerivedModeRule {
  return typeof x === "object" && x !== null
      && (x as { __derivedModeRule?: unknown }).__derivedModeRule === true;
}

export interface DerivedModeOptions<A extends ModeAxis> {
  readonly domain: ReadonlyArray<ModeValue<A>>;
  readonly declared: aval<ModeValue<A>> | ModeValue<A>;
}

/**
 * Define a derived-mode rule by tracing an IR-builder closure.
 *
 *     const cullRule = derivedMode("cull", (u, declared) => {
 *       const det = u.ModelTrafo.upperLeft3x3().determinant();
 *       // flipCull lookup: none→none, front→back, back→front.
 *       const flipped = pickEnum(declared,
 *         DerivedExpr.u32(0),  // none → none
 *         DerivedExpr.u32(2),  // front → back
 *         DerivedExpr.u32(1),  // back → front
 *       );
 *       return flipped.select(declared, det.lt(DerivedExpr.f32(0)));
 *     }, {
 *       domain: ["none", "front", "back"],
 *       declared: cullModeC,
 *     });
 *
 * The closure body must build pure expressions on top of `u.<Name>`
 * leaves and `declared`. It must NOT use host-side conditionals — use
 * `.select(...)` and `.eq(...)` etc. to encode branches.
 */
export function derivedMode<A extends ModeAxis>(
  axis: A,
  build: (u: Record<string, DerivedExpr>, declared: DerivedExpr) => DerivedExpr,
  options: DerivedModeOptions<A>,
): DerivedModeRule<A> {
  const leaves = new Set<string>();
  const uProxy = makeUniformProxy(leaves);
  const declared = declaredLeaf(axis);
  const result = build(uProxy, declared);
  if (!(result instanceof DerivedExpr)) {
    throw new Error("derivedMode: the builder must return a DerivedExpr");
  }
  if (result.ir.type.kind !== "Int" || result.ir.type.signed || result.ir.type.width !== 32) {
    throw new Error(
      `derivedMode("${axis}"): rule body must return a u32 (slot index into domain). ` +
      `Got ${JSON.stringify(result.ir.type)}`,
    );
  }
  if (options.domain.length === 0) {
    throw new Error(`derivedMode("${axis}"): domain must be non-empty`);
  }
  return {
    __derivedModeRule: true,
    axis,
    ir: result.ir,
    inputUniforms: Array.from(leaves),
    domain: options.domain,
    declared: options.declared,
  };
}

/**
 * Pick the n-th DerivedExpr from `cases` based on the u32 value of
 * `index`. Lowered to a chain of `select` (no array literals in the
 * IR). For an out-of-range index returns `cases[0]`.
 *
 *     pickEnum(declared, valueWhenZero, valueWhenOne, valueWhenTwo)
 */
export function pickEnum(index: DerivedExpr, ...cases: DerivedExpr[]): DerivedExpr {
  if (cases.length === 0) throw new Error("pickEnum: at least one case is required");
  if (cases.length === 1) return cases[0]!;
  let acc = cases[0]!;
  for (let i = 1; i < cases.length; i++) {
    acc = cases[i]!.select(acc, index.eq(DerivedExpr.u32(i)));
  }
  return acc;
}

/** Internal: the IR scope used for `u.<X>` leaves. */
export const MODE_RULE_SCOPES = {
  UNIFORM: SCOPE_UNIFORM,
} as const;
