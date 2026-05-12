// §7 v2 — derived-uniform rule: a content-hashed pure IR fragment.
//
// A DerivedRule is a closure-shaped value: an output WGSL type plus an IR
// expression ("ir") that computes it from `ReadInput("Uniform", name)` leaves.
// The names are resolved per-RenderObject at registration time (constituent
// slot / host uniform). See docs/derived-uniforms-extensible.md.
//
// This is the pre-plugin surface: rules are hand-built from IR exprs. The
// inline `derivedUniform(u => …)` marker (lowered by the wombat-shader-vite
// plugin) lands later and produces exactly this shape.

import type { Expr, Type } from "@aardworx/wombat.shader/ir";
import { hashValue, stableStringify } from "@aardworx/wombat.shader/ir";

/** A rule's IR fragment: the result expression, over `ReadInput("Uniform", …)` leaves. */
export type IRFragment = Expr;

export interface DerivedRule<T = unknown> {
  /** Output WGSL type (`=== ir.type`). Drives the storer codegen and the drawHeader byte budget. */
  readonly outputType: Type;
  /** Pure IR computing the output; every leaf input is a `ReadInput("Uniform", name)`. */
  readonly ir: IRFragment;
  /** Stable structural content hash of `ir` — the registry dedup key. */
  readonly hash: string;
  /** Phantom — carries the TS result type through `derivedUniform<T>`. Never read. */
  readonly __t?: T;
}

/** True iff two IR `Type`s are structurally identical. */
export function sameType(a: Type, b: Type): boolean {
  return a === b || stableStringify(a) === stableStringify(b);
}

/** Stable content hash of an IR expression. */
export function hashIR(ir: Expr): string {
  return hashValue(ir);
}

/** Build a `DerivedRule` from a hand-constructed IR expression. */
export function ruleFromIR<T = unknown>(ir: Expr, outputType: Type = ir.type): DerivedRule<T> {
  if (!sameType(outputType, ir.type)) {
    throw new Error(
      `derived rule: declared outputType ${stableStringify(outputType)} != ir type ${stableStringify(ir.type)}`,
    );
  }
  return { outputType, ir, hash: hashIR(ir) };
}

/** A `ReadInput("Uniform", name)` leaf of the given type — the only leaf kind a rule may use. */
export function uniformRef(name: string, type: Type): Expr {
  return { kind: "ReadInput", scope: "Uniform", name, type };
}
