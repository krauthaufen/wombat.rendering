// §7 v2 — chain flattening.
//
// A rule whose IR mentions another derived-uniform name has that producer's
// IR substituted in place (recursively) until the rule reads only non-derived
// inputs (constituents / host uniforms / globals). After flattening every
// record is independent ⇒ one compute dispatch, no levels, no barriers.
// See docs/derived-uniforms-extensible.md § "Chain flattening".

import type { Expr, Type } from "@aardworx/wombat.shader/ir";
import { visitExprChildren, stableStringify } from "@aardworx/wombat.shader/ir";
import { mapExpr } from "@aardworx/wombat.shader/passes";
import type { DerivedRule } from "./rule.js";

export interface RuleInput {
  readonly name: string;
  readonly type: Type;
  /** `true` ⇒ the leaf appeared as `Inverse(ReadInput(name))` — for a constituent
   *  trafo this selects the stored backward half (free); the forward half otherwise. */
  readonly inverse: boolean;
}

/**
 * The rule's input leaves in **first-appearance order** (DFS pre-order of `ir`),
 * deduplicated by `(name, inverse)`. A leaf is either `ReadInput("Uniform", name)`
 * or `Inverse(ReadInput("Uniform", name))`. Order matters: it is the order the
 * generated kernel arm consumes its arguments (e.g. the multiplication order of a
 * matmul chain). Throws if a name is used at two structurally-different types.
 */
export function inputsOf(ir: Expr): RuleInput[] {
  const out: RuleInput[] = [];
  const seen = new Set<string>(); // key = `${inverse ? "i" : "v"}:${name}`
  const typeByName = new Map<string, Type>();
  const record = (name: string, type: Type, inverse: boolean): void => {
    const prev = typeByName.get(name);
    if (prev !== undefined && !typesEqual(prev, type)) {
      throw new Error(`derived rule: uniform '${name}' used at conflicting types`);
    }
    typeByName.set(name, type);
    const key = `${inverse ? "i" : "v"}:${name}`;
    if (!seen.has(key)) {
      seen.add(key);
      out.push({ name, type, inverse });
    }
  };
  const walk = (e: Expr): void => {
    if (e.kind === "ReadInput" && e.scope === "Uniform") {
      record(e.name, e.type, false);
      return;
    }
    if (e.kind === "Inverse" && e.value.kind === "ReadInput" && e.value.scope === "Uniform") {
      record(e.value.name, e.value.type, true);
      return; // do not also count the inner ReadInput as a separate forward leaf
    }
    visitExprChildren(e, walk);
  };
  walk(ir);
  return out;
}

/**
 * Substitute every derived-uniform leaf in `ir` with its (recursively
 * flattened) producer. `name` is the uniform name `ir` is being registered
 * under; it is pushed on the expansion stack so a self-reference — or any
 * cycle through other rules — is caught.
 */
export function flatten(
  name: string,
  ir: Expr,
  derived: ReadonlyMap<string, DerivedRule>,
  _stack: readonly string[] = [],
): Expr {
  if (_stack.includes(name)) {
    throw new Error(`derived uniform cycle: ${[..._stack, name].join(" → ")}`);
  }
  const stack = [..._stack, name];
  return mapExpr(ir, (e) => {
    if (e.kind === "ReadInput" && e.scope === "Uniform") {
      const producer = derived.get(e.name);
      if (producer !== undefined) {
        return flatten(e.name, producer.ir, derived, stack);
      }
    }
    return e;
  });
}

function typesEqual(a: Type, b: Type): boolean {
  return a === b || stableStringify(a) === stableStringify(b);
}
