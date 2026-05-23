// CPU evaluation of a derived-MODE rule for the legacy per-RO path.
//
// A `derivedMode(...)` rule is a RuleExpr whose body computes a pipeline-axis
// value (cull / frontFace / …) from per-RO uniforms (e.g. determinant of the
// model matrix) and the SG-declared default. The heap path lowers this to a
// GPU partition kernel; the legacy per-RO path has no kernel, so we evaluate
// the rule body on the CPU here — the "v1 evaluates rules CPU-side" design in
// core/renderObject.ts.
//
// Reuses the matrix-aware expression interpreter (derivedUniforms/cpuEval) and
// adds a small Stmt walker (locals + control flow) plus user-function `Call`
// resolution against the rule's own Module (helpers like `flipCull` are NOT
// inlined — they're separate Function items in template.values).

import type { Expr, Stmt, ValueDef } from "@aardworx/wombat.shader/ir";
import type { AdaptiveToken } from "@aardworx/wombat.adaptive";
import { interpretExpr, type EvalValue } from "../derivedUniforms/cpuEval.js";
import type { DerivedModeRule, ModeAxis, ModeValue } from "./rule.js";
import { declaredToU32, resolveAxisValue } from "./axisEnum.js";

const NO_RETURN = Symbol("no-return");
type RunResult = EvalValue | typeof NO_RETURN;

interface FnDef { readonly params: readonly string[]; readonly body: Stmt; }

function truthy(v: EvalValue): boolean {
  return typeof v === "boolean" ? v : typeof v === "number" ? v !== 0 : true;
}

/**
 * Evaluate a derived-mode rule to its concrete axis value for one RO.
 * `readUniform(name)` returns the RO's current value for a `u.<name>` leaf
 * (Trafo3d / M44f / number / …); reading it under `token` registers the
 * dependency so the pipeline re-keys when it changes. Returns the mapped
 * mode value (e.g. "back", or `true` for depthWrite).
 */
export function evaluateModeRule<A extends ModeAxis>(
  rule: DerivedModeRule<A>,
  readUniform: (name: string) => unknown,
  token: AdaptiveToken,
): ModeValue<A> {
  // Collect helper functions + the entry body from the rule's Module.
  const fns = new Map<string, FnDef>();
  let body: Stmt | undefined;
  for (const v of rule.expr.template.values as readonly ValueDef[]) {
    if (v.kind === "Function") fns.set(v.signature.name, { params: v.signature.parameters.map((p) => p.name), body: v.body });
    else if (v.kind === "Entry") body = v.entry.body;
  }
  if (body === undefined) {
    throw new Error(`derivedModes: rule for axis '${rule.axis}' has no Entry in its template`);
  }

  const declaredU32 = declaredToU32(rule, token);
  const readLeaf = (name: string): unknown => (name === "declared" ? declaredU32 : readUniform(name));

  // Mutually-recursive: an expr `Call` runs a function body; a function body
  // evaluates exprs via interpretExpr with `callFn` wired back here.
  const callFn = (name: string, args: EvalValue[]): EvalValue => {
    const fn = fns.get(name);
    if (fn === undefined) throw new Error(`derivedModes: rule calls unknown function '${name}'`);
    const locals = new Map<string, EvalValue>();
    fn.params.forEach((p, i) => { if (args[i] !== undefined) locals.set(p, args[i]!); });
    return evalBody(fn.body, locals);
  };
  const evalExpr = (e: Expr, locals: Map<string, EvalValue>): EvalValue =>
    interpretExpr(e, readLeaf, { locals, callFn });

  const run = (s: Stmt, locals: Map<string, EvalValue>): RunResult => {
    switch (s.kind) {
      case "Sequential":
      case "Isolated": {
        for (const c of s.body) { const r = run(c, locals); if (r !== NO_RETURN) return r; }
        return NO_RETURN;
      }
      case "Declare": {
        if (s.init !== undefined && s.init.kind === "Expr") locals.set(s.var.name, evalExpr(s.init.value, locals));
        return NO_RETURN;
      }
      case "Write": {
        if (s.target.kind === "LVar") locals.set(s.target.var.name, evalExpr(s.value, locals));
        return NO_RETURN;
      }
      case "If": {
        if (truthy(evalExpr(s.cond, locals))) return run(s.then, locals);
        return s.else !== undefined ? run(s.else, locals) : NO_RETURN;
      }
      case "ReturnValue": return evalExpr(s.value, locals);
      case "Expression":
      case "Nop":
      case "Return":
        return NO_RETURN;
      default:
        throw new Error(`derivedModes: unsupported statement '${s.kind}' in rule body`);
    }
  };
  const evalBody = (b: Stmt, locals: Map<string, EvalValue>): EvalValue => {
    const r = run(b, locals);
    if (r === NO_RETURN) throw new Error(`derivedModes: rule body for axis '${rule.axis}' produced no return value`);
    return r;
  };

  const raw = evalBody(body, new Map());
  return resolveAxisValue(rule, raw) as ModeValue<A>;
}
