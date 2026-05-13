// derivedMode — branded wrapper around a shader-IR RuleExpr.

import { describe, expect, test } from "vitest";
import {
  derivedMode, isDerivedModeRule,
  type DerivedModeRule,
} from "@aardworx/wombat.rendering/runtime";
import type { RuleExpr } from "@aardworx/wombat.shader";

function fakeRuleExpr(): RuleExpr<number> {
  // Minimal stand-in for what `rule(...)` would emit at build time.
  return {
    id: "test-rule",
    template: { types: [], values: [] },
    holes: {},
    avalHoles: {},
    dumpIR: () => "// fake rule",
  };
}

describe("derivedMode (RuleExpr-based)", () => {
  test("brands the rule + carries axis + expr (declared filled in by SG)", () => {
    const expr = fakeRuleExpr();
    const r = derivedMode("cull", expr);
    expect(r.__derivedModeRule).toBe(true);
    expect(r.axis).toBe("cull");
    expect(r.expr).toBe(expr);
    // `declared` is undefined until the SG traversal wraps the rule
    // with the surrounding-scope value for this axis.
    expect(r.declared).toBeUndefined();
    expect(isDerivedModeRule(r)).toBe(true);
    expect(isDerivedModeRule({})).toBe(false);
  });

  test("type-stable: DerivedModeRule<\"cull\">", () => {
    const r: DerivedModeRule<"cull"> = derivedMode("cull", fakeRuleExpr());
    expect(r.axis).toBe("cull");
  });

  test("works for every axis (no options arg needed)", () => {
    derivedMode("frontFace",      fakeRuleExpr());
    derivedMode("topology",       fakeRuleExpr());
    derivedMode("depthCompare",   fakeRuleExpr());
    derivedMode("depthWrite",     fakeRuleExpr());
    derivedMode("alphaToCoverage", fakeRuleExpr());
    derivedMode("blend",           fakeRuleExpr());
    expect(true).toBe(true);
  });
});
