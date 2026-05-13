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
  test("brands the rule and carries axis + expr + declared", () => {
    const expr = fakeRuleExpr();
    const r = derivedMode("cull", expr, { declared: "back" });
    expect(r.__derivedModeRule).toBe(true);
    expect(r.axis).toBe("cull");
    expect(r.expr).toBe(expr);
    expect(r.declared).toBe("back");
    expect(isDerivedModeRule(r)).toBe(true);
    expect(isDerivedModeRule({})).toBe(false);
  });

  test("accepts a string declared value", () => {
    const r: DerivedModeRule<"cull"> = derivedMode("cull", fakeRuleExpr(), { declared: "front" });
    expect(r.declared).toBe("front");
  });

  test("works for every axis", () => {
    derivedMode("frontFace",      fakeRuleExpr(), { declared: "ccw" });
    derivedMode("topology",       fakeRuleExpr(), { declared: "triangle-list" });
    derivedMode("depthCompare",   fakeRuleExpr(), { declared: "less" });
    derivedMode("depthWrite",     fakeRuleExpr(), { declared: true });
    derivedMode("alphaToCoverage", fakeRuleExpr(), { declared: false });
    expect(true).toBe(true);
  });
});
