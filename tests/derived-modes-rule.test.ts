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

  test("blend axis: full AttachmentBlend object via `resolve` callback", () => {
    const STRAIGHT = {
      enabled: true,
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
      alpha: { srcFactor: "one",       dstFactor: "one-minus-src-alpha", operation: "add" },
      writeMask: 0xF,
    } as const;
    const PREMULT = {
      enabled: true,
      color: { srcFactor: "one",       dstFactor: "one-minus-src-alpha", operation: "add" },
      alpha: { srcFactor: "one",       dstFactor: "one-minus-src-alpha", operation: "add" },
      writeMask: 0xF,
    } as const;
    const r = derivedMode("blend", fakeRuleExpr(), {
      declared: 0,
      resolve: (i: number) => (i === 1 ? PREMULT : STRAIGHT),
    });
    expect(r.axis).toBe("blend");
    expect(r.resolve).toBeDefined();
    expect(r.resolve!(0)).toBe(STRAIGHT);
    expect(r.resolve!(1)).toBe(PREMULT);
  });

  test("blend axis: `values` array shorthand desugars to `resolve`", () => {
    const STRAIGHT = {
      enabled: true,
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
      alpha: { srcFactor: "one",       dstFactor: "one-minus-src-alpha", operation: "add" },
      writeMask: 0xF,
    } as const;
    const PREMULT = {
      enabled: true,
      color: { srcFactor: "one",       dstFactor: "one-minus-src-alpha", operation: "add" },
      alpha: { srcFactor: "one",       dstFactor: "one-minus-src-alpha", operation: "add" },
      writeMask: 0xF,
    } as const;
    const r = derivedMode("blend", fakeRuleExpr(), {
      declared: 0,
      values: [STRAIGHT, PREMULT],
    });
    expect(r.resolve!(0)).toBe(STRAIGHT);
    expect(r.resolve!(1)).toBe(PREMULT);
    // Out-of-range u32 errors cleanly.
    expect(() => r.resolve!(2)).toThrow(/u32=2.+only has 2 entries/);
  });
});
