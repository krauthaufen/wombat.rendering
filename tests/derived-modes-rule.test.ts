// Phase 5c.3 — IR-traced mode rules.

import { describe, expect, test } from "vitest";
import {
  derivedMode, isDerivedModeRule, pickEnum, DerivedExpr,
} from "@aardworx/wombat.rendering/runtime";

describe("derivedMode (IR-traced)", () => {
  test("derivedMode brands the rule and captures axis/domain/declared", () => {
    const rule = derivedMode("cull", (_u, declared) => declared, {
      domain: ["none", "front", "back"],
      declared: "back",
    });
    expect(rule.__derivedModeRule).toBe(true);
    expect(rule.axis).toBe("cull");
    expect(rule.domain).toEqual(["none", "front", "back"]);
    expect(rule.declared).toBe("back");
    expect(isDerivedModeRule(rule)).toBe(true);
    expect(isDerivedModeRule({})).toBe(false);
  });

  test("the builder traces a u32-typed IR", () => {
    const rule = derivedMode("cull", (_u, declared) => declared, {
      domain: ["none", "front", "back"],
      declared: "back",
    });
    expect(rule.ir.type.kind).toBe("Int");
    expect((rule.ir.type as { signed: boolean }).signed).toBe(false);
  });

  test("body that doesn't return u32 throws", () => {
    expect(() =>
      derivedMode("cull",
        (_u) => DerivedExpr.f32(1.0),   // f32, not u32
        { domain: ["none"], declared: "none" } as never,
      ),
    ).toThrow(/must return a u32/);
  });

  test("inputUniforms accumulates leaf names visited by the builder", () => {
    const rule = derivedMode("cull", (u, declared) => {
      // Touch ModelTrafo via .upperLeft3x3().determinant() and combine
      // with declared so the body type is u32.
      const det = u.ModelTrafo.upperLeft3x3().determinant();
      const flipped = pickEnum(declared,
        DerivedExpr.u32(0), DerivedExpr.u32(2), DerivedExpr.u32(1),
      );
      return flipped.select(declared, det.lt(DerivedExpr.f32(0)));
    }, {
      domain: ["none", "front", "back"],
      declared: "back",
    });
    expect(rule.inputUniforms).toContain("ModelTrafo");
  });

  test("pickEnum chains selects in domain order", () => {
    // Build a tiny rule whose body uses pickEnum with three cases.
    const rule = derivedMode("cull", (_u, declared) =>
      pickEnum(declared,
        DerivedExpr.u32(0),
        DerivedExpr.u32(2),
        DerivedExpr.u32(1),
      ),
    {
      domain: ["none", "front", "back"],
      declared: "back",
    });
    expect(rule.ir.kind).toBe("Conditional");
  });
});
