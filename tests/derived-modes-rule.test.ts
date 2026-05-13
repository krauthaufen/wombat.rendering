import { describe, test, expect } from "vitest";
import {
  derivedMode,
  isDerivedModeRule,
  flipCull,
} from "@aardworx/wombat.rendering/runtime";

describe("derivedModes/rule", () => {
  test("derivedMode builds a tagged rule with axis + evaluate", () => {
    const rule = derivedMode("cull", (_u, declared) => declared);
    expect(rule.axis).toBe("cull");
    expect(rule.__derivedModeRule).toBe(true);
    expect(typeof rule.evaluate).toBe("function");
    expect(rule.domain).toBeUndefined();
  });

  test("isDerivedModeRule distinguishes rules from plain objects", () => {
    expect(isDerivedModeRule(derivedMode("cull", (_u, d) => d))).toBe(true);
    expect(isDerivedModeRule({})).toBe(false);
    expect(isDerivedModeRule(null)).toBe(false);
    expect(isDerivedModeRule("back")).toBe(false);
  });

  test("evaluate runs the closure against a uniforms proxy", () => {
    const rule = derivedMode("cull", (u, declared) =>
      (u.Side as number) > 0 ? declared : flipCull(declared));
    expect(rule.evaluate({ Side: 1 }, "back")).toBe("back");
    expect(rule.evaluate({ Side: -1 }, "back")).toBe("front");
    expect(rule.evaluate({ Side: -1 }, "none")).toBe("none");
  });

  test("flipCull swaps back ↔ front, passes 'none' through", () => {
    expect(flipCull("back")).toBe("front");
    expect(flipCull("front")).toBe("back");
    expect(flipCull("none")).toBe("none");
  });

  test("domain option is preserved for forward-compat enumeration", () => {
    const rule = derivedMode("cull", (_u, d) => d, { domain: ["back", "front"] });
    expect(rule.domain).toEqual(["back", "front"]);
  });

  test("determinant-flip example evaluates correctly", () => {
    // Identity matrix -> det = 1 -> declared.
    // -1 scale on x -> det = -1 -> flip.
    type M44 = { M00: number; M01: number; M02: number;
                 M10: number; M11: number; M12: number;
                 M20: number; M21: number; M22: number };
    const rule = derivedMode("cull", (u, declared) => {
      const m = u.ModelTrafo as M44;
      const det =
        m.M00 * (m.M11 * m.M22 - m.M12 * m.M21) -
        m.M01 * (m.M10 * m.M22 - m.M12 * m.M20) +
        m.M02 * (m.M10 * m.M21 - m.M11 * m.M20);
      return det < 0 ? flipCull(declared) : declared;
    });
    const identity: M44 = {
      M00: 1, M01: 0, M02: 0,
      M10: 0, M11: 1, M12: 0,
      M20: 0, M21: 0, M22: 1,
    };
    const mirroredX: M44 = { ...identity, M00: -1 };
    expect(rule.evaluate({ ModelTrafo: identity   }, "back")).toBe("back");
    expect(rule.evaluate({ ModelTrafo: mirroredX  }, "back")).toBe("front");
  });
});
