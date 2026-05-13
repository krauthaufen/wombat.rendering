// Phase 5c.3 — partition kernel codegen from mode-rule IR.

import { describe, expect, test } from "vitest";
import {
  derivedMode, pickEnum, DerivedExpr,
  emitPartitionKernel, packModeKey,
} from "@aardworx/wombat.rendering/runtime";

describe("partition kernel codegen", () => {
  test("packModeKey is the mixed-radix combiner the kernel uses", () => {
    expect(packModeKey([0, 0], [3, 2])).toBe(0);
    expect(packModeKey([1, 0], [3, 2])).toBe(1);
    expect(packModeKey([2, 0], [3, 2])).toBe(2);
    expect(packModeKey([0, 1], [3, 2])).toBe(3);
    expect(packModeKey([2, 1], [3, 2])).toBe(5);
  });

  test("single-axis cull rule emits clear + partition + the right number of slot bindings", () => {
    const rule = derivedMode("cull", (u, declared) => {
      const det = u.ModelTrafo.upperLeft3x3().determinant();
      const flipped = pickEnum(declared,
        DerivedExpr.u32(0), DerivedExpr.u32(2), DerivedExpr.u32(1),
      );
      return flipped.select(declared, det.lt(DerivedExpr.f32(0)));
    }, {
      domain: ["none", "front", "back"],
      declared: "back",
    });
    const wgsl = emitPartitionKernel({
      rules: [{ axis: "cull", ir: rule.ir, inputUniforms: rule.inputUniforms, domainSize: 3 }],
      totalSlots: 3,
    });
    // Has both entry points
    expect(wgsl).toMatch(/fn\s+clear\s*\(/);
    expect(wgsl).toMatch(/fn\s+partitionRecords\s*\(/);
    // params has the declared field
    expect(wgsl).toMatch(/decl_cull:\s*u32/);
    // 3 slot draw tables + 3 slot count atomics
    expect(wgsl).toMatch(/slot0DrawTable/);
    expect(wgsl).toMatch(/slot1DrawTable/);
    expect(wgsl).toMatch(/slot2DrawTable/);
    expect(wgsl).toMatch(/slot0Count/);
    expect(wgsl).toMatch(/slot2Count/);
    // The rule's WGSL body lowers determinant + select
    expect(wgsl).toMatch(/determinant/);
    expect(wgsl).toMatch(/select\(/);
    // Uniform input lowers to the arena loader
    expect(wgsl).toMatch(/load_mat3_upper\(r\.modelRef\)/);
    // The dispatch switch covers slot 0..2
    expect(wgsl).toMatch(/case\s+0u\s*:/);
    expect(wgsl).toMatch(/case\s+2u\s*:/);
  });

  test("emits scan-format (5-u32) records into slot tables", () => {
    const rule = derivedMode("cull", (_u, declared) => declared, {
      domain: ["none", "back"],
      declared: "back",
    });
    const wgsl = emitPartitionKernel({
      rules: [{ axis: "cull", ir: rule.ir, inputUniforms: rule.inputUniforms, domainSize: 2 }],
      totalSlots: 2,
    });
    expect(wgsl).toMatch(/SCAN_REC_U32:\s*u32\s*=\s*5u/);
    expect(wgsl).toMatch(/slot0DrawTable\[base \+ 4u\]\s*=\s*r\.instanceCount/);
  });
});
