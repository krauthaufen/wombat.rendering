import { describe, it, expect } from "vitest";
import type { Expr, Type } from "@aardworx/wombat.shader/ir";
import { ruleFromIR, uniformRef } from "../packages/rendering/src/runtime/derivedUniforms/rule.js";
import { DerivedUniformRegistry } from "../packages/rendering/src/runtime/derivedUniforms/registry.js";
import { buildUberKernel } from "../packages/rendering/src/runtime/derivedUniforms/codegen.js";

const Tf32: Type = { kind: "Float", width: 32 };
const Tmat4: Type = { kind: "Matrix", element: Tf32, rows: 4, cols: 4 };
const Tmat3: Type = { kind: "Matrix", element: Tf32, rows: 3, cols: 3 };
const mul4 = (l: Expr, r: Expr): Expr => ({ kind: "MulMatMat", lhs: l, rhs: r, type: Tmat4 });
const inv4 = (v: Expr): Expr => ({ kind: "Inverse", value: v, type: Tmat4 });
const u = (n: string) => uniformRef(n, Tmat4);

describe("derived-uniforms: codegen", () => {
  it("emits a collapse arm for a single-constituent rule", () => {
    const reg = new DerivedUniformRegistry();
    const id = reg.register(ruleFromIR(u("Model"))); // identity transform → collapse
    const { wgsl, strideU32 } = buildUberKernel(reg, 5);
    expect(strideU32).toBe(5);
    expect(wgsl).toContain(`fn arm_${id}(in0: u32, out_byte: u32)`);
    expect(wgsl).toContain(`case ${id}u: { arm_${id}(RecordData[base + 2u], out_byte); }`);
    expect(wgsl).toContain("const RECORD_STRIDE: u32 = 5u;");
    expect(wgsl).not.toContain("fn df_mul("); // no df32 lib needed
  });

  it("emits a 2-matmul chain arm with the df32 lib", () => {
    const reg = new DerivedUniformRegistry();
    const id = reg.register(ruleFromIR(mul4(u("View"), u("Model"))));
    const { wgsl } = buildUberKernel(reg, 5);
    expect(wgsl).toContain(`fn arm_${id}(in0: u32, in1: u32, out_byte: u32)`);
    expect(wgsl).toContain(`case ${id}u: { arm_${id}(RecordData[base + 2u], RecordData[base + 3u], out_byte); }`);
    expect(wgsl).toContain("fn df_mul(");
    expect(wgsl).toContain("fn df_add(");
  });

  it("emits a 3-matmul chain arm", () => {
    const reg = new DerivedUniformRegistry();
    const id = reg.register(ruleFromIR(mul4(mul4(u("Proj"), u("View")), u("Model"))));
    const { wgsl } = buildUberKernel(reg, 5);
    expect(wgsl).toContain(`fn arm_${id}(in0: u32, in1: u32, in2: u32, out_byte: u32)`);
  });

  it("treats Inverse(constituent) as just another mat4 leaf", () => {
    const reg = new DerivedUniformRegistry();
    const id = reg.register(ruleFromIR(mul4(inv4(u("Model")), inv4(u("View")))));
    const { wgsl } = buildUberKernel(reg, 5);
    expect(wgsl).toContain(`fn arm_${id}(in0: u32, in1: u32, out_byte: u32)`);
  });

  it("emits the normal-matrix arm for a mat4→mat3 rule", () => {
    const reg = new DerivedUniformRegistry();
    const ir: Expr = { kind: "Transpose", value: inv4(u("Model")), type: Tmat4 };
    // declared output mat3 (codegen takes upper-3×3 of the transposed input)
    const id = reg.register({ outputType: Tmat3, ir: { ...ir, type: Tmat3 } as Expr, hash: "nm-test" });
    const { wgsl } = buildUberKernel(reg, 5);
    expect(wgsl).toContain(`fn arm_${id}(in0: u32, out_byte: u32)`);
    expect(wgsl).toContain("write_mat3_entry");
  });

  it("throws on an unsupported rule shape", () => {
    const reg = new DerivedUniformRegistry();
    // vec4 * f32 — not a recognised v0 shape
    const Tvec4: Type = { kind: "Vector", element: Tf32, dim: 4 };
    const bad: Expr = { kind: "Mul", lhs: uniformRef("Color", Tvec4), rhs: uniformRef("Alpha", Tf32), type: Tvec4 };
    reg.register(ruleFromIR(bad));
    expect(() => buildUberKernel(reg, 5)).toThrow(/unsupported/);
  });
});
