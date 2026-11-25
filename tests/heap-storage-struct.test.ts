// Heap-managed storage buffers with STRUCT elements.
//
// The inline-marker frontend cannot declare a struct-typed storage array
// (its type mapper knows scalars, vectors, matrices and arrays), so this
// builds the effect from source + an explicit StorageBuffer declaration
// and inspects the WGSL the heap rewrite emits.

import { describe, expect, it } from "vitest";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import { compileHeapEffectIR } from "../packages/rendering/src/runtime/heapEffectIR.js";
import { buildShaderFamily } from "../packages/rendering/src/runtime/heapShaderFamily.js";
import { storageLayoutOf, storageStrideOf, storageFieldOffset } from "../packages/rendering/src/runtime/heapIrBuilders.js";
import { Lamp, makeLampEffect } from "./_heapStorageStructEffect.js";

const Tvec3f: Type = Vec(Tf32, 3);

describe("heap-managed storage buffers: struct elements", () => {
  it("lays a struct out by WGSL's rules", () => {
    expect(storageLayoutOf(Lamp)).toEqual({ size: 32, align: 16 });
    expect(storageStrideOf(Lamp)).toBe(32);
    expect(storageFieldOffset(Lamp, "Position").offset).toBe(0);
    expect(storageFieldOffset(Lamp, "Strength").offset).toBe(12);
    expect(storageFieldOffset(Lamp, "Tint").offset).toBe(16);
    // vec3 then vec3: the second is pushed to its 16-byte alignment
    const two: Type = { kind: "Struct", name: "Two", fields: [{ name: "a", type: Tvec3f }, { name: "b", type: Tvec3f }] } as Type;
    expect(storageFieldOffset(two, "b").offset).toBe(16);
    expect(storageLayoutOf(two)).toEqual({ size: 32, align: 16 });
  });

  it("reads a whole struct element and a single member from the arena", () => {
    const effect = makeLampEffect();
    const fol = { locations: new Map([["outColor", 0]]) };
    const fam = buildShaderFamily([effect], fol, undefined, {
      perEffectPerInstance: new Map([[effect, { attributes: new Set<string>(), storage: new Set(["Lamps"]) }]]),
    });
    expect(fam.drawHeaderUnion.drawHeaderFields.find(f => f.name === "Lamps")?.kind).toBe("storage-ref");
    const ir = compileHeapEffectIR(effect, fam.drawHeaderUnion, { target: "wgsl", fragmentOutputLayout: fol }, "standalone");
    // the buffer has no binding of its own any more
    expect(ir.fs).not.toMatch(/var<storage[^>]*>\s*Lamps/);
    expect(ir.vs).not.toMatch(/var<storage[^>]*>\s*Lamps/);
    // the whole element is built from its members by the struct's constructor:
    // Position from three floats, Strength 12 bytes in, Tint through the vec4 view 16 bytes in
    expect(ir.fs).toMatch(/let lamp: Lamp = Lamp\(vec3<f32>\(heapF32\[.*\+ 12u\) \/ 4u\)\], heapV4f\[.*\+ 16u\) \/ 16u\)\]\);/);
    // a single member is ONE load (16-byte allocation header + offset 12), not the whole struct
    expect(ir.fs).toMatch(/let dim: f32 = heapF32\[.*\+ 28u\)/);
    expect(ir.fs.match(/Lamp\(/g)!.length).toBe(1);
  });
});
