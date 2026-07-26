// Heap samplerCube → 2D face-grid rewrite (iOS-safe skybox path).
//
// A `samplerCube` binding must NOT reach the GPU as a real texture_cube
// (WebKit/Metal reject cube views over the atlas array). Instead the heap:
//   - buildSchema records `dim === "cube"` from the IR sampler target,
//   - buildBucketLayout derives `cubeTextureBindings` from the schema,
//   - compileHeapEffectIR retargets the sampler decl Cube→2D and rewrites
//     `textureSample(cube, dir)` → `textureSample(name, cubeProjectZup(dir))`,
//     prepending the cubeProjectZup helper to the FS.
// This test asserts the emitted WGSL binds texture_2d (never texture_cube)
// and carries the projection helper.

import { describe, expect, it } from "vitest";
import { compileHeapEffect, buildBucketLayout } from "../packages/rendering/src/runtime/heapEffect.js";
import { compileHeapEffectIR } from "../packages/rendering/src/runtime/heapEffectIR.js";
import { makeHeapTestEffectCube, makeHeapTestEffectTextured } from "./_heapTestEffect.js";

describe("heap samplerCube → 2D grid rewrite", () => {
  it("buildSchema marks the cube binding with dim === 'cube'", () => {
    const { schema } = compileHeapEffect(makeHeapTestEffectCube());
    expect(schema.textures.length).toBeGreaterThan(0);
    const cube = schema.textures.find(t => t.dim === "cube");
    expect(cube).toBeDefined();
    // The wgslType stays texture_2d — that IS what the heap binds.
    expect(cube!.wgslType).toBe("texture_2d<f32>");
  });

  it("a plain 2D texture is NOT marked cube", () => {
    const { schema } = compileHeapEffect(makeHeapTestEffectTextured());
    expect(schema.textures.every(t => t.dim !== "cube")).toBe(true);
  });

  it("buildBucketLayout derives cubeTextureBindings from the schema", () => {
    const { schema } = compileHeapEffect(makeHeapTestEffectCube());
    const layout = buildBucketLayout(schema, true);
    const cubeName = schema.textures.find(t => t.dim === "cube")!.name;
    expect(layout.cubeTextureBindings.has(cubeName)).toBe(true);
  });

  it("emitted FS binds texture_2d (never texture_cube) and calls cubeProjectZup", () => {
    const effect = makeHeapTestEffectCube();
    const { schema } = compileHeapEffect(effect);
    const layout = buildBucketLayout(schema, true);
    const ir = compileHeapEffectIR(effect, layout, { target: "wgsl" });
    // No real cube texture anywhere in the emitted shader.
    expect(ir.fs).not.toContain("texture_cube");
    expect(ir.vs).not.toContain("texture_cube");
    // The 2D grid binding + projection helper are present.
    expect(ir.fs).toContain("texture_2d");
    expect(ir.fs).toContain("cubeProjectZup");
    // The helper is a real fn definition, not just a dangling call.
    expect(ir.fs).toMatch(/fn\s+cubeProjectZup\s*\(/);
  });
});
