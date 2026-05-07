// Unit tests for `compileHeapFragment` — runs the wombat.shader
// emit on a small DSL fragment and checks the output WGSL has the
// shape `buildHeapScene` expects: bare `@fragment fn fs(in: VsOut)
// -> @location(0) vec4<f32>`, no UBO/struct boilerplate.
//
// Effects are built via the test-helper `makeEffect` (the same
// shape every other rendering test uses) rather than the inline
// `fragment(...)` markers — vitest doesn't run the wombat-shader
// vite plugin, so the markers can't be processed at runtime.

import { describe, expect, it } from "vitest";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import { compileHeapFragment } from "@aardworx/wombat.rendering/runtime";
import { makeEffect } from "./_makeEffect.js";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

function flatColorEffect(): ReturnType<typeof makeEffect> {
  return makeEffect(
    `
      function fs(input:{ color: V4f }): { outColor: V4f } {
        return { outColor: input.color };
      }
    `,
    [{
      name: "fs", stage: "fragment",
      inputs:  [{ name: "color",    type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 2 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    }],
  );
}

function lambertEffect(): ReturnType<typeof makeEffect> {
  return makeEffect(
    `
      function fs(input:{ worldPos: V3f; normal: V3f; color: V4f; lightLoc: V3f }): { outColor: V4f } {
        const n = input.normal.normalize();
        const l = input.lightLoc.sub(input.worldPos).normalize();
        const k = 0.2 + 0.8 * abs(l.dot(n));
        return { outColor: new V4f(input.color.xyz.mul(k), input.color.w) };
      }
    `,
    [{
      name: "fs", stage: "fragment",
      inputs: [
        { name: "worldPos", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "normal",   type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
        { name: "color",    type: Tvec4f, semantic: "Color",    decorations: [{ kind: "Location", value: 2 }] },
        { name: "lightLoc", type: Tvec3f, semantic: "Generic",  decorations: [{ kind: "Location", value: 3 }] },
      ],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    }],
  );
}

describe("compileHeapFragment", () => {
  it("flat-color FS emits @location(0) vec4<f32> directly", () => {
    const wgsl = compileHeapFragment(flatColorEffect());
    expect(wgsl).toMatch(/@fragment\s+fn\s+fs\s*\(\s*in\s*:\s*VsOut\s*\)\s*->\s*@location\(0\)\s+vec4<f32>/);
    expect(wgsl).toMatch(/return\s+in\.color\s*;/);
    expect(wgsl).not.toMatch(/Wombat_fragment_\w+Input/);
    expect(wgsl).not.toMatch(/Wombat_fragment_\w+Output/);
    expect(wgsl).not.toMatch(/_UB_uniform/);
  });

  it("lambert FS uses VsOut fields directly, no UBO", () => {
    const wgsl = compileHeapFragment(lambertEffect());
    expect(wgsl).toMatch(/in\.normal/);
    expect(wgsl).toMatch(/in\.worldPos/);
    expect(wgsl).toMatch(/in\.lightLoc/);
    expect(wgsl).toMatch(/in\.color/);
    expect(wgsl).toMatch(/return\s+vec4<f32>/);
    expect(wgsl).not.toMatch(/_w_uniform/);
  });
});
