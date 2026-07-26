// Standalone (non-atlas) texture path — BGL ↔ shader binding agreement.
//
// A Tier-L / non-atlas texture keeps its OWN texture + sampler decls in the
// emitted WGSL (unlike the atlas path, which strips them and routes through
// the drawHeader). `compileModule` assigns those decls' `@binding` numbers,
// and the runtime builds the bind-group layout from `buildBucketLayout`'s
// `textureBindings` / `samplerBindings`. If the two disagree on which slot is
// the texture vs the sampler, WebGPU rejects the pipeline with
// "Binding type in the shader (sampler) doesn't match the type in the layout
// (texture)". An invalid pipeline poisons the WHOLE shared render pass — every
// co-resident heap/atlas draw silently vanishes. This test locks the two in
// agreement so the standalone path both draws itself and doesn't corrupt the
// shared megacall.

import { describe, expect, it } from "vitest";
import { compileHeapEffect, buildBucketLayout } from "../packages/rendering/src/runtime/heapEffect.js";
import { compileHeapEffectIR } from "../packages/rendering/src/runtime/heapEffectIR.js";
import { makeHeapTestEffectTextured } from "./_heapTestEffect.js";

/** Parse `@group(0) @binding(N) var NAME: TYPE;` decls out of a WGSL source. */
function bindingDecls(wgsl: string): Map<number, { name: string; type: string }> {
  const out = new Map<number, { name: string; type: string }>();
  const re = /@group\(0\)\s*@binding\((\d+)\)\s*var\s+(\w+)\s*:\s*([a-zA-Z_][\w<>,\s]*?)\s*;/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(wgsl)) !== null) {
    out.set(Number(m[1]), { name: m[2]!, type: m[3]!.trim() });
  }
  return out;
}

const isTexture = (t: string) => t.startsWith("texture_");
const isSampler = (t: string) => t === "sampler" || t === "sampler_comparison";

describe("heap standalone texture binding agreement", () => {
  it("routes a plain 2D texture as standalone (no atlas) — non-empty texture+sampler bindings", () => {
    const { schema } = compileHeapEffect(makeHeapTestEffectTextured());
    // No `atlasTextureBindings` opt → the texture is NOT atlas-routed → it
    // keeps a real texture_2d + sampler binding (the standalone shape).
    const layout = buildBucketLayout(schema, true);
    expect(layout.textureBindings.length).toBe(1);
    expect(layout.samplerBindings.length).toBe(1);
  });

  it("BGL binding numbers + types match the emitted WGSL decls", () => {
    const effect = makeHeapTestEffectTextured();
    const { schema } = compileHeapEffect(effect);
    const layout = buildBucketLayout(schema, true);
    const ir = compileHeapEffectIR(effect, layout, { target: "wgsl" });
    const decls = bindingDecls(ir.fs);

    // Every texture binding the BGL declares must be a texture in the shader
    // at the SAME binding number, with the same name.
    for (const t of layout.textureBindings) {
      const d = decls.get(t.binding);
      expect(d, `no WGSL decl at texture binding ${t.binding}`).toBeDefined();
      expect(isTexture(d!.type), `binding ${t.binding} is '${d!.type}', expected a texture`).toBe(true);
      expect(d!.name).toBe(t.name);
    }
    // Same for samplers.
    for (const s of layout.samplerBindings) {
      const d = decls.get(s.binding);
      expect(d, `no WGSL decl at sampler binding ${s.binding}`).toBeDefined();
      expect(isSampler(d!.type), `binding ${s.binding} is '${d!.type}', expected a sampler`).toBe(true);
      expect(d!.name).toBe(s.name);
    }
  });

  it("emits the sampler in the lower slot (matches wombat.shader's split order)", () => {
    // Regression guard on the fix: for the single sampler2d pair, the sampler
    // takes the lower @binding and the texture the next. If buildBucketLayout
    // ever flips back to texture-first, this fails before it reaches the GPU.
    const effect = makeHeapTestEffectTextured();
    const { schema } = compileHeapEffect(effect);
    const layout = buildBucketLayout(schema, true);
    expect(layout.samplerBindings[0]!.binding).toBeLessThan(layout.textureBindings[0]!.binding);
  });
});
