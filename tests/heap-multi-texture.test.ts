// Several textures on one draw. An effect may sample any number of
// textures; on the heap every one of them is an atlas lookup, and the draw
// header carries a placement PER TEXTURE BINDING (page, origin, size, mips,
// sampler state). Nothing about a texture's role is known to the heap.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap, cset, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3f, V4f } from "@aardworx/wombat.base";
import {
  BufferView, ElementType, ISampler, ITexture, PipelineState, RenderTree, asAttributeProvider, asUniformProvider,
  type DrawCall, type RenderObject,
} from "@aardworx/wombat.rendering/core";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { compileHybridScene } from "../packages/rendering/src/runtime/hybridScene.js";
import { isHeapEligible } from "../packages/rendering/src/runtime/heapEligibility.js";
import { MockGPU } from "./_mockGpu.js";
import { makeHeapTestEffectThreeTextures } from "./_heapTestEffect.js";

const g = globalThis as Record<string, unknown>;
g.GPUTextureUsage ??= { COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04, STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10 };
g.GPUBufferUsage ??= {
  MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008, INDEX: 0x0010, VERTEX: 0x0020,
  UNIFORM: 0x0040, STORAGE: 0x0080, INDIRECT: 0x0100, QUERY_RESOLVE: 0x0200,
};
g.GPUShaderStage ??= { VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 };

const top = AdaptiveToken.top;
const effect = makeHeapTestEffectThreeTextures();
const sig = () => createFramebufferSignature({ colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
const state = PipelineState.constant({ rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" }, depth: { write: true, compare: "less" } });
const texture = (w: number, h: number, format: GPUTextureFormat = "rgba8unorm"): aval<ITexture> =>
  cval(ITexture.fromRaw({ data: new Uint8Array(w * h * 4), width: w, height: h, format }));
const clamp = AVal.constant(ISampler.fromDescriptor({ magFilter: "linear", minFilter: "linear" }));
const repeat = AVal.constant(ISampler.fromDescriptor({ magFilter: "nearest", minFilter: "nearest", addressModeU: "repeat", addressModeV: "repeat" }));

/** An object sampling `checker`, `detail` and `mask` (named as a scenegraph binds them: `X` and `X_view`). */
function object(tex: { checker: aval<ITexture>; detail: aval<ITexture>; mask: aval<ITexture> }, i = 0): RenderObject {
  let textures = HashMap.empty<string, aval<ITexture>>();
  for (const [name, t] of Object.entries(tex)) textures = textures.add(name, t).add(`${name}_view`, t);
  return {
    effect, pipelineState: state,
    vertexAttributes: asAttributeProvider(HashMap.empty<string, BufferView>()
      .add("Positions", BufferView.ofArray(new Float32Array(12).fill(i + 1), { elementType: ElementType.V4f }))
      .add("Normals", BufferView.ofArray(new Float32Array(9).fill(0.5), { elementType: ElementType.V3f }))),
    uniforms: asUniformProvider(HashMap.empty<string, aval<unknown>>()
      .add("ModelTrafo", AVal.constant(Trafo3d.identity) as aval<unknown>)
      .add("ViewProjTrafo", AVal.constant(Trafo3d.identity) as aval<unknown>)
      .add("LightLocation", AVal.constant(new V3f(0, 0, 1)) as aval<unknown>)
      .add("Color", AVal.constant(new V4f(1, 1, 1, 1)) as aval<unknown>)),
    textures,
    samplers: HashMap.empty<string, aval<ISampler>>().add("checker", clamp).add("detail", repeat).add("mask", clamp),
    indices: BufferView.ofArray(new Uint32Array([0, 1, 2]), { elementType: ElementType.U32 }),
    drawCall: AVal.constant<DrawCall>({ kind: "indexed", indexCount: 3, instanceCount: 1, firstIndex: 0, baseVertex: 0, firstInstance: 0 }),
  };
}

interface Placement { pageRef: number; formatBits: number; origin: [number, number]; size: [number, number] }
/** What a draw's header says about each texture binding. */
function placements(scene: ReturnType<typeof compileHybridScene>, draw = 0): Map<string, Placement> {
  type Field = { byteOffset: number; kind: string; textureSub?: string; textureBindingName?: string };
  const heap = scene.heapSceneForTest as unknown as {
    _debug: { bucketsForTest(): readonly { layout: { drawHeaderFields: readonly Field[]; drawHeaderBytes: number }; drawHeaderStaging: Float32Array }[] };
  };
  const bucket = heap._debug.bucketsForTest()[0]!;
  const f32 = bucket.drawHeaderStaging, u32 = new Uint32Array(f32.buffer, f32.byteOffset, f32.length);
  const out = new Map<string, Placement>();
  for (const f of bucket.layout.drawHeaderFields) {
    if (f.kind !== "texture-ref") continue;
    const at = (draw * bucket.layout.drawHeaderBytes + f.byteOffset) / 4;
    const p = out.get(f.textureBindingName!) ?? { pageRef: -1, formatBits: -1, origin: [0, 0], size: [0, 0] };
    if (f.textureSub === "pageRef") p.pageRef = u32[at]!;
    else if (f.textureSub === "formatBits") p.formatBits = u32[at]!;
    else if (f.textureSub === "origin") p.origin = [f32[at]!, f32[at + 1]!];
    else if (f.textureSub === "size") p.size = [f32[at]!, f32[at + 1]!];
    out.set(f.textureBindingName!, p);
  }
  return out;
}
const binding = (m: Map<string, Placement>, name: string): Placement => {
  const hit = [...m].find(([k]) => k === name || k === `${name}_view`);
  if (hit === undefined) throw new Error(`no texture binding ${name} in ${[...m.keys()].join(", ")}`);
  return hit[1];
};

describe("several textures on one heap draw", () => {
  it("an object with three textures is heap-served, and each binding gets its own placement", () => {
    const gpu = new MockGPU();
    const tex = { checker: texture(64, 64), detail: texture(32, 16), mask: texture(128, 8, "rgba8unorm-srgb") };
    const ro = object(tex);
    expect(isHeapEligible(ro).getValue(top)).toBe(true);
    const scene = compileHybridScene(gpu.device, sig(), RenderTree.leaf(ro), {});
    scene.update(top);
    expect(scene.heapSceneForTest.stats.totalDraws).toBe(1);
    const p = placements(scene);
    expect(p.size).toBe(3);
    // the size is the texture's own, per binding
    expect(binding(p, "checker").size).toEqual([64, 64]);
    expect(binding(p, "detail").size).toEqual([32, 16]);
    expect(binding(p, "mask").size).toEqual([128, 8]);
    // two textures on the same page do not lie on top of each other
    expect(binding(p, "checker").origin).not.toEqual(binding(p, "detail").origin);
    // each has its format (bit 0: 0 linear, 1 sRGB) and its OWN sampler state (bits 4-5: wrap u; 1 = repeat)
    expect(binding(p, "checker").formatBits & 1).toBe(0);
    expect(binding(p, "mask").formatBits & 1).toBe(1);
    expect((binding(p, "checker").formatBits >> 4) & 3).toBe(0);
    expect((binding(p, "detail").formatBits >> 4) & 3).toBe(1);
  });

  it("objects sharing some textures share their atlas entries and one bucket", () => {
    const gpu = new MockGPU();
    const colour = texture(64, 64), maskA = texture(16, 16), maskB = texture(16, 16), detail = texture(32, 32);
    const set = cset<RenderTree>([
      RenderTree.leaf(object({ checker: colour, detail, mask: maskA }, 0)),
      RenderTree.leaf(object({ checker: colour, detail, mask: maskB }, 1)),
    ]);
    const scene = compileHybridScene(gpu.device, sig(), RenderTree.unorderedFromSet(set), {});
    scene.update(top);
    expect(scene.heapSceneForTest.stats.totalDraws).toBe(2);
    expect(scene.heapSceneForTest.stats.groups).toBe(1);
    const a = placements(scene, 0), b = placements(scene, 1);
    expect(binding(a, "checker")).toEqual(binding(b, "checker"));
    expect(binding(a, "detail")).toEqual(binding(b, "detail"));
    expect(binding(a, "mask").origin).not.toEqual(binding(b, "mask").origin);
  });

  it("changing ONE of an object's textures rewrites that binding only", () => {
    const gpu = new MockGPU();
    const mask = cval(ITexture.fromRaw({ data: new Uint8Array(16 * 16 * 4), width: 16, height: 16, format: "rgba8unorm" }));
    const scene = compileHybridScene(gpu.device, sig(), RenderTree.leaf(object({ checker: texture(64, 64), detail: texture(32, 32), mask })), {});
    scene.update(top);
    const before = placements(scene);
    transact(() => { mask.value = ITexture.fromRaw({ data: new Uint8Array(48 * 24 * 4), width: 48, height: 24, format: "rgba8unorm" }); });
    scene.update(top);
    const after = placements(scene);
    expect(binding(after, "mask").size).toEqual([48, 24]);
    expect(binding(after, "checker")).toEqual(binding(before, "checker"));
    expect(binding(after, "detail")).toEqual(binding(before, "detail"));
  });

  it("removing the object gives every one of its atlas entries back", () => {
    const gpu = new MockGPU();
    const tree = RenderTree.leaf(object({ checker: texture(64, 64), detail: texture(32, 32), mask: texture(16, 16) }));
    const set = cset<RenderTree>([tree]);
    const scene = compileHybridScene(gpu.device, sig(), RenderTree.unorderedFromSet(set), {});
    scene.update(top);
    const pool = (globalThis as unknown as { __atlasDebug: { evictIdle(): number } }).__atlasDebug;
    expect(pool.evictIdle()).toBe(0);            // all three in use
    transact(() => set.remove(tree));
    scene.update(top);
    expect(scene.heapSceneForTest.stats.totalDraws).toBe(0);
    expect(pool.evictIdle()).toBe(3);            // all three idle now, and freed
  });
});
