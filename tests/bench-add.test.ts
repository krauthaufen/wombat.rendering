// What it costs to bring a render object into a heap scene, and to take
// it out again. Not a test of behaviour: run with WOMBAT_BENCH=1 to print
// the figures (it is skipped otherwise).
//
// The scene is shaped like a crowd built from parts: a few meshes, one
// shader, one pipeline state, and per object only a row of a uniform
// table (where it is, its colour) — objects that differ in nothing the
// renderer has to look at twice.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap, cset, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3f } from "@aardworx/wombat.base";
import {
  BufferView, ElementType, PipelineState, RenderTree, asAttributeProvider, asUniformProvider,
  type DrawCall, type RenderObject,
} from "@aardworx/wombat.rendering/core";
import { uniformTable } from "../packages/rendering/src/core/uniformColumn.js";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { compileHybridScene } from "../packages/rendering/src/runtime/hybridScene.js";
import { MockGPU } from "./_mockGpu.js";
import { makeHeapTestEffect } from "./_heapTestEffect.js";

const g = globalThis as Record<string, unknown>;
g.GPUTextureUsage ??= { COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04, STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10 };
g.GPUBufferUsage ??= {
  MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008, INDEX: 0x0010, VERTEX: 0x0020,
  UNIFORM: 0x0040, STORAGE: 0x0080, INDIRECT: 0x0100, QUERY_RESOLVE: 0x0200,
};
g.GPUShaderStage ??= { VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 };

const N = 12000, MESHES = 12;
const FIELDS = { ModelTrafo: "mat4x4<f32>", Color: "vec4<f32>" } as const;

function world() {
  const gpu = new MockGPU();
  const effect = makeHeapTestEffect();
  const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
  const state = PipelineState.constant({
    rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" },
    depth: { write: true, compare: "less" },
  });
  const meshes = Array.from({ length: MESHES }, (_, m) => {
    const verts = 300 + m * 10;
    return {
      attributes: asAttributeProvider(HashMap.empty<string, BufferView>()
        .add("Positions", BufferView.ofArray(new Float32Array(verts * 4), { elementType: ElementType.V4f }))
        .add("Normals", BufferView.ofArray(new Float32Array(verts * 3), { elementType: ElementType.V3f }))),
      indices: BufferView.ofArray(new Uint32Array(verts * 3), { elementType: ElementType.U32 }),
      drawCall: AVal.constant<DrawCall>({ kind: "indexed", indexCount: verts * 3, instanceCount: 1, firstIndex: 0, baseVertex: 0, firstInstance: 0 }),
      bounds: AVal.constant({ min: { x: -1, y: -1, z: -1 }, max: { x: 1, y: 1, z: 1 } }),
    };
  });
  const rows = new Float32Array(N * 20);
  for (let i = 0; i < N; i++) { rows[i * 20] = 1; rows[i * 20 + 5] = 1; rows[i * 20 + 10] = 1; rows[i * 20 + 15] = 1; }
  const table = uniformTable(FIELDS, cval(rows));
  const shared = HashMap.empty<string, aval<unknown>>()
    .add("ViewProjTrafo", cval(Trafo3d.identity) as aval<unknown>)
    .add("LightLocation", AVal.constant(new V3f(0, 0, 1)) as aval<unknown>);
  const object = (i: number): RenderObject => {
    const mesh = meshes[i % MESHES]!, row = table.row(i);
    return {
      effect, pipelineState: state,
      vertexAttributes: mesh.attributes,
      uniforms: asUniformProvider(shared.add("ModelTrafo", row.ModelTrafo as aval<unknown>).add("Color", row.Color as aval<unknown>)),
      bounds: mesh.bounds,
      textures: HashMap.empty(), samplers: HashMap.empty(),
      indices: mesh.indices, drawCall: mesh.drawCall,
    };
  };
  const set = cset<RenderTree>();
  const scene = compileHybridScene(gpu.device, sig, RenderTree.unorderedFromSet(set));
  scene.update(AdaptiveToken.top);
  return { set, scene, object };
}

/** WOMBAT_BENCH=prof: CPU-profile `f` and print where its time went, per object. */
async function profiled(label: string, n: number, f: () => void): Promise<void> {
  const { Session } = await import("node:inspector");
  const session = new Session();
  session.connect();
  const post = (m: string, p?: object): Promise<any> => new Promise((res, rej) => session.post(m as any, p as any, (e: unknown, r: unknown) => e ? rej(e) : res(r)));
  await post("Profiler.enable");
  await post("Profiler.setSamplingInterval", { interval: 50 });
  await post("Profiler.start");
  f();
  const { profile } = await post("Profiler.stop");
  session.disconnect();
  const byId = new Map<number, any>(profile.nodes.map((x: any) => [x.id, x]));
  const parent = new Map<number, number>();
  for (const x of profile.nodes) for (const c of x.children ?? []) parent.set(c, x.id);
  const key = (x: any): string => `${x.callFrame.functionName || "(anon)"} ${x.callFrame.url.split("/").slice(-1)[0]}:${x.callFrame.lineNumber + 1}`;
  const self = new Map<string, number>(), incl = new Map<string, number>();
  let total = 0;
  profile.samples.forEach((id: number, i: number) => {
    const d = (profile.timeDeltas[i] ?? 0) / 1000; total += d;
    self.set(key(byId.get(id)), (self.get(key(byId.get(id))) ?? 0) + d);
    const seen = new Set<string>();
    for (let c: number | undefined = id; c !== undefined; c = parent.get(c)) {
      const k = key(byId.get(c)); if (!seen.has(k)) { seen.add(k); incl.set(k, (incl.get(k) ?? 0) + d); }
    }
  });
  const lines = new Map<string, number>();
  const tick = total / profile.samples.length;
  for (const x of profile.nodes) for (const t of x.positionTicks ?? []) {
    const k = `${x.callFrame.url.split("/").slice(-1)[0]}:${t.line} ${x.callFrame.functionName}`; lines.set(k, (lines.get(k) ?? 0) + t.ticks * tick);
  }
  const top = (m: Map<string, number>, k: number): string => [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, k)
    .map(([name, v]) => `${((v / n) * 1000).toFixed(2).padStart(7)} us  ${name}`).join("\n");
  console.log(`\n=== ${label}: ${((total / n) * 1000).toFixed(1)} us per object\n--- self\n${top(self, 45)}\n--- inclusive\n${top(incl, 60)}\n--- lines\n${top(lines, 60)}\n`);
}

const ms = (f: () => void): number => { const t = performance.now(); f(); return performance.now() - t; };
const us = (total: number, n: number): string => ((total / n) * 1000).toFixed(1).padStart(6) + " us";

describe.skipIf(process.env.WOMBAT_BENCH === undefined)("bench: adding and removing render objects", () => {
  it("prints what an add and a remove cost", () => {
    const report: string[] = [];
    // warm the JIT on a throwaway world
    { const w = world(); const t = Array.from({ length: 3000 }, (_, i) => RenderTree.leaf(w.object(i))); transact(() => { for (const x of t) w.set.add(x); }); w.scene.update(AdaptiveToken.top); }

    const best = { make: 1e9, add: 1e9, stepped: 1e9, remove: 1e9 };
    for (let round = 0; round < 7; round++) {
      const w = world();
      let trees: RenderTree[] = [];
      const make = ms(() => { trees = Array.from({ length: N }, (_, i) => RenderTree.leaf(w.object(i))); });
      const insert = ms(() => { transact(() => { for (const t of trees) w.set.add(t); }); });
      const add = ms(() => w.scene.update(AdaptiveToken.top));
      const idle = ms(() => { for (let i = 0; i < 20; i++) w.scene.update(AdaptiveToken.top); }) / 20;
      const erase = ms(() => { transact(() => { for (const t of trees) w.set.remove(t); }); });
      const remove = ms(() => w.scene.update(AdaptiveToken.top));
      // in small steps, as objects come into view: 56 per update
      const trees2 = Array.from({ length: N }, (_, i) => RenderTree.leaf(w.object(i)));
      let stepped = 0;
      for (let i = 0; i < N; i += 56) {
        transact(() => { for (let k = i; k < Math.min(N, i + 56); k++) w.set.add(trees2[k]!); });
        stepped += ms(() => w.scene.update(AdaptiveToken.top));
      }
      best.make = Math.min(best.make, make); best.add = Math.min(best.add, add);
      best.stepped = Math.min(best.stepped, stepped); best.remove = Math.min(best.remove, remove);
      report.push(
        `round ${round}: make ${us(make, N)} | set.add ${us(insert, N)} | scene add ${us(add, N)} | ` +
        `stepped add ${us(stepped, N)} | set.remove ${us(erase, N)} | scene remove ${us(remove, N)} | idle update ${idle.toFixed(3)} ms`);
    }
    console.log("\n" + report.join("\n") +
      `\nBEST: make ${us(best.make, N)} | scene add ${us(best.add, N)} | stepped add ${us(best.stepped, N)} | scene remove ${us(best.remove, N)}\n`);
    expect(report.length).toBe(7);
  }, 120000);

  it.skipIf(process.env.WOMBAT_BENCH !== "prof")("profiles an add and a remove", async () => {
    const w = world();
    const trees = Array.from({ length: N }, (_, i) => RenderTree.leaf(w.object(i)));
    transact(() => { for (const t of trees) w.set.add(t); });
    await profiled("scene add", N, () => w.scene.update(AdaptiveToken.top));
    transact(() => { for (const t of trees) w.set.remove(t); });
    await profiled("scene remove", N, () => w.scene.update(AdaptiveToken.top));
  }, 120000);
});
