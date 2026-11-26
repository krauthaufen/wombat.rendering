// Draw plans (HeapDrawSpec.shape / DrawShape): objects made of the same
// objects apart from their uniforms are added by a plan made for the
// first of them. A plan is a shortcut, so the scene it leaves behind must
// be the one the full way leaves: the same scene is driven twice, with
// shapes on and off, and after every step the two heaps must agree on
// every header word, every draw record and every pool use count.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap, cset, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V3f, V4f } from "@aardworx/wombat.base";
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

const N = 240;
const top = AdaptiveToken.top;

/** One scene and everything a step may change; `shapes` is the only difference between two worlds. */
function world(shapes: boolean) {
  const gpu = new MockGPU();
  const effect = makeHeapTestEffect();
  const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
  const states = [
    PipelineState.constant({ rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" }, depth: { write: true, compare: "less" } }),
    PipelineState.constant({ rasterizer: { topology: "triangle-list", cullMode: "back", frontFace: "ccw" }, depth: { write: true, compare: "less" } }),
  ];
  const meshes = [0, 1, 2].map((m) => {
    const verts = 30 + m * 9;
    return {
      attributes: asAttributeProvider(HashMap.empty<string, BufferView>()
        .add("Positions", BufferView.ofArray(new Float32Array(verts * 4).fill(m + 1), { elementType: ElementType.V4f }))
        .add("Normals", BufferView.ofArray(new Float32Array(verts * 3).fill(0.5), { elementType: ElementType.V3f }))),
      indices: BufferView.ofArray(Uint32Array.from({ length: verts * 3 }, (_, i) => i % verts), { elementType: ElementType.U32 }),
      drawCall: AVal.constant<DrawCall>({ kind: "indexed", indexCount: verts * 3, instanceCount: 1, firstIndex: 0, baseVertex: 0, firstInstance: 0 }),
      bounds: AVal.constant({ min: { x: -1, y: -1, z: -1 }, max: { x: 1, y: 1, z: m + 1 } }),
    };
  });
  const rows = cval(new Float32Array(N * 16));
  const table = uniformTable({ ModelTrafo: "mat4x4<f32>" } as const, rows);
  const viewProj = cval(Trafo3d.identity);
  const light = cval(new V3f(0, 0, 1));
  const sharedColor = cval(new V4f(1, 0, 0, 1));
  const ownColors = Array.from({ length: N }, (_, i) => cval(new V4f(i / N, 0, 0, 1)));
  const shared = HashMap.empty<string, aval<unknown>>()
    .add("ViewProjTrafo", viewProj as aval<unknown>).add("LightLocation", light as aval<unknown>);
  const object = (i: number): RenderObject => {
    const mesh = meshes[i % 3]!;
    return {
      effect, pipelineState: states[i % 2]!,
      vertexAttributes: mesh.attributes,
      uniforms: asUniformProvider(shared
        .add("ModelTrafo", table.row(i).ModelTrafo as aval<unknown>)
        // every fourth has a colour of its own; the others share one
        .add("Color", (i % 4 === 0 ? ownColors[i]! : sharedColor) as aval<unknown>)),
      // (some meshes without bounds: not culled)
      ...(i % 3 !== 2 ? { bounds: mesh.bounds } : {}),
      textures: HashMap.empty(), samplers: HashMap.empty(),
      indices: mesh.indices, drawCall: mesh.drawCall,
    };
  };
  const trees = Array.from({ length: N }, (_, i) => RenderTree.leaf(object(i)));
  const set = cset<RenderTree>();
  const scene = compileHybridScene(gpu.device, sig, RenderTree.unorderedFromSet(set), { enableDrawShapes: shapes });
  const heap = scene.heapSceneForTest as unknown as {
    _debug: { stateForTest(): unknown; plannedAdds(): number; forceCompact(): number };
  };
  return { set, scene, trees, heap, rows, viewProj, light, sharedColor, ownColors };
}
type World = ReturnType<typeof world>;

describe("heap: draw plans", () => {
  it("leave the scene exactly as adding every draw in full does", () => {
    const a = world(true), b = world(false);
    const both = (f: (w: World) => void): void => {
      for (const w of [a, b]) { transact(() => f(w)); w.scene.update(top); }
      expect(a.heap._debug.stateForTest()).toEqual(b.heap._debug.stateForTest());
    };
    const range = (from: number, to: number, step = 1): number[] => {
      const out: number[] = []; for (let i = from; i < to; i += step) out.push(i); return out;
    };

    both((w) => { for (const i of range(0, 120)) w.set.add(w.trees[i]!); });
    // all but the first of each shape went by its plan (3 meshes x 2 states, with and without bounds)
    expect(a.heap._debug.plannedAdds()).toBeGreaterThan(100);
    expect(b.heap._debug.plannedAdds()).toBe(0);

    both((w) => { for (const i of range(120, N)) w.set.add(w.trees[i]!); });
    both((w) => { for (const i of range(0, N, 3)) w.set.remove(w.trees[i]!); });
    both((w) => { for (const i of range(0, N, 3)) w.set.add(w.trees[i]!); });
    // values change: shared ones, one object's own, the table
    both((w) => {
      w.viewProj.value = Trafo3d.translation(new V3d(1, 2, 3));
      w.sharedColor.value = new V4f(0, 1, 0, 1);
      w.ownColors[8]!.value = new V4f(0, 0, 1, 1);
      w.rows.value = new Float32Array(N * 16).fill(2);
    });
    // a shape's last draw goes and comes back: its plan's entries were freed in between
    both((w) => { for (const i of range(0, N)) if (i % 6 === 1) w.set.remove(w.trees[i]!); });
    both((w) => { for (const i of range(0, N)) if (i % 6 === 1) w.set.add(w.trees[i]!); });
    both((w) => { w.set.clear(); });
    both((w) => { for (const i of range(0, N)) w.set.add(w.trees[i]!); });
    expect(a.heap._debug.plannedAdds()).toBeGreaterThan(400);
  });

  it("survive a compaction: refs that moved are read from the entries", () => {
    const a = world(true), b = world(false);
    const both = (f: (w: World) => void): void => {
      for (const w of [a, b]) { transact(() => f(w)); w.scene.update(top); }
      expect(a.heap._debug.stateForTest()).toEqual(b.heap._debug.stateForTest());
    };
    both((w) => { for (let i = 0; i < N; i++) w.set.add(w.trees[i]!); });
    // free the objects with a colour of their own, leaving holes, and pack the arena
    both((w) => { for (let i = 0; i < N; i += 4) w.set.remove(w.trees[i]!); });
    for (const w of [a, b]) { w.heap._debug.forceCompact(); w.scene.update(top); }
    expect(a.heap._debug.stateForTest()).toEqual(b.heap._debug.stateForTest());
    const before = a.heap._debug.plannedAdds();
    both((w) => { for (let i = 0; i < N; i += 4) w.set.add(w.trees[i]!); });
    expect(a.heap._debug.plannedAdds()).toBeGreaterThan(before);
  });
});
