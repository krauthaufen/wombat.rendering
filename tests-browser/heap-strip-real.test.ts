// Real-GPU regression for STRIP topologies on the heap path.
//
// Found via PoEtry's passive tree: a unit quad as a triangle strip
// (4 vertices), drawn a few hundred times, came out with triangles
// BETWEEN the quads. A heap bucket is one draw over all its records,
// so the GPU assembled one strip across them. Strips are now rewritten
// into the list of the same primitives when a draw is added
// (runtime/heapScene/stripToList.ts).
//
// Each case draws two separate shapes into one target and asserts
// that both are there AND that the gap between them is untouched.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, cset, cval, transact, type aset } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import {
  buildHeapScene, type HeapDrawSpec,
} from "../packages/rendering/src/runtime/heapScene.js";
import { PipelineState, type Topology } from "../packages/rendering/src/core/pipelineState.js";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";
import { makeHeapTestEffect } from "../tests/_heapTestEffect.js";

const trafoIdentity = AVal.constant(Trafo3d.identity) as unknown;
const blueInputs = {
  ModelTrafo: trafoIdentity, Color: new V4f(0, 0, 1, 1) as unknown,
  ViewProjTrafo: trafoIdentity, LightLocation: new V3d(0, 0, 1) as unknown,
};

const W = 32, H = 32;

/** A quad spanning x0..x1 (clip space), full height, in strip order. */
const stripQuad = (x0: number, x1: number): Float32Array => new Float32Array([
  x0, -0.9, 0,   x1, -0.9, 0,   x0, 0.9, 0,   x1, 0.9, 0,
]);
const normals = (n: number): Float32Array =>
  new Float32Array(n * 3).map((_, i) => (i % 3 === 2 ? 1 : 0));

const state = (topology: Topology | ReturnType<typeof cval<Topology>>) =>
  typeof topology === "string"
    ? PipelineState.constant({
        rasterizer: { topology, cullMode: "none", frontFace: "ccw" },
        depth: { write: false, compare: "always" },
      })
    : (() => {
        const c = PipelineState.constant({
          rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" },
          depth: { write: false, compare: "always" },
        });
        return { ...c, rasterizer: { ...c.rasterizer, topology } } as PipelineState;
      })();

interface Shot {
  /** Is pixel (x, y) the draw colour (blue)? y counts from the top. */
  blue(x: number, y: number): boolean;
  count(x0: number, x1: number): number;
  errors: string[];
}

async function withScene<T>(
  specs: ReadonlyArray<Omit<HeapDrawSpec, "effect">> | ((effect: HeapDrawSpec["effect"]) => aset<HeapDrawSpec>),
  body: (render: () => Promise<Shot>) => Promise<T>,
): Promise<T> {
  const device = await requestRealDevice();
  const errors: string[] = [];
  device.onuncapturederror = (e) =>
    errors.push((e.error as { message?: string }).message ?? String(e.error));
  try {
    const sig = createFramebufferSignature({
      colors: { outColor: "rgba8unorm" },
      depthStencil: { format: "depth24plus" },
    });
    const effect = makeHeapTestEffect();
    const draws = typeof specs === "function"
      ? specs(effect)
      : specs.map((s) => ({ effect, ...s }) as HeapDrawSpec);
    const scene = buildHeapScene(device, sig, draws, {
      fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
    });
    const colorTex = device.createTexture({
      size: { width: W, height: H, depthOrArrayLayers: 1 },
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const depthTex = device.createTexture({
      size: { width: W, height: H, depthOrArrayLayers: 1 },
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const render = async (): Promise<Shot> => {
      scene.update(AdaptiveToken.top);
      const enc = device.createCommandEncoder();
      scene.encodeComputePrep(enc, AdaptiveToken.top);
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: colorTex.createView(),
          clearValue: { r: 0, g: 1, b: 0, a: 1 },
          loadOp: "clear", storeOp: "store",
        }],
        depthStencilAttachment: {
          view: depthTex.createView(),
          depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
        },
      });
      scene.encodeIntoPass(pass);
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      const px = await readTexturePixels(device, colorTex);
      const blue = (x: number, y: number): boolean => {
        const i = (y * W + x) * 4;
        return px[i + 2]! > 200 && px[i + 1]! < 60 && px[i]! < 60;
      };
      const count = (x0: number, x1: number): number => {
        let n = 0;
        for (let y = 0; y < H; y++) for (let x = x0; x < x1; x++) if (blue(x, y)) n++;
        return n;
      };
      return { blue, count, errors };
    };
    const r = await body(render);
    scene.dispose();
    return r;
  } finally {
    device.destroy();
  }
}

// Left quad covers columns 2..9, right quad 22..29 (of 32); rows 2..29.
const LEFT = stripQuad(-0.875, -0.375), RIGHT = stripQuad(0.375, 0.875);
const QUAD_PIXELS = 8 * 28;

function expectTwoQuads(s: Shot): void {
  expect(s.errors, "gpu errors").toEqual([]);
  expect(s.count(0, 12), "left quad").toBe(QUAD_PIXELS);
  expect(s.count(20, 32), "right quad").toBe(QUAD_PIXELS);
  expect(s.count(12, 20), "between the quads").toBe(0);
}

describe("heap strip topologies real-GPU", () => {
  it("two NON-indexed triangle strips stay two quads", async () => {
    const ps = state("triangle-strip");
    await withScene([
      { inputs: { Positions: LEFT,  Normals: normals(4), ...blueInputs }, vertexCount: 4, pipelineState: ps },
      { inputs: { Positions: RIGHT, Normals: normals(4), ...blueInputs }, vertexCount: 4, pipelineState: ps },
    ], async (render) => expectTwoQuads(await render()));
  });

  it("two INDEXED triangle strips stay two quads (one shared index array)", async () => {
    const ps = state("triangle-strip");
    const idx = new Uint32Array([0, 1, 2, 3]);
    await withScene([
      { inputs: { Positions: LEFT,  Normals: normals(4), ...blueInputs }, indices: idx, pipelineState: ps },
      { inputs: { Positions: RIGHT, Normals: normals(4), ...blueInputs }, indices: idx, pipelineState: ps },
    ], async (render) => expectTwoQuads(await render()));
  });

  it("a primitive restart inside one indexed strip separates its parts", async () => {
    const pos = new Float32Array([...LEFT, ...RIGHT]);
    await withScene([{
      inputs: { Positions: pos, Normals: normals(8), ...blueInputs },
      indices: new Uint32Array([0, 1, 2, 3, 0xFFFFFFFF, 4, 5, 6, 7]),
      pipelineState: state("triangle-strip"),
    }], async (render) => expectTwoQuads(await render()));
  });

  it("an index slice of a strip draws only that part", async () => {
    const pos = new Float32Array([...LEFT, ...RIGHT]);
    await withScene([{
      inputs: { Positions: pos, Normals: normals(8), ...blueInputs },
      indices: new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7]),
      firstIndex: 4, indexCount: 4,
      pipelineState: state("triangle-strip"),
    }], async (render) => {
      const s = await render();
      expect(s.errors, "gpu errors").toEqual([]);
      expect(s.count(0, 20), "left of the slice").toBe(0);
      expect(s.count(20, 32), "right quad").toBe(QUAD_PIXELS);
    });
  });

  it("a strip keeps its winding: back faces of odd triangles are culled alike", async () => {
    // Front-face culling of a ccw strip quad must remove BOTH triangles
    // (a list that flipped the odd triangle would leave half the quad).
    const front = PipelineState.constant({
      rasterizer: { topology: "triangle-strip", cullMode: "front", frontFace: "ccw" },
      depth: { write: false, compare: "always" },
    });
    const back = PipelineState.constant({
      rasterizer: { topology: "triangle-strip", cullMode: "back", frontFace: "ccw" },
      depth: { write: false, compare: "always" },
    });
    await withScene([
      { inputs: { Positions: LEFT,  Normals: normals(4), ...blueInputs }, vertexCount: 4, pipelineState: front },
      { inputs: { Positions: RIGHT, Normals: normals(4), ...blueInputs }, vertexCount: 4, pipelineState: back },
    ], async (render) => {
      const s = await render();
      expect(s.errors, "gpu errors").toEqual([]);
      expect(s.count(0, 12), "left quad, front faces culled").toBe(0);
      expect(s.count(20, 32), "right quad, back faces culled").toBe(QUAD_PIXELS);
    });
  });

  it("two line strips are not joined by a line", async () => {
    // Two vertical lines; a joined strip would add a diagonal across.
    const ps = state("line-strip");
    const line = (x: number): Float32Array => new Float32Array([x, -0.9, 0,  x, 0, 0,  x, 0.9, 0]);
    await withScene([
      { inputs: { Positions: line(-0.72), Normals: normals(3), ...blueInputs }, vertexCount: 3, pipelineState: ps },
      { inputs: { Positions: line(0.72),  Normals: normals(3), ...blueInputs }, vertexCount: 3, pipelineState: ps },
    ], async (render) => {
      const s = await render();
      expect(s.errors, "gpu errors").toEqual([]);
      expect(s.count(0, 12), "left line").toBeGreaterThanOrEqual(26);
      expect(s.count(20, 32), "right line").toBeGreaterThanOrEqual(26);
      expect(s.count(12, 20), "between the lines").toBe(0);
    });
  });

  it("a topology that changes between list and strip is redrawn with the new one", async () => {
    // The same 6 vertices (a zigzag over a rectangle): as a triangle
    // LIST two of its four triangles are drawn, as a STRIP all of it.
    const zigzag = (x0: number, x1: number): Float32Array => {
      const xm = (x0 + x1) / 2;
      return new Float32Array([
        x0, -0.9, 0,  x0, 0.9, 0,  xm, -0.9, 0,  xm, 0.9, 0,  x1, -0.9, 0,  x1, 0.9, 0,
      ]);
    };
    const topo = cval<Topology>("triangle-list");
    const ps = state(topo);
    await withScene([
      { inputs: { Positions: zigzag(-0.875, -0.375), Normals: normals(6), ...blueInputs }, vertexCount: 6, pipelineState: ps },
      { inputs: { Positions: zigzag(0.375, 0.875),   Normals: normals(6), ...blueInputs }, vertexCount: 6, pipelineState: ps },
    ], async (render) => {
      const a = await render();
      expect(a.errors, "gpu errors").toEqual([]);
      expect(a.count(0, 12), "left, list").toBeLessThan(QUAD_PIXELS * 0.7);
      expect(a.count(12, 20), "between, list").toBe(0);
      transact(() => { topo.value = "triangle-strip"; });
      expectTwoQuads(await render());
      transact(() => { topo.value = "triangle-list"; });
      const c = await render();
      expect(c.count(0, 12), "left, list again").toBe(a.count(0, 12));
      expect(c.count(12, 20), "between, list again").toBe(0);
    });
  });

  it("a draw added later with trafos that are already in use is drawn at once", async () => {
    // Its derived uniforms (ModelViewProj …) come from constituents that
    // are live and unchanged: they must still be computed for the new draw.
    const ps = state("triangle-list");
    const list = (q: Float32Array): Float32Array => new Float32Array([
      ...q.subarray(0, 9), ...q.subarray(6, 9), ...q.subarray(3, 6), ...q.subarray(9, 12),
    ]);
    const set = cset<HeapDrawSpec>();
    let effect: HeapDrawSpec["effect"];
    const spec = (q: Float32Array): HeapDrawSpec => ({
      effect, inputs: { Positions: list(q), Normals: normals(6), ...blueInputs }, vertexCount: 6, pipelineState: ps,
    }) as HeapDrawSpec;
    await withScene((e) => { effect = e; transact(() => set.add(spec(LEFT))); return set; }, async (render) => {
      const a = await render();
      expect(a.errors, "gpu errors").toEqual([]);
      expect(a.count(0, 12), "left quad").toBe(QUAD_PIXELS);
      expect(a.count(20, 32), "right quad, not added yet").toBe(0);
      transact(() => set.add(spec(RIGHT)));
      expectTwoQuads(await render());
    });
  });
});
