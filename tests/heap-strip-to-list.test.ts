// Strips on the heap path are rewritten into the list of their
// primitives (runtime/heapScene/stripToList.ts). Pixel correctness is
// in tests-browser/heap-strip-real.test.ts; here: the index data, the
// sharing it must keep, and the adapter's primitive-restart handling.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap, cval } from "@aardworx/wombat.adaptive";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import {
  IBuffer, ElementType, PipelineState,
  type BufferView, type DrawCall, type RenderObject,
} from "@aardworx/wombat.rendering/core";
import { AttributeProvider, UniformProvider } from "../packages/rendering/src/core/provider.js";
import type { Topology } from "../packages/rendering/src/core/pipelineState.js";
import {
  expandStrip, stripKindOfSpec, stripSpecToList, STRIP_RESTART,
} from "../packages/rendering/src/runtime/heapScene/stripToList.js";
import { renderObjectToHeapSpec } from "../packages/rendering/src/runtime/heapAdapter.js";
import { makeEffect } from "./_makeEffect.js";

const u32 = (...v: number[]): Uint32Array => new Uint32Array(v);
const R = STRIP_RESTART;

const strip = (topology: Topology) => PipelineState.constant({
  rasterizer: { topology, cullMode: "none", frontFace: "ccw" },
});

describe("expandStrip", () => {
  it("a triangle strip becomes its triangles, odd ones as (i, i+2, i+1)", () => {
    expect([...expandStrip("triangle-strip", undefined, 0, 5)])
      .toEqual([0, 1, 2,  1, 3, 2,  2, 3, 4]);
    expect([...expandStrip("triangle-strip", u32(7, 8, 9, 5), 0, 4)])
      .toEqual([7, 8, 9,  8, 5, 9]);
  });

  it("a line strip becomes its segments", () => {
    expect([...expandStrip("line-strip", undefined, 0, 4)]).toEqual([0, 1,  1, 2,  2, 3]);
  });

  it("fewer vertices than one primitive give nothing", () => {
    expect(expandStrip("triangle-strip", undefined, 0, 2).length).toBe(0);
    expect(expandStrip("line-strip", undefined, 0, 1).length).toBe(0);
    expect(expandStrip("triangle-strip", undefined, 0, 0).length).toBe(0);
  });

  it("a restart ends the strip; the next one starts with even parity", () => {
    expect([...expandStrip("triangle-strip", u32(0, 1, 2, 3, R, 4, 5, 6, 7), 0, 9)])
      .toEqual([0, 1, 2,  1, 3, 2,  4, 5, 6,  5, 7, 6]);
    // a restart after an ODD number of triangles must not flip what follows
    expect([...expandStrip("triangle-strip", u32(0, 1, 2, R, 4, 5, 6), 0, 7)])
      .toEqual([0, 1, 2,  4, 5, 6]);
    expect([...expandStrip("line-strip", u32(0, 1, R, 2, 3), 0, 5)]).toEqual([0, 1,  2, 3]);
    // restarts in a row, and a strip too short to draw
    expect([...expandStrip("triangle-strip", u32(0, 1, R, R, 2, 3, 4), 0, 7)]).toEqual([2, 3, 4]);
  });

  it("a slice is a strip of its own: parity counts from the slice start", () => {
    expect([...expandStrip("triangle-strip", u32(9, 9, 9, 0, 1, 2, 3), 3, 4)])
      .toEqual([0, 1, 2,  1, 3, 2]);
  });
});

describe("stripSpecToList", () => {
  it("leaves list and point draws alone, and draws whose topology a rule decides", () => {
    expect(stripKindOfSpec({})).toBeUndefined();
    expect(stripKindOfSpec({ pipelineState: strip("triangle-list") })).toBeUndefined();
    expect(stripKindOfSpec({ pipelineState: strip("point-list") })).toBeUndefined();
    expect(stripKindOfSpec({ pipelineState: strip("line-strip") })).toBe("line-strip");
    expect(stripKindOfSpec({ pipelineState: strip("triangle-strip"), modeRules: { topology: {} } }))
      .toBeUndefined();
  });

  it("gives a list topology and list indices, and keeps the spec's other fields", () => {
    const ps = strip("triangle-strip");
    const spec = { pipelineState: ps, vertexCount: 4, inputs: { A: 1 } };
    const l = stripSpecToList(spec, "triangle-strip");
    expect(l.pipelineState!.rasterizer.topology.force()).toBe("triangle-list");
    expect(l.pipelineState!.rasterizer.cullMode).toBe(ps.rasterizer.cullMode);
    expect([...(l.indices as Uint32Array)]).toEqual([0, 1, 2,  1, 3, 2]);
    expect(l.vertexCount).toBeUndefined();
    expect(l.inputs).toBe(spec.inputs);
  });

  it("draws of one strip mesh share ONE list and ONE pipeline state", () => {
    const ps = strip("triangle-strip");
    // non-indexed: by vertex count
    const a = stripSpecToList({ pipelineState: ps, vertexCount: 4 }, "triangle-strip");
    const b = stripSpecToList({ pipelineState: ps, vertexCount: 4 }, "triangle-strip");
    expect(b.indices).toBe(a.indices);
    expect(b.pipelineState).toBe(a.pipelineState);
    // two states with constant strip topology share the topology aval
    // (bucket keys hash the inner avals by identity)
    const c = stripSpecToList({ pipelineState: strip("triangle-strip"), vertexCount: 4 }, "triangle-strip");
    expect(c.pipelineState!.rasterizer.topology).toBe(a.pipelineState!.rasterizer.topology);
    // indexed: by the index source, array or aval
    const arr = u32(0, 1, 2, 3);
    const d = stripSpecToList({ pipelineState: ps, indices: arr }, "triangle-strip");
    const e = stripSpecToList({ pipelineState: ps, indices: arr }, "triangle-strip");
    expect(e.indices).toBe(d.indices);
    const av = cval(arr);
    const f = stripSpecToList({ pipelineState: ps, indices: av }, "triangle-strip");
    const g = stripSpecToList({ pipelineState: ps, indices: av }, "triangle-strip");
    expect(g.indices).toBe(f.indices);
    expect([...(f.indices as { force(): Uint32Array }).force()]).toEqual([0, 1, 2,  1, 3, 2]);
    // a slice gets its own list
    const h = stripSpecToList({ pipelineState: ps, indices: arr, firstIndex: 1, indexCount: 3 }, "triangle-strip");
    expect(h.indices).not.toBe(d.indices);
    expect([...(h.indices as Uint32Array)]).toEqual([1, 2, 3]);
    expect(h.firstIndex).toBeUndefined();
    expect(h.indexCount).toBeUndefined();
  });
});

describe("heap adapter — strip index buffers", () => {
  const Tvec3f: Type = Vec(Tf32, 3);
  const Tvec4f: Type = Vec(Tf32, 4);
  const effect = makeEffect(
    `
      function vsMain(input: { position: V3f }): { gl_Position: V4f } {
        return { gl_Position: new V4f(input.position.x, input.position.y, input.position.z, 1.0) };
      }
      function fsMain(_input: {}): { outColor: V4f } {
        return { outColor: new V4f(1.0, 0.0, 0.0, 1.0) };
      }
    `,
    [
      { name: "vsMain", stage: "vertex",
        inputs: [{ name: "position", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] }],
      },
      { name: "fsMain", stage: "fragment",
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ],
  );
  const ro = (topology: Topology, baseVertex: number): RenderObject => {
    const view: BufferView = {
      buffer: AVal.constant(IBuffer.fromHost(new Float32Array(30).buffer)),
      offset: 0, stride: 12, elementType: ElementType.V3f,
    };
    const indices: BufferView = {
      buffer: AVal.constant(IBuffer.fromHost(new Uint16Array([0, 1, 2, 0xFFFF, 3, 4, 5]).buffer)),
      offset: 0, stride: 2, elementType: ElementType.U16,
    };
    return {
      effect,
      pipelineState: strip(topology),
      vertexAttributes: AttributeProvider.ofObject({ position: view }),
      uniforms: UniformProvider.empty,
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      indices,
      drawCall: AVal.constant<DrawCall>({
        kind: "indexed", indexCount: 7, instanceCount: 1,
        firstIndex: 0, baseVertex, firstInstance: 0,
      }),
    };
  };
  const indicesOf = (r: RenderObject): number[] => {
    const spec = renderObjectToHeapSpec(r, AdaptiveToken.top)!;
    return [...(spec.indices as { force(): Uint32Array }).force()];
  };

  it("a 16-bit restart becomes the heap's restart and takes no baseVertex", () => {
    expect(indicesOf(ro("triangle-strip", 0))).toEqual([0, 1, 2, R, 3, 4, 5]);
    expect(indicesOf(ro("triangle-strip", 4))).toEqual([4, 5, 6, R, 7, 8, 9]);
  });

  it("in a LIST the same value is an ordinary index", () => {
    expect(indicesOf(ro("triangle-list", 0))).toEqual([0, 1, 2, 0xFFFF, 3, 4, 5]);
    expect(indicesOf(ro("triangle-list", 4))).toEqual([4, 5, 6, 0xFFFF + 4, 7, 8, 9]);
  });
});
