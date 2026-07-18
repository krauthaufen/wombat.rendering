// prepareUniformBuffer — pack named uniforms into a UBO per layout.

import { describe, expect, it } from "vitest";
import { AdaptiveToken, HashMap, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { V4f, M44f } from "@aardworx/wombat.base";
import { prepareUniformBuffer } from "@aardworx/wombat.rendering/resources";
import type { UniformBlockInfo } from "@aardworx/wombat.rendering/core";
import type { Type } from "@aardworx/wombat.shader/ir";
import { MockGPU } from "./_mockGpu.js";

const f32:   Type = { kind: "Float", width: 32 };
const vec4f: Type = { kind: "Vector", element: f32, dim: 4 };
const mat4f: Type = { kind: "Matrix", element: f32, rows: 4, cols: 4 };

const layout: UniformBlockInfo = {
  name: "Globals",
  group: 0,
  slot: 0,
  size: 16 + 16 + 64,
  fields: [
    { name: "tint",     type: vec4f, offset: 0,  size: 16, align: 16 },
    { name: "scale",    type: f32,   offset: 16, size: 4,  align: 4 },
    { name: "viewProj", type: mat4f, offset: 32, size: 64, align: 16 },
  ],
};

describe("prepareUniformBuffer", () => {
  it("packs number / V4f / M44f at the right offsets", () => {
    const gpu = new MockGPU();
    const inputs = HashMap.empty<string, aval<unknown>>()
      .add("tint", cval(new V4f(1, 0, 0, 1)))
      .add("scale", cval(2.5))
      .add("viewProj", cval(M44f.identity));
    const r = prepareUniformBuffer(gpu.device, layout, inputs, { label: "globals" });
    r.acquire();
    r.getValue(AdaptiveToken.top);
    expect(gpu.buffers).toHaveLength(1);
    expect(gpu.buffers[0]!.label).toBe("globals");
    expect(gpu.writeBufferCalls).toHaveLength(1);
    const data = gpu.writeBufferCalls[0]!.data as ArrayBuffer;
    const view = new Float32Array(data);
    // tint at f32[0..4]
    expect([view[0], view[1], view[2], view[3]]).toEqual([1, 0, 0, 1]);
    // scale at f32[4]
    expect(view[4]).toBe(2.5);
    // viewProj — diagonal identity at offsets 8/13/18/23
    expect(view[8]).toBe(1);
    expect(view[13]).toBe(1);
    expect(view[18]).toBe(1);
    expect(view[23]).toBe(1);
    r.release();
  });

  it("re-uploads on uniform change, reuses GPU buffer", () => {
    const gpu = new MockGPU();
    const tint = cval(new V4f(1, 1, 1, 1));
    const inputs = HashMap.empty<string, aval<unknown>>()
      .add("tint", tint)
      .add("scale", cval(1))
      .add("viewProj", cval(M44f.identity));
    const r = prepareUniformBuffer(gpu.device, layout, inputs);
    r.acquire();
    r.getValue(AdaptiveToken.top);
    transact(() => { tint.value = new V4f(0.5, 0.25, 0, 1); });
    r.getValue(AdaptiveToken.top);
    expect(gpu.buffers).toHaveLength(1);
    expect(gpu.writeBufferCalls).toHaveLength(2);
    const data = gpu.writeBufferCalls[1]!.data as ArrayBuffer;
    const view = new Float32Array(data);
    expect([view[0], view[1], view[2], view[3]]).toEqual([0.5, 0.25, 0, 1]);
    r.release();
  });

  it("missing inputs are zero-padded; extras are ignored", () => {
    const gpu = new MockGPU();
    const inputs = HashMap.empty<string, aval<unknown>>()
      .add("scale", cval(7))
      .add("ignored", cval(new V4f(9, 9, 9, 9))); // not in layout
    const r = prepareUniformBuffer(gpu.device, layout, inputs);
    r.acquire();
    r.getValue(AdaptiveToken.top);
    const data = gpu.writeBufferCalls[0]!.data as ArrayBuffer;
    const view = new Float32Array(data);
    // tint missing → zeros
    expect([view[0], view[1], view[2], view[3]]).toEqual([0, 0, 0, 0]);
    // scale at f32[4]
    expect(view[4]).toBe(7);
    // viewProj missing → zeros
    expect(view[8]).toBe(0);
    r.release();
  });
});

describe("prepareUniformBuffer — plain number[] backed d-types", () => {
  // wombat.base's double types (V*d / M*d / Trafo3d) back `_data`
  // with plain JS arrays (packed doubles), NOT typed arrays.
  // Regression: the packer's duck-typed branches must accept them —
  // a raw Trafo3d / M44d / V4d uniform on the legacy path previously
  // threw "unsupported uniform value" every frame (the frame loop
  // then gives up → frozen picture after placing a measurement).
  it("packs Trafo3d-shaped, M44d-shaped and raw number[] values", () => {
    const gpu = new MockGPU();
    const m = [
      2, 0, 0, 0,
      0, 3, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ];
    const trafoLike = { forward: { _data: m }, backward: { _data: m } };
    const v4dLike = { _data: [1, 2, 3, 4] };
    const inputs = HashMap.empty<string, aval<unknown>>()
      .add("tint", cval<unknown>(v4dLike))
      .add("scale", cval<unknown>([9]))
      .add("viewProj", cval<unknown>(trafoLike));
    const r = prepareUniformBuffer(gpu.device, layout, inputs, { label: "d-types" });
    r.acquire();
    r.getValue(AdaptiveToken.top);
    const data = gpu.writeBufferCalls[0]!.data as ArrayBuffer;
    const f = new Float32Array(data.slice(0));
    expect([f[0], f[1], f[2], f[3]]).toEqual([1, 2, 3, 4]);   // v4d-like
    expect(f[4]).toBe(9);                                     // number[1]
    expect(f[8]).toBe(2);                                     // trafo forward[0]
    expect(f[13]).toBe(3);                                    // forward[5]
    r.release();
  });
});
