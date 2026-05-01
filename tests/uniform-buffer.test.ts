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
