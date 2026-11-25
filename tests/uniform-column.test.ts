// Uniform columns / tables (core/uniformColumn.ts): layout, and reading a
// cell as the ordinary aval it is (what the classic path and user code do).

import { describe, expect, it } from "vitest";
import { AdaptiveToken, cval, transact } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d } from "@aardworx/wombat.base";
import {
  isUniformColumnCell, tableLayout, uniformColumn, uniformTable,
} from "../packages/rendering/src/core/uniformColumn.js";

const top = AdaptiveToken.top;

describe("uniform columns", () => {
  it("a table lays its fields out for the heap: vec4/mat4 on 16 bytes, the rest on 4", () => {
    expect(tableLayout({ Model: "mat4x4<f32>", Tint: "vec4<f32>" })).toEqual({ stride: 80, offsets: { Model: 0, Tint: 64 } });
    expect(tableLayout({ Speed: "f32", Dir: "vec3<f32>", Tint: "vec4<f32>", Id: "u32" }))
      .toEqual({ stride: 48, offsets: { Speed: 0, Dir: 4, Tint: 16, Id: 32 } });
    expect(tableLayout({ A: "f32", B: "vec2<f32>" })).toEqual({ stride: 12, offsets: { A: 0, B: 4 } });
  });

  it("a cell is an aval of its value and follows the array and its row", () => {
    const data = cval(new Float32Array([1, 2, 3, 4,  5, 6, 7, 8]));
    const col = uniformColumn("vec4<f32>", data);
    const row = cval(0);
    const fixed = col.at(1), moving = col.at(row);
    expect(isUniformColumnCell(fixed)).toBe(true);
    expect(isUniformColumnCell(data)).toBe(false);
    expect(fixed.getValue(top).toArray()).toEqual([5, 6, 7, 8]);
    expect(moving.getValue(top).toArray()).toEqual([1, 2, 3, 4]);
    transact(() => { row.value = 1; });
    expect(moving.getValue(top).toArray()).toEqual([5, 6, 7, 8]);
    transact(() => { data.value = new Float32Array([1, 2, 3, 4,  9, 9, 9, 9]); });
    expect(fixed.getValue(top).toArray()).toEqual([9, 9, 9, 9]);
  });

  it("a table row gives every field; integers keep their bits; a matrix comes back as a trafo", () => {
    const fields = { Model: "mat4x4<f32>", Id: "u32", Speed: "f32" } as const;
    const { stride, offsets } = tableLayout(fields);
    const data = new Float32Array((stride / 4) * 2);
    const u32 = new Uint32Array(data.buffer);
    const t = Trafo3d.translation(new V3d(3, 4, 5));
    t.forward.copyTo(data, stride / 4 + offsets.Model / 4);        // row 1
    u32[stride / 4 + offsets.Id / 4] = 0xdeadbeef;
    data[stride / 4 + offsets.Speed / 4] = 2.5;
    const row = uniformTable(fields, data).row(1);
    expect(row.Id.getValue(top)).toBe(0xdeadbeef);
    expect(row.Speed.getValue(top)).toBe(2.5);
    const m = row.Model.getValue(top);
    expect(m.forward.toArray()).toEqual(t.forward.toArray());
    expect(m.backward.toArray()).toEqual(t.backward.toArray());
  });

  it("rejects a vec4 that would not sit on a 16-byte boundary", () => {
    // (a table never produces this; the check guards hand-made layouts)
    expect(() => tableLayout({ A: "f32", Tint: "vec4<f32>" })).not.toThrow();
    expect(tableLayout({ A: "f32", Tint: "vec4<f32>" }).offsets.Tint).toBe(16);
  });
});
