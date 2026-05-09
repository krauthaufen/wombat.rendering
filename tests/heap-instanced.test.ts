// Per-RO instancing in the heap path (mock GPU).
//
// Verifies the plumbing for the new shape (HeapDrawSpec.instanceAttributes
// + .instanceCount, drawTable record carries instanceCount, drawHeader
// has refs for the per-instance attributes). Pixel correctness lives in
// `tests-browser/heap-instanced-real.test.ts`.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken } from "@aardworx/wombat.adaptive";
import { parseShader } from "@aardworx/wombat.shader/frontend";
import { stage, type Effect } from "@aardworx/wombat.shader";
import {
  Tf32, Vec, type Module, type Type, type ValueDef,
} from "@aardworx/wombat.shader/ir";
import {
  buildHeapScene, type HeapDrawSpec,
} from "../packages/rendering/src/runtime/heapScene.js";
import { createFramebufferSignature } from "../packages/rendering/src/resources/framebufferSignature.js";
import { MockGPU } from "./_mockGpu.js";

if (typeof (globalThis as { GPUTextureUsage?: unknown }).GPUTextureUsage === "undefined") {
  (globalThis as Record<string, unknown>).GPUTextureUsage = {
    COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04,
    STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10,
  };
}
if (typeof (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage === "undefined") {
  (globalThis as Record<string, unknown>).GPUBufferUsage = {
    MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
    INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
    INDIRECT: 0x0100, QUERY_RESOLVE: 0x0200,
  };
}
if (typeof (globalThis as { GPUShaderStage?: unknown }).GPUShaderStage === "undefined") {
  (globalThis as Record<string, unknown>).GPUShaderStage = {
    VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4,
  };
}

const Tvec2f: Type = Vec(Tf32, 2);
const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

// Minimal effect: per-vertex Position (vec2) + per-instance Offset (vec2)
// + per-instance Tint (vec3). VS sums position+offset and threads tint;
// FS writes tint. No uniforms.
function instancedHeapEffect(): Effect {
  const source = `
    function vsMain(input: { Position: V2f; Offset: V2f; Tint: V3f }): {
      gl_Position: V4f; v_color: V3f;
    } {
      return {
        gl_Position: new V4f(input.Position.x + input.Offset.x, input.Position.y + input.Offset.y, 0.0, 1.0),
        v_color: input.Tint,
      };
    }
    function fsMain(input: { v_color: V3f }): { outColor: V4f } {
      return { outColor: new V4f(input.v_color.x, input.v_color.y, input.v_color.z, 1.0) };
    }
  `;
  const parsed = parseShader({
    source,
    entries: [
      {
        name: "vsMain", stage: "vertex",
        inputs: [
          { name: "Position", type: Tvec2f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
          { name: "Offset",   type: Tvec2f, semantic: "Position", decorations: [{ kind: "Location", value: 1 }] },
          { name: "Tint",     type: Tvec3f, semantic: "Color",    decorations: [{ kind: "Location", value: 2 }] },
        ],
        outputs: [
          { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
          { name: "v_color",     type: Tvec3f, semantic: "Color",    decorations: [{ kind: "Location", value: 0 }] },
        ],
      },
      {
        name: "fsMain", stage: "fragment",
        inputs:  [{ name: "v_color",  type: Tvec3f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ],
  });
  const merged: Module = { ...parsed, values: [...([] as ValueDef[]), ...parsed.values] };
  return stage(merged);
}

const sig = () => createFramebufferSignature({
  colors: { outColor: "rgba8unorm" },
  depthStencil: { format: "depth24plus" },
});

describe("heap per-RO instancing — bucket plumbing (mock GPU)", () => {
  it("single RO with instanceCount=4 produces 1 bucket and 1 indirect record", () => {
    const gpu = new MockGPU();
    const effect = instancedHeapEffect();

    // Triangle, 3 indices over 3 verts.
    const positions = new Float32Array([-0.5, -0.5,  0.5, -0.5,  0, 0.5]);
    const offsets = new Float32Array([
      0, 0,
      1, 0,
      0, 1,
      1, 1,
    ]);
    const tints = new Float32Array([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
      1, 1, 0,
    ]);

    const spec: HeapDrawSpec = {
      effect,
      inputs: {
        Position: AVal.constant(positions),
      },
      instanceAttributes: {
        Offset: AVal.constant(offsets),
        Tint:   AVal.constant(tints),
      },
      instanceCount: 4,
      indices: AVal.constant(new Uint32Array([0, 1, 2])),
    };

    const scene = buildHeapScene(gpu.device, sig(), [spec], {
      fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
    });
    // Drive update so the staged drawTable record is written.
    scene.update(AdaptiveToken.top);

    // Exactly one bucket — the per-RO instancing path doesn't fan out
    // into multiple buckets the way the legacy `instances` shape does.
    expect(scene.stats.groups).toBe(1);

    // Test-only debug surface — confirm the drawTable record carries
    // (indexCount, instanceCount).
    const debug = (scene as unknown as { _debug: { bucketsForTest(): readonly {
      recordCount: number;
      totalEmitEstimate: number;
      drawTableBuf: GPUBuffer | undefined;
      indirectBuf:  GPUBuffer | undefined;
    }[] } })._debug;
    const bs = debug.bucketsForTest();
    expect(bs.length).toBe(1);
    const b = bs[0]!;
    expect(b.recordCount).toBe(1);
    // indexCount=3, instanceCount=4 → totalEmit = 12.
    expect(b.totalEmitEstimate).toBe(12);

    // Confirm the drawTable shadow buffer was written with the
    // (firstEmit, drawIdx, indexStart, indexCount, instanceCount)
    // record. We inspect the writeBuffer recorded for the drawTable.
    const dtBuf = b.drawTableBuf!;
    const writes = gpu.writeBufferCalls.filter(w => (w.buffer as unknown as GPUBuffer) === dtBuf);
    expect(writes.length).toBeGreaterThan(0);
    const last = writes[writes.length - 1]!;
    // writeBuffer(buffer, bufferOffset, data, dataOffset, size). `data` may
    // be an ArrayBuffer or a typed-array view; normalise to a u32 view.
    const dataAB: ArrayBuffer = last.data instanceof ArrayBuffer
      ? last.data
      : (last.data as ArrayBufferView).buffer;
    const dataBaseOff: number = last.data instanceof ArrayBuffer
      ? 0
      : (last.data as ArrayBufferView).byteOffset;
    const u32 = new Uint32Array(dataAB, dataBaseOff + last.dataOffset, last.size / 4);
    // record = [firstEmit, drawIdx, indexStart, indexCount, instanceCount]
    expect(u32[3]).toBe(3);   // indexCount
    expect(u32[4]).toBe(4);   // instanceCount
  });

});
