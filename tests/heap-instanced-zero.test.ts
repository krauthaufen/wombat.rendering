// instanceCount = 0 on the heap path — a row-shaped RO whose adaptive
// count is currently zero (e.g. a one-point annotation with no
// segments) must emit NOTHING, not throw and not draw a garbage
// instance. Legacy renders 0 instances as nothing; the heap path must
// match for the producer-asserted eligibility path to be safe.

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

describe("heap per-RO instancing — instanceCount 0 (mock GPU)", () => {
  it("a zero-instance spec beside a live one emits only the live draws", () => {
    const gpu = new MockGPU();
    const effect = instancedHeapEffect();
    const positions = new Float32Array([-0.5, -0.5,  0.5, -0.5,  0, 0.5]);

    const zero: HeapDrawSpec = {
      effect,
      inputs: { Position: AVal.constant(positions) },
      instanceAttributes: {
        Offset: AVal.constant(new Float32Array(0)),
        Tint:   AVal.constant(new Float32Array(0)),
      },
      instanceCount: 0,
      indices: AVal.constant(new Uint32Array([0, 1, 2])),
    };
    const live: HeapDrawSpec = {
      effect,
      inputs: { Position: AVal.constant(positions) },
      instanceAttributes: {
        Offset: AVal.constant(new Float32Array([0, 0, 1, 0])),
        Tint:   AVal.constant(new Float32Array([1, 0, 0, 0, 1, 0])),
      },
      instanceCount: 2,
      indices: AVal.constant(new Uint32Array([0, 1, 2])),
    };

    const scene = buildHeapScene(gpu.device, sig(), [zero, live], {
      fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
    });
    scene.update(AdaptiveToken.top);

    const debug = (scene as unknown as { _debug: { bucketsForTest(): readonly {
      recordCount: number;
      totalEmitEstimate: number;
    }[] } })._debug;
    const bs = debug.bucketsForTest();
    const totalEmit = bs.reduce((s, b) => s + b.totalEmitEstimate, 0);
    // live: 3 indices × 2 instances = 6; zero contributes nothing.
    expect(totalEmit).toBe(6);
  });
});
