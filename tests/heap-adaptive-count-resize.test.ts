// Adaptive instanceCount + sized repack (geometry morphing): the
// epoch-switching / point-add path. A count tick is a drawTable write;
// a payload size change reallocs the arena slot and re-seats the
// drawHeader refs — no add/remove churn either way.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, cval, transact } from "@aardworx/wombat.adaptive";
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

type DebugScene = { _debug: { bucketsForTest(): readonly {
  recordCount: number; totalEmitEstimate: number;
}[] } };

const totalEmit = (scene: unknown): number =>
  (scene as DebugScene)._debug.bucketsForTest().reduce((s, b) => s + b.totalEmitEstimate, 0);

describe("adaptive instanceCount + sized repack (mock GPU)", () => {
  it("count ticks rewrite the drawTable in place (no add/remove)", () => {
    const gpu = new MockGPU();
    const effect = instancedHeapEffect();
    const positions = new Float32Array([-0.5, -0.5,  0.5, -0.5,  0, 0.5]);
    const count = cval(2);
    const spec: HeapDrawSpec = {
      effect,
      inputs: { Position: AVal.constant(positions) },
      instanceAttributes: {
        Offset: AVal.constant(new Float32Array([0, 0, 1, 0, 2, 0, 3, 0])),
        Tint:   AVal.constant(new Float32Array(12)),
      },
      instanceCount: count,
      indices: AVal.constant(new Uint32Array([0, 1, 2])),
    };
    const scene = buildHeapScene(gpu.device, sig(), [spec], {
      fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
    });
    scene.update(AdaptiveToken.top);
    expect(totalEmit(scene)).toBe(6);        // 3 idx × 2

    transact(() => { count.value = 4; });
    scene.update(AdaptiveToken.top);
    expect(totalEmit(scene)).toBe(12);       // 3 × 4

    transact(() => { count.value = 0; });
    scene.update(AdaptiveToken.top);
    expect(totalEmit(scene)).toBe(0);

    transact(() => { count.value = 3; });
    scene.update(AdaptiveToken.top);
    expect(totalEmit(scene)).toBe(9);
  });

  it("payload growth reallocs and re-seats refs (morphing geometry)", () => {
    const gpu = new MockGPU();
    const effect = instancedHeapEffect();
    const positions = new Float32Array([-0.5, -0.5,  0.5, -0.5,  0, 0.5]);
    const count = cval(2);
    const offsets = cval(new Float32Array([0, 0, 1, 0]));
    const tints = cval(new Float32Array([1, 0, 0, 0, 1, 0]));
    const spec: HeapDrawSpec = {
      effect,
      inputs: { Position: AVal.constant(positions) },
      instanceAttributes: { Offset: offsets, Tint: tints },
      instanceCount: count,
      indices: AVal.constant(new Uint32Array([0, 1, 2])),
    };
    const scene = buildHeapScene(gpu.device, sig(), [spec], {
      fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
    });
    scene.update(AdaptiveToken.top);
    expect(totalEmit(scene)).toBe(6);

    // Epoch switch: DIFFERENT geometry — more instances, bigger buffers.
    transact(() => {
      offsets.value = new Float32Array([0, 0, 1, 0, 2, 0, 3, 0, 4, 0]);
      tints.value   = new Float32Array(15).fill(0.5);
      count.value   = 5;
    });
    scene.update(AdaptiveToken.top);
    expect(totalEmit(scene)).toBe(15);       // 3 × 5

    // And back down (shrink).
    transact(() => {
      offsets.value = new Float32Array([0, 0]);
      tints.value   = new Float32Array([1, 1, 1]);
      count.value   = 1;
    });
    scene.update(AdaptiveToken.top);
    expect(totalEmit(scene)).toBe(3);
  });
});
