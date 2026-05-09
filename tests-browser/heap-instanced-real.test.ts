// Per-RO instancing through the heap path — real GPU.
//
// Mirrors `instancing-real.test.ts` but routes the 4-instance triangle
// through `buildHeapScene` with megacall + per-RO instance attributes.
// One bucket, one drawIndirect — verified by inspecting the framebuffer
// pixels for the four expected colors at four expected quadrants.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken } from "@aardworx/wombat.adaptive";
import { parseShader } from "@aardworx/wombat.shader/frontend";
import { stage, type Effect } from "@aardworx/wombat.shader";
import {
  Tf32, Vec, type Module, type Type, type ValueDef,
} from "@aardworx/wombat.shader/ir";
import {
  buildHeapScene,
  type HeapDrawSpec,
} from "@aardworx/wombat.rendering.experimental/runtime";
import { createFramebufferSignature } from "@aardworx/wombat.rendering.experimental/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";

const Tvec2f: Type = Vec(Tf32, 2);
const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

function instancedHeapEffect(): Effect {
  // Per-vertex Position (vec2) + per-instance Offset (vec2) + Tint (vec3).
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

describe("heap per-RO instancing — real GPU", () => {
  it("renders 4 instances of a quad through the heap path (one bucket, one drawIndirect)", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      // Quad as triangle list (6 verts → 6 indices), bottom-left
      // quadrant in clip space [-1,-1]..[0,0].
      const positions = new Float32Array([
        -1, -1,   0, -1,  -1,  0,
         0, -1,   0,  0,  -1,  0,
      ]);
      // Per-instance offsets (vec2) — 4 quadrants.
      const offsets = new Float32Array([
        0, 0,        // bottom-left
        1, 0,        // bottom-right
        0, 1,        // top-left
        1, 1,        // top-right
      ]);
      // Per-instance tints (vec3) — 4 distinct colors.
      const tints = new Float32Array([
        1, 0, 0,     // red
        0, 1, 0,     // green
        0, 0, 1,     // blue
        1, 1, 0,     // yellow
      ]);
      const indices = new Uint32Array([0, 1, 2, 3, 4, 5]);

      const sig = createFramebufferSignature({
        colors: { outColor: "rgba8unorm" },
        depthStencil: { format: "depth24plus" },
      });

      const spec: HeapDrawSpec = {
        effect: instancedHeapEffect(),
        inputs: { Position: AVal.constant(positions) },
        instanceAttributes: {
          Offset: AVal.constant(offsets),
          Tint:   AVal.constant(tints),
        },
        instanceCount: 4,
        indices: AVal.constant(indices),
      };

      const scene = buildHeapScene(device, sig, [spec], {
        fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
      });

      // One bucket, one indirect record (verified through the test
      // debug surface).
      const debug = (scene as unknown as { _debug: { bucketsForTest(): readonly {
        recordCount: number;
        totalEmitEstimate: number;
      }[] } })._debug;
      const dbg = debug.bucketsForTest();
      expect(dbg.length).toBe(1);
      expect(dbg[0]!.recordCount).toBe(1);
      expect(dbg[0]!.totalEmitEstimate).toBe(6 * 4);  // indices * instances

      // Render at 8×8 and read back pixels.
      const W = 8, H = 8;
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

      scene.update(AdaptiveToken.top);
      const enc = device.createCommandEncoder({ label: "heap-inst-real" });
      scene.encodeComputePrep(enc, AdaptiveToken.top);
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: colorTex.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear", storeOp: "store",
        }],
        depthStencilAttachment: {
          view: depthTex.createView(),
          depthClearValue: 1.0,
          depthLoadOp: "clear", depthStoreOp: "store",
        },
      });
      scene.encodeIntoPass(pass);
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();

      const px = await readTexturePixels(device, colorTex);
      const at = (x: number, y: number, c: 0 | 1 | 2 | 3) => px[4 * (y * W + x) + c]!;

      // Expected layout (matches `instancing-real.test.ts`):
      //   instance 0 (offset 0,0) → bottom-left, RED
      //   instance 1 (offset 1,0) → bottom-right, GREEN
      //   instance 2 (offset 0,1) → top-left,    BLUE
      //   instance 3 (offset 1,1) → top-right,   YELLOW
      // Bottom-left interior pixel: red.
      expect(at(1, 6, 0)).toBeGreaterThan(200);
      expect(at(1, 6, 1)).toBeLessThan(40);
      expect(at(1, 6, 2)).toBeLessThan(40);
      // Bottom-right: green.
      expect(at(5, 6, 0)).toBeLessThan(40);
      expect(at(5, 6, 1)).toBeGreaterThan(200);
      expect(at(5, 6, 2)).toBeLessThan(40);
      // Top-left: blue.
      expect(at(1, 1, 0)).toBeLessThan(40);
      expect(at(1, 1, 1)).toBeLessThan(40);
      expect(at(1, 1, 2)).toBeGreaterThan(200);
      // Top-right: yellow.
      expect(at(5, 1, 0)).toBeGreaterThan(200);
      expect(at(5, 1, 1)).toBeGreaterThan(200);
      expect(at(5, 1, 2)).toBeLessThan(40);

      expect(errors).toEqual([]);

      scene.dispose();
      colorTex.destroy();
      depthTex.destroy();
    } finally {
      device.destroy();
    }
  }, 30000);
});
