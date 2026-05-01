// Instance attributes — real GPU. Renders the same triangle 4
// times, each instance offset to a different quadrant of an 8×8
// FBO and given a different per-instance color. Verifies that:
//   - prepareRenderObject puts the per-instance attribute into
//     a vertex-buffer layout with stepMode: "instance".
//   - WebGPU consumes it correctly (correct offsets + colors).

import { describe, expect, it } from "vitest";
import { AList, AdaptiveToken, HashMap, cval, type aval } from "@aardworx/wombat.adaptive";
import { V4f } from "@aardworx/wombat.base";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { stage, type Effect } from "@aardworx/wombat.shader";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import {
  IBuffer,
  RenderTree,
  type BufferView,
  type ClearValues,
  type Command,
  type DrawCall,
  type RenderObject,
} from "@aardworx/wombat.rendering-core";
import {
  TextureUsage,
  allocateFramebuffer,
  createFramebufferSignature,
} from "@aardworx/wombat.rendering-resources";
import { Runtime } from "@aardworx/wombat.rendering-runtime";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";

const Tvec2f: Type = Vec(Tf32, 2);
const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

function instancedEffect(): Effect {
  // Per-vertex position, per-instance offset (vec2) + tint (vec3).
  const source = `
    function vsMain(input: { a_position: V2f; i_offset: V2f; i_tint: V3f }): { gl_Position: V4f; v_color: V3f } {
      return {
        gl_Position: new V4f(input.a_position.x + input.i_offset.x, input.a_position.y + input.i_offset.y, 0.0, 1.0),
        v_color: input.i_tint,
      };
    }
    function fsMain(input: { v_color: V3f }): { outColor: V4f } {
      return { outColor: new V4f(input.v_color.x, input.v_color.y, input.v_color.z, 1.0) };
    }
  `;
  const entries: EntryRequest[] = [
    {
      name: "vsMain", stage: "vertex",
      inputs: [
        { name: "a_position", type: Tvec2f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "i_offset",   type: Tvec2f, semantic: "Position", decorations: [{ kind: "Location", value: 1 }] },
        { name: "i_tint",     type: Tvec3f, semantic: "Color",    decorations: [{ kind: "Location", value: 2 }] },
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
  ];
  return stage(parseShader({ source, entries }));
}

describe("instancing — real GPU", () => {
  it("renders 4 instances of a small triangle to 4 quadrants with different colors", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      // Quad (6 verts) covering bottom-left quadrant clip(-1,-1)..(0,0).
      // Each instance translates it to a different quadrant.
      const positions = new Float32Array([
        -1, -1,   0, -1,  -1,  0,
         0, -1,   0,  0,  -1,  0,
      ]);
      const offsets = new Float32Array([
        0, 0,        // bottom-left
        1, 0,        // bottom-right
        0, 1,        // top-left
        1, 1,        // top-right
      ]);
      const tints = new Float32Array([
        1, 0, 0,     // red
        0, 1, 0,     // green
        0, 0, 1,     // blue
        1, 1, 0,     // yellow
      ]);

      const obj: RenderObject = {
        effect: instancedEffect(),
        pipelineState: { rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } },
        vertexAttributes: HashMap.empty<string, aval<BufferView>>()
          .add("a_position", cval<BufferView>({
            buffer: IBuffer.fromHost(positions), offset: 0, count: 6, stride: 8, format: "float32x2",
          })),
        instanceAttributes: HashMap.empty<string, aval<BufferView>>()
          .add("i_offset", cval<BufferView>({
            buffer: IBuffer.fromHost(offsets), offset: 0, count: 4, stride: 8, format: "float32x2",
          }))
          .add("i_tint", cval<BufferView>({
            buffer: IBuffer.fromHost(tints), offset: 0, count: 4, stride: 12, format: "float32x3",
          })),
        uniforms: HashMap.empty(),
        textures: HashMap.empty(),
        samplers: HashMap.empty(),
        drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 6, instanceCount: 4, firstVertex: 0, firstInstance: 0 }),
      };

      const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
      const fbo = allocateFramebuffer(device, sig, cval({ width: 8, height: 8 }), {
        extraUsage: TextureUsage.COPY_SRC,
      });
      fbo.acquire();

      const runtime = new Runtime({ device });
      const cmds = AList.ofArray<Command>([
        { kind: "Clear",  output: fbo, values: { colors: HashMap.empty<string, V4f>().add("outColor", new V4f(0, 0, 0, 1)) } as ClearValues },
        { kind: "Render", output: fbo, tree: RenderTree.leaf(obj) },
      ]);
      runtime.compile(cmds).run(AdaptiveToken.top);
      await device.queue.onSubmittedWorkDone();

      const ifb = fbo.getValue(AdaptiveToken.top);
      const tex = ifb.colorTextures!.tryFind("outColor")!;
      const px = await readTexturePixels(device, tex);
      const W = 8;
      const at = (x: number, y: number, c: 0 | 1 | 2 | 3) => px[4 * (y * W + x) + c]!;

      // Instance 0 (offset 0,0) covers bottom-left quadrant, RED.
      //   Clip-space y∈[-1,-0.05] in framebuffer terms = bottom rows (y∈[4,7]).
      //   Clip-space x∈[-1,-0.05] = left cols (x∈[0,3]).
      // Instance 1 (offset 1,0): bottom-right, GREEN.
      // Instance 2 (offset 0,1): top-left, BLUE.
      // Instance 3 (offset 1,1): top-right, YELLOW.

      // Sample one pixel from each quadrant.
      // Bottom-left interior pixel: should be red.
      expect(at(1, 6, 0)).toBeGreaterThan(200);
      expect(at(1, 6, 1)).toBeLessThan(40);
      expect(at(1, 6, 2)).toBeLessThan(40);

      // Bottom-right interior pixel: should be green.
      expect(at(5, 6, 0)).toBeLessThan(40);
      expect(at(5, 6, 1)).toBeGreaterThan(200);
      expect(at(5, 6, 2)).toBeLessThan(40);

      // Top-left interior pixel: should be blue.
      expect(at(1, 1, 0)).toBeLessThan(40);
      expect(at(1, 1, 1)).toBeLessThan(40);
      expect(at(1, 1, 2)).toBeGreaterThan(200);

      // Top-right interior pixel: should be yellow.
      expect(at(5, 1, 0)).toBeGreaterThan(200);
      expect(at(5, 1, 1)).toBeGreaterThan(200);
      expect(at(5, 1, 2)).toBeLessThan(40);

      expect(errors).toEqual([]);
      fbo.release();
    } finally {
      device.destroy();
    }
  }, 30000);
});
