// Hello-triangle on real GPU. Compiles a vertex+fragment effect with
// real wombat.shader, renders one fullscreen triangle into a 4×4
// framebuffer, reads pixels back, asserts the center is the
// vertex color and the corners are the clear color.

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
  PipelineState,
} from "@aardworx/wombat.rendering/core";
import {
  allocateFramebuffer,
  BufferUsage,
  createFramebufferSignature,
  TextureUsage,
} from "@aardworx/wombat.rendering/resources";
import { Runtime } from "@aardworx/wombat.rendering/runtime";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";

const Tvec2f: Type = Vec(Tf32, 2);
const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

function helloTriangleEffect(): Effect {
  const source = `
    function vsMain(input: { a_position: V2f; a_color: V3f }): { gl_Position: V4f; v_color: V3f } {
      return {
        gl_Position: new V4f(input.a_position.x, input.a_position.y, 0.0, 1.0),
        v_color: input.a_color,
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
        { name: "a_color",    type: Tvec3f, semantic: "Color",    decorations: [{ kind: "Location", value: 1 }] },
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

function f32Buffer(values: number[]): aval<BufferView> {
  const arr = new Float32Array(values);
  return cval<BufferView>({
    buffer: IBuffer.fromHost(arr),
    offset: 0,
    count: values.length,
    stride: 0,  // unused for our setup; layout determined by shader format
    format: "float32",
  });
}

describe("hello-triangle — real GPU", () => {
  it("renders a triangle with per-vertex colors; pixels match", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      // Fullscreen-ish triangle (clip-space). Three vertices:
      //   v0 = (-1, -1) red
      //   v1 = ( 3, -1) green
      //   v2 = (-1,  3) blue
      // This covers the entire 4x4 viewport.
      const positions = new Float32Array([
        -1, -1,
         3, -1,
        -1,  3,
      ]);
      const colors = new Float32Array([
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
      ]);

      const obj: RenderObject = {
        effect: helloTriangleEffect(),
        pipelineState: PipelineState.constant({ rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } }),
        vertexAttributes: HashMap.empty<string, aval<BufferView>>()
          .add("a_position", cval<BufferView>({
            buffer: IBuffer.fromHost(positions), offset: 0, count: 3, stride: 8, format: "float32x2",
          }))
          .add("a_color", cval<BufferView>({
            buffer: IBuffer.fromHost(colors), offset: 0, count: 3, stride: 12, format: "float32x3",
          })),
        uniforms: HashMap.empty(),
        textures: HashMap.empty(),
        samplers: HashMap.empty(),
        drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
      };

      const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
      const fbo = allocateFramebuffer(device, sig, cval({ width: 4, height: 4 }), {
        extraUsage: TextureUsage.COPY_SRC,
      });
      fbo.acquire();
      const clearValues: ClearValues = {
        colors: HashMap.empty<string, V4f>().add("outColor", new V4f(0, 0, 0, 1)),
      };

      const runtime = new Runtime({ device });
      const cmds = AList.ofArray<Command>([
        { kind: "Clear",  output: fbo, values: clearValues },
        { kind: "Render", output: fbo, tree: RenderTree.leaf(obj) },
      ]);
      const task = runtime.compile(cmds);
      task.run(AdaptiveToken.top);
      await device.queue.onSubmittedWorkDone();
      expect(errors).toEqual([]);

      const ifb = fbo.getValue(AdaptiveToken.top);
      const tex = ifb.colorTextures!.tryFind("outColor")!;
      const pixels = await readTexturePixels(device, tex);

      // 4×4 rgba8 = 64 bytes. Triangle vertices in clip space:
      //   v0(-1,-1) red, v1(3,-1) green, v2(-1,3) blue.
      // WebGPU clip-y +1 maps to framebuffer top, -1 to bottom.
      // So:
      //   bottom-left  pixel ≈ clip(-1,-1)  → v0 (red)
      //   bottom-right pixel ≈ clip(+1,-1)  → 50% red + 50% green
      //   top-left     pixel ≈ clip(-1,+1)  → 50% red + 50% blue
      //   top-right    pixel ≈ clip(+1,+1)  → 50% green + 50% blue
      expect(pixels.length).toBe(64);

      const W = 4;
      const px = (x: number, y: number) => 4 * (y * W + x);

      // Bottom-left = v0 = pure red.
      expect(pixels[px(0, 3) + 0]!).toBeGreaterThan(200);
      expect(pixels[px(0, 3) + 1]!).toBeLessThan(60);
      expect(pixels[px(0, 3) + 2]!).toBeLessThan(60);

      // Top-left = mix of red + blue (~128, 0, 128).
      expect(pixels[px(0, 0) + 0]!).toBeGreaterThan(80);
      expect(pixels[px(0, 0) + 0]!).toBeLessThan(180);
      expect(pixels[px(0, 0) + 1]!).toBeLessThan(40);
      expect(pixels[px(0, 0) + 2]!).toBeGreaterThan(80);
      expect(pixels[px(0, 0) + 2]!).toBeLessThan(180);

      // Top-right = mix of green + blue (~0, 128, 128).
      expect(pixels[px(3, 0) + 0]!).toBeLessThan(40);
      expect(pixels[px(3, 0) + 1]!).toBeGreaterThan(80);
      expect(pixels[px(3, 0) + 2]!).toBeGreaterThan(80);

      // All alphas = 255.
      for (let i = 3; i < 64; i += 4) expect(pixels[i]!).toBe(255);

      task.dispose();
      fbo.release();
    } finally {
      device.destroy();
    }
  }, 30000);
});

void f32Buffer;
