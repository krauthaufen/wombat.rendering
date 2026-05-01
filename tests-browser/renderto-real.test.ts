// renderTo on real GPU. Builds a renderTo result for a 4×4 FBO,
// acquires a derived aval<ITexture>, runs the inner scene
// implicitly (the result aval evaluates inside an outer encoder
// scope), reads the FBO back, asserts pixels.

import { describe, expect, it } from "vitest";
import { AdaptiveToken, HashMap, cval, type aval } from "@aardworx/wombat.adaptive";
import { V4f } from "@aardworx/wombat.base";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { stage, type Effect } from "@aardworx/wombat.shader";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import {
  IBuffer,
  RenderContext,
  RenderTree,
  type BufferView,
  type ClearValues,
  type DrawCall,
  type RenderObject,
} from "@aardworx/wombat.rendering/core";
import {
  TextureUsage,
  createFramebufferSignature,
} from "@aardworx/wombat.rendering/resources";
import { Runtime } from "@aardworx/wombat.rendering/runtime";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";

const Tvec2f: Type = Vec(Tf32, 2);
const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

function fullscreenColorEffect(): Effect {
  const source = `
    function vsMain(input: { a_position: V2f }): { gl_Position: V4f } {
      return { gl_Position: new V4f(input.a_position.x, input.a_position.y, 0.0, 1.0) };
    }
    function fsMain(_input: {}): { outColor: V4f } {
      return { outColor: new V4f(0.2, 0.8, 0.4, 1.0) };
    }
  `;
  const entries: EntryRequest[] = [
    {
      name: "vsMain", stage: "vertex",
      inputs: [{ name: "a_position", type: Tvec2f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] }],
    },
    {
      name: "fsMain", stage: "fragment",
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    },
  ];
  void Tvec3f;
  return stage(parseShader({ source, entries }));
}

describe("renderTo — real GPU", () => {
  it("offscreen FBO is allocated lazily, encodes inner scene, contains expected color", async () => {
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const positions = new Float32Array([-1, -1,  3, -1, -1, 3]);
      const obj: RenderObject = {
        effect: fullscreenColorEffect(),
        pipelineState: { rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } },
        vertexAttributes: HashMap.empty<string, aval<BufferView>>().add("a_position", cval<BufferView>({
          buffer: IBuffer.fromHost(positions), offset: 0, count: 3, stride: 8, format: "float32x2",
        })),
        uniforms: HashMap.empty(),
        textures: HashMap.empty(),
        samplers: HashMap.empty(),
        drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
      };

      const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
      const runtime = new Runtime({ device });
      const result = runtime.renderTo(RenderTree.leaf(obj), {
        size: cval({ width: 4, height: 4 }),
        signature: sig,
        clear: { colors: HashMap.empty<string, V4f>().add("outColor", new V4f(0, 0, 0, 1)) } as ClearValues,
        extraUsage: TextureUsage.COPY_SRC,
      });
      const colorAval = result.color("outColor");
      colorAval.acquire();

      // Drive a frame by hand: open an encoder, evaluate the aval inside
      // it (which encodes the inner clear+render into the same encoder),
      // submit, read pixels back from the FBO's color texture.
      const enc = device.createCommandEncoder();
      let tex!: GPUTexture;
      RenderContext.withEncoder(enc, () => {
        const itex = colorAval.getValue(AdaptiveToken.top);
        if (itex.kind !== "gpu") throw new Error("renderTo result must be ITexture.fromGPU");
        tex = itex.texture;
      });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      const pixels = await readTexturePixels(device, tex);

      expect(errors).toEqual([]);
      // (0.2, 0.8, 0.4, 1.0) → (51, 204, 102, 255) ± 1.
      for (let i = 0; i < 16; i++) {
        const r = pixels[i * 4 + 0]!;
        const g = pixels[i * 4 + 1]!;
        const b = pixels[i * 4 + 2]!;
        const a = pixels[i * 4 + 3]!;
        expect(Math.abs(r - 51)).toBeLessThan(3);
        expect(Math.abs(g - 204)).toBeLessThan(3);
        expect(Math.abs(b - 102)).toBeLessThan(3);
        expect(a).toBe(255);
      }

      colorAval.release();
    } finally {
      device.destroy();
    }
  }, 30000);
});
