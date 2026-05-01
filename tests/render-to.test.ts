// renderTo lifecycle: acquiring the derived `aval<ITexture>`
// brings the FBO live; releasing tears it down. Pixel-correctness
// of the inner-scene-into-FBO render lives in
// tests-browser/renderto-real.test.ts.

import { describe, expect, it } from "vitest";
import { AdaptiveToken, HashMap, cval, type aval } from "@aardworx/wombat.adaptive";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import {
  IBuffer,
  RenderTree,
  type BufferView,
  type DrawCall,
  type RenderObject,
} from "@aardworx/wombat.rendering/core";
import {
  createFramebufferSignature,
} from "@aardworx/wombat.rendering/resources";
import { Runtime } from "@aardworx/wombat.rendering/runtime";
import { makeEffect } from "./_makeEffect.js";
import { MockGPU } from "./_mockGpu.js";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

function passEffect() {
  return makeEffect(
    `
      function vsMain(input: { position: V3f }): { gl_Position: V4f } {
        return { gl_Position: new V4f(input.position.x, input.position.y, input.position.z, 1.0) };
      }
      function fsMain(_input: {}): { outColor: V4f } {
        return { outColor: new V4f(1.0, 0.0, 0.0, 1.0) };
      }
    `,
    [
      { name: "vsMain", stage: "vertex",
        inputs: [{ name: "position", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] }],
      },
      { name: "fsMain", stage: "fragment",
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ],
  );
}

function bv(): aval<BufferView> {
  return cval<BufferView>({
    buffer: IBuffer.fromHost(new ArrayBuffer(36)),
    offset: 0, count: 3, stride: 12, format: "float32x3",
  });
}

describe("renderTo lifecycle", () => {
  it("acquire on derived aval brings the FBO live; release tears it down", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device });
    const innerObj: RenderObject = {
      effect: passEffect(),
      pipelineState: { rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } },
      vertexAttributes: HashMap.empty<string, aval<BufferView>>().add("position", bv()),
      uniforms: HashMap.empty(),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };
    const result = runtime.renderTo(RenderTree.leaf(innerObj), {
      size: cval({ width: 16, height: 16 }),
      signature: createFramebufferSignature({ colors: { outColor: "rgba8unorm" } }),
    });
    const colorAval = result.color("outColor");
    expect(gpu.textures).toHaveLength(0);
    colorAval.acquire();
    expect(gpu.textures).toHaveLength(0);  // lazy until first compute
    colorAval.getValue(AdaptiveToken.top);
    expect(gpu.textures).toHaveLength(1);
    colorAval.release();
    expect(gpu.textures[0]!.destroyed).toBe(true);
  });
});
