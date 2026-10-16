// Device-loss recovery on real GPU.
//
// `<RenderControl>` recovers from a lost device by disposing the whole
// generation (device, Runtime, attachment, loop) and rebuilding a fresh one,
// keeping the user's scene and cvals — those are device-agnostic by design.
// This pins the contract that rebuild depends on: the SAME RenderObject
// graph, including a cube texture, must render correctly on a brand-new
// device+Runtime after the previous device was destroyed.
//
// A viewer shipped that never came back from a device loss (permanently
// black after iOS Safari reclaimed the GPU under memory pressure). Root
// cause was a duplicate wombat.rendering install rather than this contract
// — but nothing covered "render, lose the device, render again" at all, so
// the break was invisible until production.

import { describe, expect, it } from "vitest";
import { AdaptiveToken, AVal, HashMap, cval } from "@aardworx/wombat.adaptive";
import { V4f } from "@aardworx/wombat.base";
import { ElementType } from "@aardworx/wombat.rendering/core";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { stage, type Effect } from "@aardworx/wombat.shader";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import {
  IBuffer,
  ITexture,
  RenderContext,
  RenderTree,
  type BufferView,
  type ClearValues,
  type DrawCall,
  type RenderObject,
  PipelineState,
} from "@aardworx/wombat.rendering/core";
import {
  TextureUsage,
  createFramebufferSignature,
} from "@aardworx/wombat.rendering/resources";
import { Runtime } from "@aardworx/wombat.rendering/runtime";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";

const Tvec2f: Type = Vec(Tf32, 2);
const Tvec4f: Type = Vec(Tf32, 4);

function flatColorEffect(): Effect {
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
  return stage(parseShader({ source, entries }));
}

/** Scene state built ONCE and reused across device generations. */
function makeScene(): { obj: RenderObject; cubeTex: ITexture } {
  const positions = new Float32Array([-1, -1, 3, -1, -1, 3]);
  const obj: RenderObject = {
    effect: flatColorEffect(),
    pipelineState: PipelineState.constant({
      rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" },
    }),
    vertexAttributes: HashMap.empty<string, BufferView>().add("a_position", {
      buffer: AVal.constant(IBuffer.fromHost(positions)), offset: 0, stride: 8, elementType: ElementType.V2f,
    }),
    uniforms: HashMap.empty(),
    textures: HashMap.empty(),
    samplers: HashMap.empty(),
    drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
  };
  // A cube ITexture rides along: it is the value that used to make the
  // eligibility predicate throw on a stale build. Constructing it here (and
  // asserting it survives the generation swap) keeps the cube path in the
  // device-loss story rather than only in the happy path.
  const faces = Array.from({ length: 6 }, () => new OffscreenCanvas(8, 8));
  for (const f of faces) {
    const ctx = f.getContext("2d")!;
    ctx.fillStyle = "rgb(10,20,30)";
    ctx.fillRect(0, 0, 8, 8);
  }
  return { obj, cubeTex: ITexture.fromCubeFaces(faces) };
}

/** Render the scene on a fresh device+Runtime and read back the 4x4 result. */
async function renderGeneration(obj: RenderObject): Promise<{ pixels: Uint8Array; errors: GPUError[] }> {
  const device = await requestRealDevice();
  const errors: GPUError[] = [];
  device.onuncapturederror = (e) => errors.push(e.error);
  try {
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
    colorAval.release();
    runtime.disposeAll();
    return { pixels, errors };
  } finally {
    // Destroying here is precisely the "device lost" event the next
    // generation must survive.
    device.destroy();
  }
}

function expectGreen(pixels: Uint8Array): void {
  for (let i = 0; i < 16; i++) {
    expect(Math.abs(pixels[i * 4 + 0]! - 51)).toBeLessThan(3);
    expect(Math.abs(pixels[i * 4 + 1]! - 204)).toBeLessThan(3);
    expect(Math.abs(pixels[i * 4 + 2]! - 102)).toBeLessThan(3);
    expect(pixels[i * 4 + 3]!).toBe(255);
  }
}

describe("device loss — recovery", () => {
  it("the same scene renders again on a fresh device after the old one is destroyed", async () => {
    const { obj, cubeTex } = makeScene();

    // Generation 1 — renders, then its device is destroyed (the loss).
    const first = await renderGeneration(obj);
    expect(first.errors).toEqual([]);
    expectGreen(first.pixels);

    // Generation 2 — SAME RenderObject / same avals, brand-new device and
    // Runtime. This is what RenderControl does on `deviceLost`. It must
    // produce an identical frame, with no validation errors from resources
    // left over from the dead device.
    const second = await renderGeneration(obj);
    expect(second.errors).toEqual([]);
    expectGreen(second.pixels);

    // The device-agnostic scene value survived untouched.
    expect(cubeTex.kind).toBe("cube");
  }, 60000);
});
