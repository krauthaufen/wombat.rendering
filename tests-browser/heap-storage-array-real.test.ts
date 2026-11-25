// Real-GPU test for HEAP-MANAGED STORAGE BUFFERS.
//
// An effect's read-only `var<storage>` array, supplied as host data, lives
// in the heap arena: the draw header holds a ref to the allocation and the
// shader's `Name[i]` reads become arena loads — in the vertex AND the
// fragment stage, with WGSL's element layout (a vec3 array has a 16-byte
// stride).
//
// Two draws share ONE effect (so one bucket, one megacall) but supply
// DIFFERENT buffers:
//   - LEFT:  quad authored at x in [-2, -1], moved right by Offsets[1] =
//            (1, 0, 0); colour = Palette[1] = red.
//   - RIGHT: quad at x in [0, 1], Offsets[1] = 0; colour = Palette[2] = blue.
// A wrong stride reads Offsets[0] (garbage on purpose) and the quads land
// off screen; reading the other draw's buffer swaps the colours.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V4f } from "@aardworx/wombat.base";
import {
  IBuffer, ElementType, PipelineState, RenderTree,
  type BufferView, type DrawCall, type RenderObject,
} from "@aardworx/wombat.rendering/core";
import { AttributeProvider, UniformProvider } from "../packages/rendering/src/core/provider.js";
import { compileHybridScene } from "../packages/rendering/src/runtime/hybridScene.js";
import { buildHeapScene, type HeapDrawSpec } from "../packages/rendering/src/runtime/heapScene.js";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";
import { makeHeapTestEffectStorage } from "../tests/_heapTestEffect.js";
import { buildShaderFamily } from "../packages/rendering/src/runtime/heapShaderFamily.js";
import { compileHeapEffectIR } from "../packages/rendering/src/runtime/heapEffectIR.js";

const trafoIdentity = AVal.constant(Trafo3d.identity) as unknown;
const v4 = (x: number, y: number, z: number, w: number) => new V4f(x, y, z, w) as unknown;
const host = (data: number[]): IBuffer => IBuffer.fromHost(new Float32Array(data));

describe("heap-managed storage buffers (real GPU)", () => {
  it("reads per-draw storage arrays in both stages, and follows changes", async () => {
    // The heap persists compiled shaders in localStorage; a test must not
    // be served the output of an older build of the rewrite.
    try { globalThis.localStorage?.clear(); } catch { /* no storage */ }
    const device = await requestRealDevice();
    const errors: GPUError[] = [];
    device.onuncapturederror = (e) => errors.push(e.error);
    try {
      const sig = createFramebufferSignature({
        colors: { outColor: "rgba8unorm" },
        depthStencil: { format: "depth24plus" },
      });
      const quad = (xMin: number, xMax: number) => new Float32Array([
        xMin, -1, 0,  xMax, -1, 0,  xMin, 1, 0,
        xMin,  1, 0,  xMax, -1, 0,  xMax, 1, 0,
      ]);
      const idx = new Uint32Array([0, 1, 2, 3, 4, 5]);
      const effect = makeHeapTestEffectStorage();

      // vec3 elements, 16 bytes each: [garbage, the offset that is read]
      const offsetsLeft = AVal.constant(host([9, 9, 9, 9,  1, 0, 0, 0]));
      const offsetsRight = AVal.constant(host([9, 9, 9, 9,  0, 0, 0, 0]));
      const paletteLeft = cval(host([0, 1, 0, 1,  1, 0, 0, 1]));                   // [green, RED]
      const paletteRight = AVal.constant(host([0, 1, 0, 1,  0, 1, 0, 1,  0, 0, 1, 1])); // [.., .., BLUE]

      const draws: HeapDrawSpec[] = [
        {
          effect,
          inputs: { Positions: quad(-2, -1), ModelTrafo: trafoIdentity, ViewProjTrafo: trafoIdentity, Color: v4(1, 0, 0, 0) },
          indices: idx,
          storageArrays: new Map([["Offsets", offsetsLeft], ["Palette", paletteLeft]]),
        },
        {
          effect,
          inputs: { Positions: quad(0, 1), ModelTrafo: trafoIdentity, ViewProjTrafo: trafoIdentity, Color: v4(2, 0, 0, 0) },
          indices: idx,
          storageArrays: new Map([["Offsets", offsetsRight], ["Palette", paletteRight]]),
        },
      ];
      const scene = buildHeapScene(device, sig, draws, {
        fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
      });

      const W = 32, H = 16;
      const colorTex = device.createTexture({
        size: { width: W, height: H, depthOrArrayLayers: 1 }, format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      const depthTex = device.createTexture({
        size: { width: W, height: H, depthOrArrayLayers: 1 }, format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      const render = async (): Promise<(x: number, y: number) => number[]> => {
        scene.update(AdaptiveToken.top);
        const enc = device.createCommandEncoder();
        scene.encodeComputePrep(enc, AdaptiveToken.top);
        const pass = enc.beginRenderPass({
          colorAttachments: [{ view: colorTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
          depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" },
        });
        scene.encodeIntoPass(pass);
        pass.end();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        const pixels = await readTexturePixels(device, colorTex);
        if (errors.length > 0) {
          const fol = { locations: new Map([["outColor", 0]]) };
          const fam = buildShaderFamily([effect], fol, undefined, {
            perEffectPerInstance: new Map([[effect, { attributes: new Set<string>(), storage: new Set(["Offsets", "Palette"]) }]]),
          });
          const ir = compileHeapEffectIR(effect, fam.drawHeaderUnion, { target: "wgsl", fragmentOutputLayout: fol }, "standalone");
          throw new Error("GPU errors:\n  " + errors.map(e => (e as { message?: string }).message ?? String(e)).join("\n  ")
            + "\n--- VS ---\n" + ir.vs + "\n--- FS ---\n" + ir.fs);
        }
        return (x, y) => { const o = (y * W + x) * 4; return [pixels[o]!, pixels[o + 1]!, pixels[o + 2]!]; };
      };

      let at = await render();
      expect(at(8, 8), "left = Palette[1] of the left draw, moved by Offsets[1]").toEqual([255, 0, 0]);
      expect(at(24, 8), "right = Palette[2] of the right draw").toEqual([0, 0, 255]);

      // same size, new contents: rewritten in place
      transact(() => { paletteLeft.value = host([0, 1, 0, 1,  1, 1, 0, 1]); });
      at = await render();
      expect(at(8, 8), "left follows its buffer").toEqual([255, 255, 0]);
      expect(at(24, 8), "right untouched").toEqual([0, 0, 255]);

      // a larger buffer: the allocation moves, the draw header follows
      transact(() => { paletteLeft.value = host([0, 1, 0, 1,  0, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1]); });
      at = await render();
      expect(at(8, 8), "left after a resize").toEqual([0, 255, 255]);
      expect(at(24, 8), "right untouched").toEqual([0, 0, 255]);
    } finally {
      device.destroy();
    }
  });
});

// The same picture, reached the way an application reaches it: plain
// RenderObjects with `storageBuffers` in a RenderTree, compiled by the
// hybrid scene. With the heap enabled both draws must ride the heap (a
// storage buffer no longer sends a draw to the classic path); with it
// disabled the classic path binds the same buffers as real storage
// bindings. Both must produce identical pixels, before and after a change.
describe("render objects with storage buffers through the scene (real GPU)", () => {
  const W = 32, H = 16;
  const effect = makeHeapTestEffectStorage();
  const quadRO = (
    xMin: number, xMax: number, paletteIndex: number,
    offsets: aval<IBuffer>, palette: aval<IBuffer>, debugName: string,
  ): RenderObject => {
    const pos = new Float32Array([xMin, -1, 0, xMax, -1, 0, xMin, 1, 0, xMin, 1, 0, xMax, -1, 0, xMax, 1, 0]);
    const view: BufferView = { buffer: AVal.constant(IBuffer.fromHost(pos.buffer)), offset: 0, stride: 12, elementType: ElementType.V3f };
    return {
      effect,
      debugName,
      pipelineState: PipelineState.constant({
        rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" },
        depth: { write: true, compare: "less-equal" },
      }),
      vertexAttributes: AttributeProvider.ofObject({ Positions: view }),
      uniforms: UniformProvider.ofObject({
        ModelTrafo: AVal.constant(Trafo3d.identity),
        ViewProjTrafo: AVal.constant(Trafo3d.identity),
        Color: AVal.constant(new V4f(paletteIndex, 0, 0, 0)),
      }),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      storageBuffers: HashMap.empty<string, aval<IBuffer>>().add("Offsets", offsets).add("Palette", palette),
      drawCall: AVal.constant<DrawCall>({ kind: "non-indexed", vertexCount: 6, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    } as unknown as RenderObject;
  };

  for (const heap of [true, false]) {
    it(`${heap ? "heap" : "classic"}: per-draw storage buffers are read in both stages and follow changes`, async () => {
      try { globalThis.localStorage?.clear(); } catch { /* no storage */ }
      const device = await requestRealDevice();
      const errors: GPUError[] = [];
      device.onuncapturederror = (e) => errors.push(e.error);
      try {
        const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
        const paletteLeft = cval(host([0, 1, 0, 1,  1, 0, 0, 1]));
        const tree = RenderTree.ordered(
          RenderTree.leaf(quadRO(-2, -1, 1, AVal.constant(host([9, 9, 9, 9,  1, 0, 0, 0])), paletteLeft, "left quad")),
          RenderTree.leaf(quadRO(0, 1, 2, AVal.constant(host([9, 9, 9, 9,  0, 0, 0, 0])),
            AVal.constant(host([0, 1, 0, 1,  0, 1, 0, 1,  0, 0, 1, 1])), "right quad")),
        );
        const colorTex = device.createTexture({ size: { width: W, height: H }, format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
        const depthTex = device.createTexture({ size: { width: W, height: H }, format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
        const scene = compileHybridScene(device, sig, tree, { heapEnabled: AVal.constant(heap) });
        const render = async (): Promise<(x: number, y: number) => number[]> => {
          scene.update(AdaptiveToken.top);
          const enc = device.createCommandEncoder();
          scene.encodeComputePrep(enc, AdaptiveToken.top);
          const rp = enc.beginRenderPass({
            colorAttachments: [{ view: colorTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
            depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
          });
          scene.encodeIntoPass(rp, AdaptiveToken.top);
          rp.end();
          device.queue.submit([enc.finish()]);
          await device.queue.onSubmittedWorkDone();
          const px = await readTexturePixels(device, colorTex);
          if (errors.length > 0) {
            throw new Error("GPU errors:\n  " + errors.map(e => (e as { message?: string }).message ?? String(e)).join("\n  "));
          }
          return (x, y) => { const o = (y * W + x) * 4; return [px[o]!, px[o + 1]!, px[o + 2]!]; };
        };

        let at = await render();
        expect(scene.heapTotalDraws(), "draws on the heap").toBe(heap ? 2 : 0);
        expect(at(8, 8), "left").toEqual([255, 0, 0]);
        expect(at(24, 8), "right").toEqual([0, 0, 255]);

        transact(() => { paletteLeft.value = host([0, 1, 0, 1,  1, 1, 0, 1]); });
        at = await render();
        expect(at(8, 8), "left follows its buffer").toEqual([255, 255, 0]);
        expect(at(24, 8), "right untouched").toEqual([0, 0, 255]);

        transact(() => { paletteLeft.value = host([0, 1, 0, 1,  0, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1]); });
        at = await render();
        expect(at(8, 8), "left after a resize").toEqual([0, 255, 255]);
        expect(at(24, 8), "right untouched").toEqual([0, 0, 255]);
        expect(scene.heapTotalDraws(), "still on the heap").toBe(heap ? 2 : 0);
      } finally { device.destroy(); }
    });
  }
});
