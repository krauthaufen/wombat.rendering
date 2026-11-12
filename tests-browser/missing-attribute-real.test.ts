// A draw whose linked shader reads an attribute it does not have is left
// out of the scene and reported once (runtime/missingAttributes.ts); the
// rest of the scene draws. An attribute the vertex shader only passes on
// to an unread varying is dropped by linking and need not be there. Both
// paths: heap and legacy.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import {
  IBuffer, ElementType, PipelineState, RenderTree,
  type BufferView, type DrawCall, type RenderObject,
} from "@aardworx/wombat.rendering/core";
import { AttributeProvider, UniformProvider } from "../packages/rendering/src/core/provider.js";
import { compileHybridScene } from "../packages/rendering/src/runtime/hybridScene.js";
import { deviceErrors } from "../packages/rendering/src/runtime/deviceErrors.js";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import { readTexturePixels, requestRealDevice } from "./_realGpu.js";
import { makeHeapTestEffect } from "../tests/_heapTestEffect.js";

const W = 30, H = 8;
const effect = makeHeapTestEffect(); // reads Positions; Normals only feeds an unread varying

/** A quad over third `k` of the target, with the given attributes. */
function ro(k: number, attrs: ("Positions" | "Normals")[], debugName: string): RenderObject {
  const x0 = -1 + (2 * k) / 3, x1 = -1 + (2 * (k + 1)) / 3;
  const pos = new Float32Array([x0, -1, 0, x1, -1, 0, x0, 1, 0, x0, 1, 0, x1, -1, 0, x1, 1, 0]);
  const view = (a: Float32Array): BufferView =>
    ({ buffer: AVal.constant(IBuffer.fromHost(a.buffer)), offset: 0, stride: 12, elementType: ElementType.V3f });
  const bag: Record<string, BufferView> = {};
  if (attrs.includes("Positions")) bag.Positions = view(pos);
  if (attrs.includes("Normals")) bag.Normals = view(new Float32Array(18));
  return {
    effect,
    debugName,
    pipelineState: PipelineState.constant({
      rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" },
      depth: { write: true, compare: "less-equal" },
    }),
    vertexAttributes: AttributeProvider.ofObject(bag),
    uniforms: UniformProvider.ofObject({
      ModelTrafo: AVal.constant(Trafo3d.identity),
      ViewProjTrafo: AVal.constant(Trafo3d.identity),
      Color: AVal.constant(new V4f(1, 0, 0, 1)),
      LightLocation: AVal.constant(new V3d(0, 0, 1)),
    }),
    textures: HashMap.empty(),
    samplers: HashMap.empty(),
    drawCall: AVal.constant<DrawCall>({ kind: "non-indexed", vertexCount: 6, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
  } as unknown as RenderObject;
}

describe("a draw missing an attribute its shader reads — real GPU", () => {
  for (const heap of [true, false]) {
    it(`${heap ? "heap" : "legacy"}: left out and reported once; the rest draws`, async () => {
      const device = await requestRealDevice();
      try {
        const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
        const tree = RenderTree.ordered(
          RenderTree.leaf(ro(0, ["Positions", "Normals"], "full quad")),
          RenderTree.leaf(ro(1, ["Positions"], "pass-through quad")),   // Normals dropped by linking
          RenderTree.leaf(ro(2, ["Normals"], "bad quad")),              // Positions is read
        );
        const colorTex = device.createTexture({ size: { width: W, height: H }, format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
        const depthTex = device.createTexture({ size: { width: W, height: H }, format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
        // two scenes on one device (as a color and a pick pass): one report
        for (let pass = 0; pass < 2; pass++) {
          const scene = compileHybridScene(device, sig, tree, { heapEnabled: AVal.constant(heap) });
          for (let frame = 0; frame < 2; frame++) {
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
          }
          if (heap) expect(scene.heapTotalDraws()).toBe(2);
        }
        const px = await readTexturePixels(device, colorTex);
        const red = (k: number): boolean => px[(4 * W + Math.floor((k + 0.5) * W / 3)) * 4]! > 100;
        expect([red(0), red(1), red(2)]).toEqual([true, true, false]);
        const errs = deviceErrors(device).filter((e) => e.kind === "scene");
        expect(errs.length).toBe(1);
        expect(errs[0]!.recovered).toBe(true);
        expect(errs[0]!.message).toBe(
          `bad quad: its shader reads attribute "Positions", which it does not have (it has Normals); not drawn`);
        expect(deviceErrors(device).filter((e) => e.kind !== "scene")).toEqual([]);
      } finally { device.destroy(); }
    });
  }
});
