// Inactive draws do not pull (heap): 10k draws, each with its own colour
// uniform computed from one shared cell. Animated while the draws are
// hidden, a frame reads none of them; shown, every frame repacks all
// 10k colours. Prints the per-frame cost of both, and of the first show after
// hidden animation (each colour read once, then).

import { describe, expect, it } from "vitest";
import { AList, AdaptiveToken, AVal, HashMap, cval, transact } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3f, V4f } from "@aardworx/wombat.base";
import { makeHeapTestEffect } from "../tests/_heapTestEffect.js";
import { ElementType, IBuffer, RenderTree, PipelineState, type BufferView, type Command, type DrawCall, type RenderObject } from "@aardworx/wombat.rendering/core";
import { allocateFramebuffer, createFramebufferSignature, TextureUsage } from "@aardworx/wombat.rendering/resources";
import { Runtime } from "@aardworx/wombat.rendering/runtime";
import { requestRealDevice } from "./_realGpu.js";

const N = 10_000;
describe("heap: inactive draws do not pull (real GPU)", () => {
  it("animating a shared cell under 10k hidden draws costs nothing per frame", async () => {
    const device = await requestRealDevice();
    const errors: string[] = [];
    device.onuncapturederror = (e) => errors.push(e.error.message);
    try {
      const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
      const fbo = allocateFramebuffer(device, sig, cval({ width: 64, height: 64 }), { extraUsage: TextureUsage.COPY_SRC });
      fbo.acquire();
      const eff = makeHeapTestEffect();
      const ps = PipelineState.constant({ rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } });
      const t = cval(0);
      const shown = cval(false);
      let reads = 0;
      const host = (a: number[]): BufferView => ({ buffer: AVal.constant(IBuffer.fromHost(new Float32Array(a))), offset: 0, stride: 0, elementType: ElementType.F32 });
      const positions: BufferView = { ...host([0, 0, 0, 1, 0.02, 0, 0, 1, 0, 0.02, 0, 1]), stride: 16, elementType: ElementType.V4f };
      const normals: BufferView = { ...host([0, 0, 1, 0, 0, 1, 0, 0, 1]), stride: 12, elementType: ElementType.V3f };
      const objs: RenderObject[] = Array.from({ length: N }, (_, i) => {
        const color = t.map((v) => { reads++; return new V4f((v + i) % 7 / 7, 0.5, 0.5, 1); });
        return {
          effect: eff, pipelineState: ps, active: shown,
          vertexAttributes: HashMap.empty<string, BufferView>().add("Positions", positions).add("Normals", normals),
          uniforms: HashMap.empty<string, unknown>()
            .add("ModelTrafo", AVal.constant(Trafo3d.identity))
            .add("ViewProjTrafo", AVal.constant(Trafo3d.identity))
            .add("LightLocation", AVal.constant(new V3f(0, 0, 1)))
            .add("Color", color) as never,
          textures: HashMap.empty(), samplers: HashMap.empty(),
          drawCall: AVal.constant<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
        };
      });
      const runtime = new Runtime({ device });
      const task = runtime.compile(sig, AList.ofArray<Command>([
        { kind: "Clear", values: { colors: HashMap.empty<string, V4f>().add("outColor", new V4f(0, 0, 0, 0)) } },
        { kind: "Render", tree: RenderTree.unordered(...objs.map((o) => RenderTree.leaf(o))) },
      ]));
      const frame = (): number => {
        const s = performance.now();
        task.run(fbo.getValue(AdaptiveToken.top), AdaptiveToken.top);
        return performance.now() - s;
      };
      const animate = (frames: number): number => {
        let total = 0;
        for (let k = 0; k < frames; k++) {
          const s = performance.now();
          transact(() => { t.value = t.value + 1; });
          total += performance.now() - s + frame();
        }
        return total / frames;
      };
      const report: Record<string, number> = {};
      report["load hidden (ms)"] = frame();
      expect(reads).toBe(N);
      const r0 = reads;
      report["animate hidden, per frame (ms)"] = animate(20);
      expect(reads).toBe(r0);
      const s = performance.now();
      transact(() => { shown.value = true; });
      report["first show after hidden animation (ms)"] = performance.now() - s + frame();
      expect(reads).toBe(r0 + N);
      report["animate shown, per frame (ms)"] = animate(20);
      expect(reads).toBe(r0 + 21 * N);
      transact(() => { shown.value = false; });
      frame();
      report["animate hidden again, per frame (ms)"] = animate(20);
      expect(reads).toBe(r0 + 21 * N);
      // eslint-disable-next-line no-console
      console.log("[inactive-pull] " + JSON.stringify(Object.fromEntries(Object.entries(report).map(([k, v]) => [k, Math.round(v * 10) / 10]))));
      await device.queue.onSubmittedWorkDone();
      fbo.release();
      task.dispose();
      expect(errors).toEqual([]);
    } finally {
      device.destroy();
    }
  }, 120000);
});
