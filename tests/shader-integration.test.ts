// Real wombat.shader → wombat.rendering integration invariants:
//   - WGSL emit produces real `@vertex` / `@fragment` markers.
//   - Pipeline cache key correctly distinguishes
//     (effect, signature) pairs.
// Pixel-correctness of the actual hello-triangle renders lives in
// tests-browser/render-real.test.ts.

import { describe, expect, it } from "vitest";
import {
  AList,
  AdaptiveToken,
    HashMap,
  cval,
  type aval, AVal
} from "@aardworx/wombat.adaptive";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import {
  IBuffer,
  RenderTree,
  type BufferView,
  type Command,
  type DrawCall,
  type RenderObject,
  PipelineState,
} from "@aardworx/wombat.rendering/core";
import {
  allocateFramebuffer,
  createFramebufferSignature,
} from "@aardworx/wombat.rendering/resources";
import { Runtime } from "@aardworx/wombat.rendering/runtime";
import { makeEffect } from "./_makeEffect.js";
import { MockGPU } from "./_mockGpu.js";

const Tvec2f: Type = Vec(Tf32, 2);
const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

function helloTriangle() {
  return makeEffect(
    `
      function vsMain(input: { a_position: V2f; a_color: V3f }): { gl_Position: V4f; v_color: V3f } {
        return {
          gl_Position: new V4f(input.a_position.x, input.a_position.y, 0.0, 1.0),
          v_color: input.a_color,
        };
      }
      function fsMain(input: { v_color: V3f }): V4f {
        return new V4f(input.v_color.x, input.v_color.y, input.v_color.z, 1.0);
      }
    `,
    [
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
    ],
  );
}

function bv(format: GPUVertexFormat, bytes = 24): aval<BufferView> {
  return {
    buffer: AVal.constant(IBuffer.fromHost(new ArrayBuffer(bytes))),
    offset: 0, stride: format === "float32x2" ? 8 : 12, format,
  };
}

describe("shader integration: invariants", () => {
  it("emitted WGSL has @vertex / @fragment markers", () => {
    const eff = helloTriangle();
    const compiled = eff.compile({ target: "wgsl" });
    const vs = compiled.stages.find(s => s.stage === "vertex")!.source;
    const fs = compiled.stages.find(s => s.stage === "fragment")!.source;
    expect(vs).toMatch(/@vertex/);
    expect(fs).toMatch(/@fragment/);
  });

  it("(effect, signature) pipeline cache: same effect, two FBO shapes → two pipelines", () => {
    const gpu = new MockGPU();
    const eff = helloTriangle();
    const obj: RenderObject = {
      effect: eff,
      pipelineState: PipelineState.constant({ rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } }),
      vertexAttributes: HashMap.empty<string, BufferView>()
        .add("a_position", bv("float32x2"))
        .add("a_color",    bv("float32x3")),
      uniforms: HashMap.empty(),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };

    const sigA = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const sigB = createFramebufferSignature({ colors: { outColor: "rgba16float" } });
    const fboA = allocateFramebuffer(gpu.device, sigA, cval({ width: 4, height: 4 }));
    const fboB = allocateFramebuffer(gpu.device, sigB, cval({ width: 4, height: 4 }));
    fboA.acquire(); fboB.acquire();

    const runtime = new Runtime({ device: gpu.device });
    const cmds = AList.ofArray<Command>([
      { kind: "Render", output: fboA, tree: RenderTree.leaf(obj) },
      { kind: "Render", output: fboB, tree: RenderTree.leaf(obj) },
    ]);
    const task = runtime.compile(cmds);
    task.run(AdaptiveToken.top);
    expect(gpu.pipelines).toHaveLength(2);

    task.dispose();
    fboA.release(); fboB.release();
  });

  it("colorNames ordering drives the fragment-output @location pin", () => {
    // Two-target effect that writes both `outColor` and `pickId`. The
    // FB signature's `colorNames` order is the source of truth for
    // each output's `@location(i)`. Reverse the order, get reversed
    // locations.
    const eff = makeEffect(
      `
        function vs(input: { a_position: V2f }): { gl_Position: V4f } {
          return { gl_Position: new V4f(input.a_position.x, input.a_position.y, 0.0, 1.0) };
        }
        function fs(): { outColor: V4f; pickId: V4f } {
          return { outColor: new V4f(1.0, 0.0, 0.0, 1.0), pickId: new V4f(0.5, 0.0, 0.0, 1.0) };
        }
      `,
      [
        {
          name: "vs", stage: "vertex",
          inputs: [{ name: "a_position", type: Tvec2f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] }],
          outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] }],
        },
        {
          name: "fs", stage: "fragment",
          inputs: [],
          outputs: [
            // Effect's hand-pinned locations — irrelevant after linking.
            { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 99 }] },
            { name: "pickId",   type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 100 }] },
          ],
        },
      ],
    );

    const layoutFor = (sig: { colorNames: readonly string[] }) =>
      ({ locations: new Map(sig.colorNames.map((n, i) => [n, i])) });

    const sigAB = createFramebufferSignature({ colors: { outColor: "rgba8unorm", pickId: "rgba8unorm" } });
    const sigBA = createFramebufferSignature({ colors: { pickId: "rgba8unorm", outColor: "rgba8unorm" } });

    const compiledAB = eff.compile({ target: "wgsl", fragmentOutputLayout: layoutFor(sigAB) });
    const compiledBA = eff.compile({ target: "wgsl", fragmentOutputLayout: layoutFor(sigBA) });
    const fsAB = compiledAB.stages.find(s => s.stage === "fragment")!.source;
    const fsBA = compiledBA.stages.find(s => s.stage === "fragment")!.source;

    // Distinct sources reflecting different location pins.
    expect(fsAB).not.toBe(fsBA);
    // The hand-pinned 99 / 100 from the effect source are gone.
    expect(fsAB).not.toMatch(/@location\(99\)/);
    expect(fsAB).not.toMatch(/@location\(100\)/);
    // Both layouts assign distinct @location(0) and @location(1).
    expect(fsAB).toMatch(/@location\(0\)/);
    expect(fsAB).toMatch(/@location\(1\)/);
  });

  it("(effect, signature) pipeline cache: same effect + signature → one pipeline shared", () => {
    const gpu = new MockGPU();
    const eff = helloTriangle();
    const mk = (): RenderObject => ({
      effect: eff,
      pipelineState: PipelineState.constant({ rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } }),
      vertexAttributes: HashMap.empty<string, BufferView>()
        .add("a_position", bv("float32x2"))
        .add("a_color",    bv("float32x3")),
      uniforms: HashMap.empty(),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    });
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();
    const runtime = new Runtime({ device: gpu.device });
    const cmds = AList.ofArray<Command>([
      { kind: "Render", output: fbo, tree: RenderTree.ordered(RenderTree.leaf(mk()), RenderTree.leaf(mk())) },
    ]);
    const task = runtime.compile(cmds);
    task.run(AdaptiveToken.top);
    expect(gpu.pipelines).toHaveLength(1);    // both ROs share a pipeline
    expect(gpu.renderPasses[0]!.drawCalls).toHaveLength(2);
    task.dispose();
    fbo.release();
  });
});
