// Real wombat.shader → wombat.rendering integration. Builds a real
// `Effect` from TypeScript source via `parseShader` + `stage(...)`,
// puts it on a RenderObject, and runs the whole rendering pipeline
// against the mock device. No hand-built ProgramInterface.

import { describe, expect, it } from "vitest";
import {
  AList,
  AdaptiveToken,
  HashMap,
  cval,
  type aval,
} from "@aardworx/wombat.adaptive";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader-ir";
import {
  IBuffer,
  RenderTree,
  type BufferView,
  type Command,
  type DrawCall,
  type RenderObject,
} from "@aardworx/wombat.rendering-core";
import {
  allocateFramebuffer,
  createFramebufferSignature,
} from "@aardworx/wombat.rendering-resources";
import { Runtime } from "@aardworx/wombat.rendering-runtime";
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
  return cval<BufferView>({
    buffer: IBuffer.fromHost(new ArrayBuffer(bytes)),
    offset: 0, count: 3, stride: format === "float32x2" ? 8 : 12, format,
  });
}

describe("shader integration: real wombat.shader → wombat.rendering", () => {
  it("Effect.compile() returns a CompiledEffect we can lower to a pipeline", () => {
    const gpu = new MockGPU();
    const eff = helloTriangle();
    const compiled = eff.compile({ target: "wgsl" });
    expect(compiled.target).toBe("wgsl");
    expect(compiled.interface.attributes.map(a => a.name).sort()).toEqual(["a_color", "a_position"]);
    expect(compiled.interface.fragmentOutputs.map(o => o.name)).toEqual(["outColor"]);

    const obj: RenderObject = {
      effect: eff,
      pipelineState: { rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } },
      vertexAttributes: HashMap.empty<string, aval<BufferView>>()
        .add("a_position", bv("float32x2"))
        .add("a_color",    bv("float32x3")),
      uniforms: HashMap.empty(),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };

    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();

    const runtime = new Runtime({ device: gpu.device });
    const cmds = AList.ofArray<Command>([
      { kind: "Render", output: fbo.getValue(AdaptiveToken.top), tree: RenderTree.leaf(obj) },
    ]);
    const task = runtime.compile(cmds);
    task.run(AdaptiveToken.top);

    expect(gpu.pipelines).toHaveLength(1);
    const pdesc = gpu.pipelines[0]!;
    const vbufs = pdesc.vertex.buffers as GPUVertexBufferLayout[];
    expect(vbufs).toHaveLength(2);
    expect((pdesc.fragment!.targets as GPUColorTargetState[])[0]!.format).toBe("rgba8unorm");
    expect(gpu.renderPasses[0]!.drawCalls).toHaveLength(1);
    expect(gpu.renderPasses[0]!.setVertexBufferCalls.map(c => c.slot).sort()).toEqual([0, 1]);

    task.dispose();
    fbo.release();
  });

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
      pipelineState: { rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } },
      vertexAttributes: HashMap.empty<string, aval<BufferView>>()
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
      { kind: "Render", output: fboA.getValue(AdaptiveToken.top), tree: RenderTree.leaf(obj) },
      { kind: "Render", output: fboB.getValue(AdaptiveToken.top), tree: RenderTree.leaf(obj) },
    ]);
    const task = runtime.compile(cmds);
    task.run(AdaptiveToken.top);
    expect(gpu.pipelines).toHaveLength(2);

    task.dispose();
    fboA.release(); fboB.release();
  });

  it("(effect, signature) pipeline cache: same effect + signature → one pipeline shared", () => {
    const gpu = new MockGPU();
    const eff = helloTriangle();
    const mk = (): RenderObject => ({
      effect: eff,
      pipelineState: { rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } },
      vertexAttributes: HashMap.empty<string, aval<BufferView>>()
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
      { kind: "Render", output: fbo.getValue(AdaptiveToken.top), tree: RenderTree.ordered(RenderTree.leaf(mk()), RenderTree.leaf(mk())) },
    ]);
    const task = runtime.compile(cmds);
    task.run(AdaptiveToken.top);
    expect(gpu.pipelines).toHaveLength(1);    // both ROs share a pipeline
    expect(gpu.renderPasses[0]!.drawCalls).toHaveLength(2);
    task.dispose();
    fbo.release();
  });
});
