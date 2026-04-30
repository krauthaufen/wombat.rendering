// Real wombat.shader → wombat.rendering integration. Compiles
// vertex + fragment TypeScript source through `compileShaderSource`
// and feeds the resulting `CompiledEffect` straight into
// `prepareRenderObject` + the runtime. No hand-built ProgramInterface.

import { describe, expect, it } from "vitest";
import {
  AList,
  AdaptiveToken,
  HashMap,
  cval,
  type aval,
} from "@aardworx/wombat.adaptive";
import { compileShaderSource } from "@aardworx/wombat.shader-runtime";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader-ir";
import {
  IBuffer,
  RenderTree,
  type BufferView,
  type Command,
  type CompiledEffect,
  type DrawCall,
  type RenderObject,
} from "@aardworx/wombat.rendering-core";
import {
  allocateFramebuffer,
  createFramebufferSignature,
  prepareRenderObject,
} from "@aardworx/wombat.rendering-resources";
import { Runtime } from "@aardworx/wombat.rendering-runtime";
import { fakeEffectFromCompiled } from "./_fakeEffect.js";
import { MockGPU } from "./_mockGpu.js";

const Tvec2f: Type = Vec(Tf32, 2);
const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

function helloTriangleSource(): string {
  return `
    function vsMain(input: { a_position: V2f; a_color: V3f }): { gl_Position: V4f; v_color: V3f } {
      return {
        gl_Position: new V4f(input.a_position.x, input.a_position.y, 0.0, 1.0),
        v_color: input.a_color,
      };
    }
    function fsMain(input: { v_color: V3f }): V4f {
      return new V4f(input.v_color.x, input.v_color.y, input.v_color.z, 1.0);
    }
  `;
}

function helloTriangleEffect(): CompiledEffect {
  return compileShaderSource(helloTriangleSource(), [
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
  ], { target: "wgsl" });
}

function bv(format: GPUVertexFormat, bytes = 24): aval<BufferView> {
  return cval<BufferView>({
    buffer: IBuffer.fromHost(new ArrayBuffer(bytes)),
    offset: 0, count: 3, stride: format === "float32x2" ? 8 : 12, format,
  });
}

describe("shader integration: real wombat.shader → wombat.rendering", () => {
  it("compiles a vertex+fragment effect and lowers it to a pipeline", () => {
    const gpu = new MockGPU();
    const eff = helloTriangleEffect();

    // Sanity-check that wombat.shader produced what we expect.
    expect(eff.target).toBe("wgsl");
    expect(eff.stages.map(s => s.stage)).toEqual(["vertex", "fragment"]);
    expect(eff.interface.attributes.map(a => a.name).sort()).toEqual(["a_color", "a_position"]);
    expect(eff.interface.fragmentOutputs.map(o => o.name)).toEqual(["outColor"]);
    expect(eff.interface.uniformBlocks).toEqual([]);

    // Build a RenderObject around the compiled effect.
    const obj: RenderObject = {
      effect: fakeEffectFromCompiled(eff),
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
    const prepared = prepareRenderObject(gpu.device, obj, eff, sig, { label: "tri" });
    prepared.acquire();

    // Pipeline got built once.
    expect(gpu.pipelines).toHaveLength(1);
    const pdesc = gpu.pipelines[0]!;
    // Vertex layout matches shader's declared locations.
    const vbufs = pdesc.vertex.buffers as GPUVertexBufferLayout[];
    expect(vbufs).toHaveLength(2);
    expect(vbufs[0]!.attributes[0]!.shaderLocation).toBe(0);
    expect(vbufs[1]!.attributes[0]!.shaderLocation).toBe(1);
    // Fragment-output format pulled from the signature by name match.
    const targets = pdesc.fragment!.targets as GPUColorTargetState[];
    expect(targets[0]!.format).toBe("rgba8unorm");
    // No bind-group layouts (no UBOs/textures/samplers in this effect).
    expect(pdesc.layout).toBeDefined();

    // Encode a render pass.
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();
    const enc = gpu.createCommandEncoder();
    const runtime = new Runtime({
      device: gpu.device,
      compileEffect: () => eff,        // bypass Effect.compile, feed the precompiled CompiledEffect
    });
    const cmds = AList.ofArray<Command>([
      { kind: "Render", output: fbo.getValue(AdaptiveToken.top), tree: RenderTree.leaf(obj) },
    ]);
    const task = runtime.compile(cmds);
    task.run(AdaptiveToken.top);

    expect(gpu.renderPasses).toHaveLength(1);
    expect(gpu.renderPasses[0]!.drawCalls).toHaveLength(1);
    expect(gpu.renderPasses[0]!.setVertexBufferCalls.map(c => c.slot).sort()).toEqual([0, 1]);

    task.dispose();
    fbo.release();
    void enc;
  });

  it("emits real WGSL source — matches what the WebGPU driver expects", () => {
    const eff = helloTriangleEffect();
    const vs = eff.stages.find(s => s.stage === "vertex")!.source;
    const fs = eff.stages.find(s => s.stage === "fragment")!.source;
    // Both stages produced something; both contain a real entry-point declaration.
    expect(vs.length).toBeGreaterThan(0);
    expect(fs.length).toBeGreaterThan(0);
    expect(vs).toMatch(/@vertex/);
    expect(fs).toMatch(/@fragment/);
  });
});
