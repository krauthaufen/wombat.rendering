// renderTo end-to-end. The pass-through shader is real wombat.shader;
// the textured outer shader uses a hand-built ProgramInterface
// (combined sampler+texture in source needs the Vite-plugin
// transform; not run in vitest).

import { describe, expect, it } from "vitest";
import {
  AList,
  AdaptiveToken,
  HashMap,
  cval,
  type aval,
} from "@aardworx/wombat.adaptive";
import { V4f } from "@aardworx/wombat.base";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader-ir";
import {
  IBuffer,
  ISampler,
  ITexture,
  RenderTree,
  type BufferView,
  type ClearValues,
  type Command,
  type CompiledEffect,
  type DrawCall,
  type RenderObject,
} from "@aardworx/wombat.rendering-core";
import {
  allocateFramebuffer,
  createFramebufferSignature,
} from "@aardworx/wombat.rendering-resources";
import { Runtime } from "@aardworx/wombat.rendering-runtime";
import { fakeEffectFromCompiled } from "./_fakeEffect.js";
import { makeEffect } from "./_makeEffect.js";
import { MockGPU } from "./_mockGpu.js";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);
const TtexFloat2D: Type = { kind: "Texture", target: "2D", sampled: { kind: "Float" }, arrayed: false, multisampled: false };
const TsampFloat:  Type = { kind: "Sampler", target: "2D", sampled: { kind: "Float" }, comparison: false };

function passEffect() {
  return makeEffect(
    `
      function vsMain(input: { position: V3f }): { gl_Position: V4f } {
        return { gl_Position: new V4f(input.position.x, input.position.y, input.position.z, 1.0) };
      }
      function fsMain(_input: {}): V4f {
        return new V4f(1.0, 0.0, 0.0, 1.0);
      }
    `,
    [
      {
        name: "vsMain", stage: "vertex",
        inputs: [{ name: "position", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] }],
      },
      {
        name: "fsMain", stage: "fragment",
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ],
  );
}

/** Hand-built CompiledEffect for the textured outer shader — see _fakeEffect.ts. */
function texturedEffectCompiled(): CompiledEffect {
  return {
    target: "wgsl",
    stages: [
      { stage: "vertex",   entryName: "main", source: "// vs", bindings: [] as never, meta: {} as never, sourceMap: null },
      { stage: "fragment", entryName: "main", source: "// fs", bindings: [] as never, meta: {} as never, sourceMap: null },
    ],
    interface: {
      target: "wgsl",
      stages: [],
      attributes: [{ name: "position", location: 0, type: Tvec3f, format: "float32x3", components: 3, byteSize: 12 }],
      fragmentOutputs: [{ name: "color", location: 0, type: Tvec4f }],
      uniforms: [], uniformBlocks: [],
      textures: [{ name: "src",  group: 0, slot: 0, type: TtexFloat2D }],
      samplers: [{ name: "samp", group: 0, slot: 1, type: TsampFloat }],
      storageBuffers: [],
    },
    avalBindings: {},
  };
}

function bv(): aval<BufferView> {
  return cval<BufferView>({
    buffer: IBuffer.fromHost(new ArrayBuffer(36)),
    offset: 0, count: 3, stride: 12, format: "float32x3",
  });
}

describe("renderTo", () => {
  it("compose: inner scene renders into FBO, outer scene samples its texture", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device });

    // Inner scene: a real shader, no textures, renders to the offscreen FBO.
    const innerEff = passEffect();
    const innerObj: RenderObject = {
      effect: innerEff,
      pipelineState: { rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } },
      vertexAttributes: HashMap.empty<string, aval<BufferView>>().add("position", bv()),
      uniforms: HashMap.empty(),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };
    const innerScene = RenderTree.leaf(innerObj);
    const offscreenSig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });

    const result = runtime.renderTo(innerScene, {
      size: cval({ width: 32, height: 32 }),
      signature: offscreenSig,
      clear: { colors: HashMap.empty<string, V4f>().add("outColor", new V4f(0, 0, 0, 1)) } as ClearValues,
      label: "offscreen",
    });
    const offscreenColor = result.color("outColor");

    // Outer scene: textured (synthetic CompiledEffect — combined samplers
    // need the Vite plugin which isn't running here).
    const outerEff = fakeEffectFromCompiled(texturedEffectCompiled(), "outer-textured");
    const outerObj: RenderObject = {
      effect: outerEff,
      pipelineState: { rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" } },
      vertexAttributes: HashMap.empty<string, aval<BufferView>>().add("position", bv()),
      uniforms: HashMap.empty(),
      textures: HashMap.empty<string, aval<ITexture>>().add("src", offscreenColor),
      samplers: HashMap.empty<string, aval<ISampler>>().add("samp", cval(ISampler.fromDescriptor({ magFilter: "linear" }))),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };

    const outerSig = createFramebufferSignature({ colors: { color: "rgba8unorm" } });
    const backbuffer = allocateFramebuffer(gpu.device, outerSig, cval({ width: 64, height: 64 }));
    backbuffer.acquire();
    const cmds = AList.ofArray<Command>([
      { kind: "Render", output: backbuffer, tree: RenderTree.leaf(outerObj) },
    ]);
    const task = runtime.compile(cmds);
    task.run(AdaptiveToken.top);

    // Inner clear + inner render + outer render: at least 3 passes;
    // exactly 2 draws total.
    expect(gpu.renderPasses.length).toBeGreaterThanOrEqual(3);
    const drawCounts = gpu.renderPasses.map(p => p.drawCalls.length);
    expect(drawCounts.reduce((a, b) => a + b, 0)).toBe(2);
    // Two pipelines: inner pass + outer textured.
    expect(gpu.pipelines).toHaveLength(2);

    task.dispose();
    backbuffer.release();
  });

  it("lifecycle: acquire on derived aval brings the FBO live; release tears it down", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device });
    const innerEff = passEffect();
    const innerObj: RenderObject = {
      effect: innerEff,
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
    expect(gpu.textures).toHaveLength(0);  // lazy, allocated on first compute
    colorAval.getValue(AdaptiveToken.top);
    expect(gpu.textures).toHaveLength(1);
    colorAval.release();
    expect(gpu.textures[0]!.destroyed).toBe(true);
  });
});
