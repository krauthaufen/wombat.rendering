// Reactive PipelineState: pipeline-influencing aval changes trigger
// a fresh GPURenderPipeline (cached per snapshot key); per-frame
// state avals (blendConstant, stencil.reference) DO NOT.

import { describe, expect, it } from "vitest";
import {
  AVal, AdaptiveToken, HashMap, cval, transact, type aval,
} from "@aardworx/wombat.adaptive";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import {
  IBuffer,
  PipelineState,
  type BlendState,
  type BufferView,
  type CullMode,
  type DrawCall,
  type RenderObject,
} from "@aardworx/wombat.rendering/core";
import {
  createFramebufferSignature,
  prepareRenderObject,
} from "@aardworx/wombat.rendering/resources";
import { makeEffect } from "./_makeEffect.js";
import { MockGPU } from "./_mockGpu.js";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

function trivEffect() {
  return makeEffect(
    `
      function vsMain(input: { position: V3f }): { gl_Position: V4f } {
        return { gl_Position: new V4f(input.position.x, input.position.y, input.position.z, 1.0) };
      }
      function fsMain(_input: {}): { outColor: V4f } {
        return { outColor: new V4f(1, 0, 0, 1) };
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

describe("PipelineState reactivity", () => {
  it("changing cullMode (pipeline-influencing) rebuilds the pipeline", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const eff = trivEffect();
    const compiled = eff.compile({ target: "wgsl" });

    const cullModeC = cval<CullMode>("none");
    const obj: RenderObject = {
      effect: eff,
      pipelineState: {
        rasterizer: {
          topology: AVal.constant<GPUPrimitiveTopology>("triangle-list"),
          cullMode: cullModeC,
          frontFace: AVal.constant("ccw"),
        },
      },
      vertexAttributes: HashMap.empty<string, aval<BufferView>>().add("position", bv()),
      uniforms: HashMap.empty(),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };

    const prepared = prepareRenderObject(gpu.device, obj, compiled, sig, { effectId: eff.id });
    prepared.update(AdaptiveToken.top);
    const before = gpu.pipelines.length;
    expect(before).toBeGreaterThan(0);
    const firstPipeline = prepared.pipeline;

    transact(() => { cullModeC.value = "back"; });
    prepared.update(AdaptiveToken.top);
    expect(gpu.pipelines.length).toBe(before + 1);
    expect(prepared.pipeline).not.toBe(firstPipeline);

    // Flip back — cache hit, no new pipeline.
    transact(() => { cullModeC.value = "none"; });
    prepared.update(AdaptiveToken.top);
    expect(gpu.pipelines.length).toBe(before + 1);
    expect(prepared.pipeline).toBe(firstPipeline);
  });

  it("changing blendConstant (per-frame) does NOT rebuild the pipeline", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const eff = trivEffect();
    const compiled = eff.compile({ target: "wgsl" });

    const blendConstantC = cval<{ r: number; g: number; b: number; a: number }>({ r: 0, g: 0, b: 0, a: 1 });
    const obj: RenderObject = {
      effect: eff,
      pipelineState: {
        rasterizer: {
          topology: AVal.constant<GPUPrimitiveTopology>("triangle-list"),
          cullMode: AVal.constant<CullMode>("none"),
          frontFace: AVal.constant("ccw"),
        },
        blendConstant: blendConstantC,
      },
      vertexAttributes: HashMap.empty<string, aval<BufferView>>().add("position", bv()),
      uniforms: HashMap.empty(),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };

    const prepared = prepareRenderObject(gpu.device, obj, compiled, sig, { effectId: eff.id });
    prepared.update(AdaptiveToken.top);
    const before = gpu.pipelines.length;
    expect(before).toBeGreaterThan(0);
    const firstPipeline = prepared.pipeline;

    transact(() => { blendConstantC.value = { r: 0.5, g: 0.5, b: 0.5, a: 1 }; });
    prepared.update(AdaptiveToken.top);
    expect(gpu.pipelines.length).toBe(before);
    expect(prepared.pipeline).toBe(firstPipeline);
  });

  it("changing stencil.reference (per-frame) does NOT rebuild the pipeline", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({
      colors: { outColor: "rgba8unorm" },
      depthStencil: { format: "depth24plus-stencil8", hasDepth: true, hasStencil: true },
    });
    const eff = trivEffect();
    const compiled = eff.compile({ target: "wgsl" });

    const referenceC = cval<number>(0);
    const obj: RenderObject = {
      effect: eff,
      pipelineState: {
        rasterizer: {
          topology: AVal.constant<GPUPrimitiveTopology>("triangle-list"),
          cullMode: AVal.constant<CullMode>("none"),
          frontFace: AVal.constant("ccw"),
        },
        depth: { write: AVal.constant(true), compare: AVal.constant<GPUCompareFunction>("less") },
        stencil: {
          enabled: AVal.constant(true),
          reference: referenceC,
          readMask: AVal.constant(0xff),
          writeMask: AVal.constant(0xff),
          front: {
            compare: AVal.constant<GPUCompareFunction>("always"),
            failOp: AVal.constant<GPUStencilOperation>("keep"),
            depthFailOp: AVal.constant<GPUStencilOperation>("keep"),
            passOp: AVal.constant<GPUStencilOperation>("replace"),
          },
          back: {
            compare: AVal.constant<GPUCompareFunction>("always"),
            failOp: AVal.constant<GPUStencilOperation>("keep"),
            depthFailOp: AVal.constant<GPUStencilOperation>("keep"),
            passOp: AVal.constant<GPUStencilOperation>("replace"),
          },
        },
      },
      vertexAttributes: HashMap.empty<string, aval<BufferView>>().add("position", bv()),
      uniforms: HashMap.empty(),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };

    const prepared = prepareRenderObject(gpu.device, obj, compiled, sig, { effectId: eff.id });
    prepared.update(AdaptiveToken.top);
    const before = gpu.pipelines.length;
    const firstPipeline = prepared.pipeline;

    transact(() => { referenceC.value = 7; });
    prepared.update(AdaptiveToken.top);
    expect(gpu.pipelines.length).toBe(before);
    expect(prepared.pipeline).toBe(firstPipeline);
  });

  it("PipelineState.constant() round-trips a plain pipeline-state into avals", () => {
    const ps = PipelineState.constant({
      rasterizer: { topology: "triangle-list", cullMode: "back", frontFace: "ccw" },
      depth: { write: true, compare: "less" },
    });
    expect(ps.rasterizer.cullMode.getValue(AdaptiveToken.top)).toBe("back");
    expect(ps.rasterizer.topology.getValue(AdaptiveToken.top)).toBe("triangle-list");
    expect(ps.depth!.write.getValue(AdaptiveToken.top)).toBe(true);
    expect(ps.depth!.compare.getValue(AdaptiveToken.top)).toBe("less");
  });

  // Reference unused imports to keep tsc happy when these become helpers.
  void HashMap; void undefined as undefined | BlendState;
});
