// Unordered children get reordered to group leaves with the same
// pipeline together, minimising state-change boundaries inside a
// render pass.

import { describe, expect, it } from "vitest";
import { AList, AdaptiveToken, AVal, HashMap, cval, type aval } from "@aardworx/wombat.adaptive";
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

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

function effectWithFrontFace(face: "ccw" | "cw") {
  void face;
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

function ro(face: "ccw" | "cw"): RenderObject {
  return {
    effect: effectWithFrontFace(face),
    // Different frontFace → different pipeline → different sort rank.
    pipelineState: PipelineState.constant({ rasterizer: { topology: "triangle-list", cullMode: "back", frontFace: face } }),
    vertexAttributes: AVal.constant(HashMap.empty<string, aval<BufferView>>().add("position", cval<BufferView>({
      buffer: IBuffer.fromHost(new ArrayBuffer(36)), offset: 0, count: 3, stride: 12, format: "float32x3",
    }))),
    uniforms: HashMap.empty(),
    textures: HashMap.empty(),
    samplers: HashMap.empty(),
    drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
  };
}

describe("Unordered: sort by pipeline state", () => {
  it("alternating CCW/CW children get grouped to reduce setPipeline calls", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device });
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();

    // Build [ccw, cw, ccw, cw] inside an Unordered subtree. After
    // sorting by pipeline rank we expect the two CCWs to be
    // adjacent and the two CWs to be adjacent — so setPipeline
    // bounces at most once between groups.
    const tree = RenderTree.unordered(
      RenderTree.leaf(ro("ccw")),
      RenderTree.leaf(ro("cw")),
      RenderTree.leaf(ro("ccw")),
      RenderTree.leaf(ro("cw")),
    );
    runtime.compile(AList.ofArray<Command>([{ kind: "Render", output: fbo, tree }]))
           .run(AdaptiveToken.top);

    expect(gpu.renderPasses).toHaveLength(1);
    const pipelinesInOrder = gpu.renderPasses[0]!.setPipelineCalls;
    expect(pipelinesInOrder).toHaveLength(4);
    // After sorting, the four pipelines must have at most one
    // pipeline switch in the middle: the sequence must be
    // [P, P, Q, Q] (or [Q, Q, P, P]).
    const a = pipelinesInOrder[0]!;
    const b = pipelinesInOrder[1]!;
    const c = pipelinesInOrder[2]!;
    const d = pipelinesInOrder[3]!;
    expect(a).toBe(b);
    expect(c).toBe(d);
    expect(a).not.toBe(c);
    fbo.release();
  });

  it("Ordered preserves child argument order even with mixed state", () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device });
    const sig = createFramebufferSignature({ colors: { outColor: "rgba8unorm" } });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();

    const ros = [ro("ccw"), ro("cw"), ro("ccw")];
    const tree = RenderTree.ordered(...ros.map(r => RenderTree.leaf(r)));
    runtime.compile(AList.ofArray<Command>([{ kind: "Render", output: fbo, tree }]))
           .run(AdaptiveToken.top);
    const pipes = gpu.renderPasses[0]!.setPipelineCalls;
    expect(pipes).toHaveLength(3);
    // Ordered preserves order: pipeline switches CCW → CW → CCW.
    expect(pipes[0]).not.toBe(pipes[1]);
    expect(pipes[1]).not.toBe(pipes[2]);
    expect(pipes[0]).toBe(pipes[2]);
    fbo.release();
  });
});
