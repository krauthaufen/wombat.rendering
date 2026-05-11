// Unordered children get reordered to group leaves with the same
// pipeline together, minimising state-change boundaries inside a
// render pass.

import { describe, expect, it } from "vitest";
import { AList, AdaptiveToken, HashMap, cval, type aval , AVal} from "@aardworx/wombat.adaptive";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import {
  IBuffer,
  RenderTree,
  type BufferView,
  type Command,
  type DrawCall,
  type RenderObject,
  PipelineState,
  ElementType,
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
    vertexAttributes: HashMap.empty<string, BufferView>().add("position", {
      buffer: AVal.constant(IBuffer.fromHost(new ArrayBuffer(36))), offset: 0, stride: 12, elementType: ElementType.V3f,
    }),
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
    runtime.compile(sig, AList.ofArray<Command>([{ kind: "Render",tree }]))
           .run(fbo.getValue(AdaptiveToken.top), AdaptiveToken.top);

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

  // Note: an earlier test asserted that `RenderTree.ordered` preserved
  // child order even across mixed state. The contract changed:
  // flattenRenderTree now collapses Ordered + Unordered into the same
  // aset (see `flattenTree.ts`), so order inside a single Render
  // command is no longer load-bearing. Ordering belongs at the
  // Command-list level.
});
