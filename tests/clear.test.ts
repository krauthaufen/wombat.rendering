// clear(cmd, output, values) — render-pass loadOp wiring per
// named attachment.

import { describe, expect, it } from "vitest";
import { AdaptiveToken, HashMap, cval } from "@aardworx/wombat.adaptive";
import { V4f } from "@aardworx/wombat.base";
import type { ClearValues } from "@aardworx/wombat.rendering-core";
import { allocateFramebuffer, createFramebufferSignature } from "@aardworx/wombat.rendering-resources";
import { clear } from "@aardworx/wombat.rendering-commands";
import { MockGPU } from "./_mockGpu.js";

describe("clear", () => {
  it("clears named color attachment, loads the rest, clears depth", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({
      colors: { albedo: "rgba8unorm", normal: "rgba16float" },
      depthStencil: { format: "depth24plus" },
    });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 4, height: 4 }));
    fbo.acquire();
    const ifb = fbo.getValue(AdaptiveToken.top);

    const values: ClearValues = {
      colors: HashMap.empty<string, V4f>().add("albedo", new V4f(0.25, 0.5, 0.75, 1)),
      depth: 1,
    };
    const enc = gpu.createCommandEncoder();
    clear(enc, ifb, values);

    expect(gpu.renderPasses).toHaveLength(1);
    const desc = gpu.renderPasses[0]!.desc;
    const colors = desc.colorAttachments as GPURenderPassColorAttachment[];
    const albedo = colors.find(c => (c as GPURenderPassColorAttachment).view !== undefined && (c as GPURenderPassColorAttachment).loadOp === "clear")!;
    expect(albedo.clearValue).toEqual({ r: 0.25, g: 0.5, b: 0.75, a: 1 });
    expect(albedo.storeOp).toBe("store");
    const normal = colors.find(c => (c as GPURenderPassColorAttachment).loadOp === "load")!;
    expect(normal.loadOp).toBe("load");

    expect(desc.depthStencilAttachment?.depthLoadOp).toBe("clear");
    expect(desc.depthStencilAttachment?.depthClearValue).toBe(1);

    fbo.release();
  });

  it("loads depth when not in values", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({
      colors: { c: "rgba8unorm" },
      depthStencil: { format: "depth24plus" },
    });
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 1, height: 1 }));
    fbo.acquire();
    const enc = gpu.createCommandEncoder();
    clear(enc, fbo.getValue(AdaptiveToken.top), {});
    const desc = gpu.renderPasses[0]!.desc;
    expect((desc.colorAttachments as GPURenderPassColorAttachment[])[0]!.loadOp).toBe("load");
    expect(desc.depthStencilAttachment?.depthLoadOp).toBe("load");
    fbo.release();
  });
});
