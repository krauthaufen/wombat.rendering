// allocateFramebuffer — verify per-attachment textures, depth/stencil
// allocation, size-driven reallocation, and signature naming.

import { describe, expect, it } from "vitest";
import { AdaptiveToken, cval, transact } from "@aardworx/wombat.adaptive";
import {
  allocateFramebuffer,
  createFramebufferSignature,
  type FramebufferSize,
} from "@aardworx/wombat.rendering/resources";
import { MockGPU } from "./_mockGpu.js";

describe("createFramebufferSignature", () => {
  it("derives hasDepth / hasStencil from depth-stencil format", () => {
    expect(createFramebufferSignature({ colors: { c: "rgba8unorm" }, depthStencil: { format: "depth24plus" } }).depthStencil)
      .toEqual({ format: "depth24plus", hasDepth: true, hasStencil: false });
    expect(createFramebufferSignature({ colors: {}, depthStencil: { format: "depth24plus-stencil8" } }).depthStencil)
      .toEqual({ format: "depth24plus-stencil8", hasDepth: true, hasStencil: true });
  });
});

describe("allocateFramebuffer", () => {
  it("creates one texture per color attachment + depth", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({
      colors: { albedo: "rgba8unorm", normal: "rgba16float" },
      depthStencil: { format: "depth24plus" },
    });
    const size = cval<FramebufferSize>({ width: 256, height: 128 });
    const fbo = allocateFramebuffer(gpu.device, sig, size, { labelPrefix: "gbuffer" });
    fbo.acquire();
    const ifb = fbo.getValue(AdaptiveToken.top);
    expect(ifb.width).toBe(256);
    expect(ifb.height).toBe(128);
    expect(gpu.textures).toHaveLength(3); // 2 color + 1 depth
    expect(ifb.colors.tryFind("albedo")).toBeDefined();
    expect(ifb.colors.tryFind("normal")).toBeDefined();
    expect(ifb.depthStencil).toBeDefined();
    const albedoTex = gpu.textures.find(t => t.descriptor.label === "gbuffer.albedo");
    expect(albedoTex?.descriptor.format).toBe("rgba8unorm");
    fbo.release();
    expect(gpu.textures.every(t => t.destroyed)).toBe(true);
  });

  it("size change reallocates all attachments", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({ colors: { c: "rgba8unorm" } });
    const size = cval<FramebufferSize>({ width: 8, height: 8 });
    const fbo = allocateFramebuffer(gpu.device, sig, size);
    fbo.acquire();
    fbo.getValue(AdaptiveToken.top);
    const firstId = gpu.textures[0]!.id;
    transact(() => { size.value = { width: 16, height: 16 }; });
    fbo.getValue(AdaptiveToken.top);
    expect(gpu.textures).toHaveLength(2);
    expect(gpu.textures[0]!.id).toBe(firstId);
    expect(gpu.textures[0]!.destroyed).toBe(true);
    fbo.release();
  });

  it("MSAA: allocates msaa color + single-sample resolve target, exposes resolve as colorTextures", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({
      colors: { c: "rgba8unorm" },
      depthStencil: { format: "depth24plus" },
      sampleCount: 4,
    });
    const size = cval<FramebufferSize>({ width: 8, height: 8 });
    const fbo = allocateFramebuffer(gpu.device, sig, size, { labelPrefix: "msaa" });
    fbo.acquire();
    const ifb = fbo.getValue(AdaptiveToken.top);
    // 1 msaa color + 1 resolve + 1 msaa depth = 3 textures.
    expect(gpu.textures).toHaveLength(3);
    const msaaColor = gpu.textures.find(t => t.descriptor.label === "msaa.c.msaa");
    const resolve = gpu.textures.find(t => t.descriptor.label === "msaa.c.resolve");
    expect(msaaColor?.descriptor.sampleCount).toBe(4);
    expect(resolve?.descriptor.sampleCount).toBe(1);
    // colorTextures (the sampleable side) points at the resolve target.
    expect(ifb.colorTextures?.tryFind("c")).toBe(resolve);
    // resolveColors is populated for MSAA framebuffers.
    expect(ifb.resolveColors?.tryFind("c")).toBeDefined();
    fbo.release();
  });

  it("same-size update does not reallocate", () => {
    const gpu = new MockGPU();
    const sig = createFramebufferSignature({ colors: { c: "rgba8unorm" } });
    const size = cval<FramebufferSize>({ width: 8, height: 8 });
    const fbo = allocateFramebuffer(gpu.device, sig, size);
    fbo.acquire();
    fbo.getValue(AdaptiveToken.top);
    transact(() => { size.value = { width: 8, height: 8 }; });
    fbo.getValue(AdaptiveToken.top);
    expect(gpu.textures).toHaveLength(1);
    fbo.release();
  });
});
