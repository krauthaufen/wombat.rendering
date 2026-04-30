// prepareAdaptiveSampler — sampler dedup via descriptor cache.

import { describe, expect, it } from "vitest";
import { AdaptiveToken, cval, transact } from "@aardworx/wombat.adaptive";
import { ISampler } from "@aardworx/wombat.rendering-core";
import { prepareAdaptiveSampler } from "@aardworx/wombat.rendering-resources";
import { MockGPU } from "./_mockGpu.js";

describe("prepareAdaptiveSampler", () => {
  it("descriptor → cached GPUSampler; identical descriptors share", () => {
    const gpu = new MockGPU();
    const a = cval(ISampler.fromDescriptor({ magFilter: "linear", minFilter: "linear" }));
    const b = cval(ISampler.fromDescriptor({ magFilter: "linear", minFilter: "linear" }));
    const ra = prepareAdaptiveSampler(gpu.device, a);
    const rb = prepareAdaptiveSampler(gpu.device, b);
    ra.acquire(); rb.acquire();
    const sa = ra.getValue(AdaptiveToken.top);
    const sb = rb.getValue(AdaptiveToken.top);
    expect(sa).toBe(sb);
    expect(gpu.samplers).toHaveLength(1);
    ra.release(); rb.release();
  });

  it("descriptor change fetches a different cached sampler", () => {
    const gpu = new MockGPU();
    const src = cval(ISampler.fromDescriptor({ magFilter: "linear" }));
    const r = prepareAdaptiveSampler(gpu.device, src);
    r.acquire();
    const s1 = r.getValue(AdaptiveToken.top);
    transact(() => { src.value = ISampler.fromDescriptor({ magFilter: "nearest" }); });
    const s2 = r.getValue(AdaptiveToken.top);
    expect(s1).not.toBe(s2);
    expect(gpu.samplers).toHaveLength(2);
    r.release();
  });

  it("gpu source: passes through, no cache write", () => {
    const gpu = new MockGPU();
    const userSampler = { __user: true } as unknown as GPUSampler;
    const r = prepareAdaptiveSampler(gpu.device, cval(ISampler.fromGPU(userSampler)));
    r.acquire();
    expect(r.getValue(AdaptiveToken.top)).toBe(userSampler);
    expect(gpu.samplers).toHaveLength(0);
    r.release();
  });
});
