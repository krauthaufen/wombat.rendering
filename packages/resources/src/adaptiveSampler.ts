// prepareAdaptiveSampler — lift `aval<ISampler>` to
// `AdaptiveResource<GPUSampler>`.
//
// Samplers are cheap and structurally-identifiable — the device
// dedupes them via a descriptor cache. Per AdaptiveResource we
// just track the most recent handle; on descriptor change we
// fetch a fresh one from the cache.

import {
  AdaptiveResource,
  type ISampler,
} from "@aardworx/wombat.rendering-core";
import {
  type AdaptiveToken,
  type aval,
} from "@aardworx/wombat.adaptive";

class SamplerCache {
  private readonly map = new Map<string, GPUSampler>();
  constructor(private readonly device: GPUDevice) {}
  get(desc: GPUSamplerDescriptor): GPUSampler {
    const k = key(desc);
    let s = this.map.get(k);
    if (s === undefined) {
      s = this.device.createSampler(desc);
      this.map.set(k, s);
    }
    return s;
  }
}

function key(d: GPUSamplerDescriptor): string {
  return [
    d.addressModeU ?? "clamp-to-edge",
    d.addressModeV ?? "clamp-to-edge",
    d.addressModeW ?? "clamp-to-edge",
    d.magFilter ?? "nearest",
    d.minFilter ?? "nearest",
    d.mipmapFilter ?? "nearest",
    d.lodMinClamp ?? 0,
    d.lodMaxClamp ?? 32,
    d.compare ?? "",
    d.maxAnisotropy ?? 1,
  ].join("|");
}

const caches = new WeakMap<GPUDevice, SamplerCache>();
function cacheFor(device: GPUDevice): SamplerCache {
  let c = caches.get(device);
  if (c === undefined) {
    c = new SamplerCache(device);
    caches.set(device, c);
  }
  return c;
}

class AdaptiveSampler extends AdaptiveResource<GPUSampler> {
  constructor(
    private readonly device: GPUDevice,
    private readonly source: aval<ISampler>,
  ) { super(); }
  protected create(): void {}
  protected destroy(): void {
    // We never own the GPUSampler — the cache does.
  }
  override compute(token: AdaptiveToken): GPUSampler {
    const s = this.source.getValue(token);
    if (s.kind === "gpu") return s.sampler;
    return cacheFor(this.device).get(s.descriptor);
  }
}

export function prepareAdaptiveSampler(
  device: GPUDevice,
  source: aval<ISampler>,
): AdaptiveResource<GPUSampler> {
  return new AdaptiveSampler(device, source);
}
