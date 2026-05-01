// ISampler — sampler source. Either a pre-built `GPUSampler` or a
// descriptor for the runtime to materialise (samplers are cheap
// and dedupable, so descriptors are the common path).

export type ISampler =
  | { readonly kind: "gpu"; readonly sampler: GPUSampler }
  | { readonly kind: "desc"; readonly descriptor: GPUSamplerDescriptor };

export const ISampler = {
  fromGPU(sampler: GPUSampler): ISampler { return { kind: "gpu", sampler }; },
  fromDescriptor(descriptor: GPUSamplerDescriptor): ISampler { return { kind: "desc", descriptor }; },
} as const;
