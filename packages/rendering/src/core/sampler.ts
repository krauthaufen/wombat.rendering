// ISampler — sampler source. Either a pre-built `GPUSampler` or a
// descriptor for the runtime to materialise (samplers are cheap
// and dedupable, so descriptors are the common path).
//
// Equality / hash protocol: `desc` samplers compare structurally over
// the descriptor fields (all primitives) so two `fromDescriptor({...})`
// with the same options collapse in `aval<ISampler>`-keyed caches;
// `gpu` samplers compare by reference on the opaque `GPUSampler`.
// `gpu` samplers are additionally interned by the handle so two
// `fromGPU(sameSampler)` return the same object.

export type ISampler =
  | { readonly kind: "gpu"; readonly sampler: GPUSampler }
  | { readonly kind: "desc"; readonly descriptor: GPUSamplerDescriptor };

let _idCounter = 0;
const _idHashes = new WeakMap<object, number>();
function idHash(o: object): number {
  let v = _idHashes.get(o);
  if (v === undefined) { v = (++_idCounter) | 0; _idHashes.set(o, v); }
  return v;
}
function hashStr(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return h;
}
// Stable serialisation of a GPUSamplerDescriptor (sorted keys, only
// own enumerable primitive fields — the descriptor never contains
// objects). Used for both equality and hashing.
function descKey(d: GPUSamplerDescriptor): string {
  const keys = Object.keys(d).sort();
  let out = "";
  for (const k of keys) {
    const v = (d as Record<string, unknown>)[k];
    out += `${k}=${String(v)};`;
  }
  return out;
}

function iSamplerEquals(a: ISampler, o: unknown): boolean {
  if (a === o) return true;
  if (o === null || typeof o !== "object") return false;
  const b = o as ISampler;
  if (a.kind !== b.kind) return false;
  if (a.kind === "gpu") return a.sampler === (b as { sampler: GPUSampler }).sampler;
  return descKey(a.descriptor) === descKey((b as { descriptor: GPUSamplerDescriptor }).descriptor);
}
function iSamplerHash(a: ISampler): number {
  if (a.kind === "gpu") return idHash(a.sampler);
  return hashStr("desc:" + descKey(a.descriptor));
}
const ISAMPLER_PROTO = {
  equals(this: ISampler, o: unknown): boolean { return iSamplerEquals(this, o); },
  getHashCode(this: ISampler): number { return iSamplerHash(this) | 0; },
};
function withEq<T extends ISampler>(t: T): T {
  return Object.assign(Object.create(ISAMPLER_PROTO) as object, t) as T;
}

const _gpuIntern = new WeakMap<GPUSampler, ISampler>();

export const ISampler = {
  fromGPU(sampler: GPUSampler): ISampler {
    const cached = _gpuIntern.get(sampler);
    if (cached !== undefined) return cached;
    const t = withEq({ kind: "gpu" as const, sampler });
    _gpuIntern.set(sampler, t);
    return t;
  },
  fromDescriptor(descriptor: GPUSamplerDescriptor): ISampler {
    return withEq({ kind: "desc" as const, descriptor });
  },
} as const;
