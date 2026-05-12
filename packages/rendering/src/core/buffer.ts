// IBuffer — a buffer source that can be either a real GPU buffer
// or host-side memory (an ArrayBuffer, a TypedArray, or a packed
// view from wombat.base such as `V2fArray._data` or `arr.buffer`).
//
// The user is free to mix and match: an `aval<IBuffer>` can flip
// between `gpu` and `host` over its lifetime, and the rendering
// layer transparently handles the upload when a `host` value is
// observed.
//
// `host.data` is typed as `BufferSource` (DOM standard) which
// admits `ArrayBuffer`, `SharedArrayBuffer`, and any
// `ArrayBufferView` — covering wombat.base's packed array views
// that expose a `.buffer` (ArrayBuffer) or a typed array directly.
//
// Equality / hash protocol: `IBuffer` values flow through
// `aval<IBuffer>` into identity-keyed caches (the index pool, the
// arena attribute pool via `BufferView.buffer`). We compare by the
// underlying *resource*:
//   - `gpu`  → reference equality on the `GPUBuffer` (opaque handle).
//   - `host` → reference equality on the `data` payload (an
//              `ArrayBuffer`/typed-array is "non-trivial" — comparing
//              contents is too costly; two views over the same backing
//              buffer with the same byte range are equal, anything
//              else isn't).
// The factories additionally intern by that resource (`WeakMap`) so
// `IBuffer.fromHost(arr)` / `IBuffer.fromGPU(buf)` called twice for the
// same `arr`/`buf` return the *same* object — `===` true, no leak
// (WeakMap keyed by the resource the caller already holds).

export type HostBufferSource = ArrayBuffer | ArrayBufferView;

export type IBuffer =
  | {
      readonly kind: "gpu";
      readonly buffer: GPUBuffer;
      readonly sizeBytes: number;
    }
  | {
      readonly kind: "host";
      readonly data: HostBufferSource;
      readonly sizeBytes: number;
    };

let _idCounter = 0;
const _idHashes = new WeakMap<object, number>();
function idHash(o: object): number {
  let v = _idHashes.get(o);
  if (v === undefined) { v = (++_idCounter) | 0; _idHashes.set(o, v); }
  return v;
}

function iBufferEquals(a: IBuffer, o: unknown): boolean {
  if (a === o) return true;
  if (o === null || typeof o !== "object") return false;
  const b = o as IBuffer;
  if (a.kind !== b.kind) return false;
  if (a.kind === "gpu") return a.buffer === (b as { buffer: GPUBuffer }).buffer;
  return a.data === (b as { data: HostBufferSource }).data
    && a.sizeBytes === b.sizeBytes;
}
function iBufferHash(a: IBuffer): number {
  if (a.kind === "gpu") return idHash(a.buffer);
  return (idHash(a.data as object) * 31 + a.sizeBytes) | 0;
}
const IBUFFER_PROTO = {
  equals(this: IBuffer, o: unknown): boolean { return iBufferEquals(this, o); },
  getHashCode(this: IBuffer): number { return iBufferHash(this) | 0; },
};
function withEq<T extends IBuffer>(t: T): T {
  return Object.assign(Object.create(IBUFFER_PROTO) as object, t) as T;
}

const _gpuIntern = new WeakMap<GPUBuffer, IBuffer>();
const _hostIntern = new WeakMap<object, IBuffer>();

export const IBuffer = {
  fromGPU(buffer: GPUBuffer, sizeBytes?: number): IBuffer {
    const cached = _gpuIntern.get(buffer);
    if (cached !== undefined && cached.sizeBytes === (sizeBytes ?? buffer.size)) return cached;
    const t = withEq({ kind: "gpu" as const, buffer, sizeBytes: sizeBytes ?? buffer.size });
    _gpuIntern.set(buffer, t);
    return t;
  },
  /**
   * Wrap a host-side `ArrayBuffer` / `ArrayBufferView`. Pass any
   * wombat.base packed array's `.buffer` directly, or a `Float32Array`,
   * etc.
   */
  fromHost(data: HostBufferSource): IBuffer {
    const sizeBytes = data.byteLength;
    const cached = _hostIntern.get(data as object);
    if (cached !== undefined && cached.sizeBytes === sizeBytes) return cached;
    const t = withEq({ kind: "host" as const, data, sizeBytes });
    _hostIntern.set(data as object, t);
    return t;
  },
} as const;
