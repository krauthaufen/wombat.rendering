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

export const IBuffer = {
  fromGPU(buffer: GPUBuffer, sizeBytes?: number): IBuffer {
    return { kind: "gpu", buffer, sizeBytes: sizeBytes ?? buffer.size };
  },
  /**
   * Wrap a host-side `ArrayBuffer` / `ArrayBufferView`. Pass any
   * wombat.base packed array's `.buffer` directly, or a `Float32Array`,
   * etc.
   */
  fromHost(data: HostBufferSource): IBuffer {
    const sizeBytes =
      data instanceof ArrayBuffer
        ? data.byteLength
        : data.byteLength;
    return { kind: "host", data, sizeBytes };
  },
} as const;
