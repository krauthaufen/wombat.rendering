// BufferView — a typed slice of an `IBuffer` used as a vertex
// attribute, instance attribute, or index source. Mirrors
// Aardvark.Rendering's `BufferView`:
//
//   - `buffer` is `aval<IBuffer>` so per-frame content changes flow
//     through normal aval marking without reshaping the view.
//   - `elementType` is a singleton token from the `ElementType`
//     namespace (e.g. `ElementType.V3f`, `ElementType.U16`); the
//     backend reads `byteSize` / `vertexFormat` / `indexFormat`
//     directly off it.
//   - `singleValue` is set when the view was constructed from a
//     single adaptive value broadcast to all instances/vertices —
//     the runtime can lower it to a uniform binding instead of a
//     per-vertex/per-instance attribute.
//
// The view itself is plain (not wrapped in `aval`). Structural
// fields are eager so the binding plan / pipeline layout can be
// computed without forcing.

import type { aval } from "@aardworx/wombat.adaptive";
import { AVal } from "@aardworx/wombat.adaptive";
import { IBuffer } from "./buffer.js";
import { ElementType } from "./elementType.js";
import type { ElementKind } from "./elementType.js";

export interface BufferView {
  /** Adaptive buffer source — content changes flow through this aval. */
  readonly buffer: aval<IBuffer>;
  /** Element kind (singleton from the `ElementType` namespace). */
  readonly elementType: ElementType;
  /** Byte offset into `buffer` where the view starts. Default 0. */
  readonly offset?: number;
  /** Stride between consecutive elements in bytes. 0 = tight (= `elementType.byteSize`). */
  readonly stride?: number;
  /** Integer attributes treated as normalized fixed-point floats. Default false. */
  readonly normalized?: boolean;
  /**
   * Present iff this view was constructed from a single adaptive
   * value (see `BufferView.ofValue`). Lets the backend lower the
   * binding to a uniform instead of a per-vertex/per-instance
   * attribute.
   */
  readonly singleValue?: aval<unknown>;
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

interface OfArrayOptions {
  readonly elementType?: ElementType;
  readonly offset?: number;
  readonly stride?: number;
  readonly normalized?: boolean;
}

interface OfBufferOptions {
  readonly offset?: number;
  readonly stride?: number;
  readonly normalized?: boolean;
}

function isAval<T>(v: unknown): v is aval<T> {
  return typeof v === "object" && v !== null
    && typeof (v as { getValue?: unknown }).getValue === "function";
}

function arrayToHostBytes(arr: unknown): ArrayBuffer | ArrayBufferView {
  // wombat.base packed array types expose `.buffer: ArrayBuffer`.
  const a = arr as { buffer?: unknown };
  if (a.buffer instanceof ArrayBuffer) return a.buffer;
  // Native TypedArray.
  if (ArrayBuffer.isView(arr as ArrayBufferView)) return arr as ArrayBufferView;
  throw new Error("BufferView.ofArray: unsupported array source");
}

export const BufferView = {
  /**
   * Build a view from a wombat.base packed array (`V3fArray`,
   * `Uint16Array`, …) or an `aval<…>` of one. The element kind is
   * inferred from the array's constructor identity; pass it
   * explicitly via `opts.elementType` for unusual cases (e.g. a
   * `Float32Array` interpreted as packed `V4f` with `ElementType.V4f`).
   */
  ofArray<T>(
    a: aval<T> | T,
    opts: OfArrayOptions = {},
  ): BufferView {
    const adaptive = isAval<T>(a) ? a : AVal.constant(a);
    // One-time element-kind discovery from the source's constructor.
    // Marked allow-force: structural-eager, runs at construction only;
    // per-frame content changes flow through `buffer` without re-running.
    const sample = adaptive.force(/* allow-force */);
    const elementType = opts.elementType ?? ElementType.fromArray(sample);
    const buffer = adaptive.map((arr): IBuffer =>
      IBuffer.fromHost(arrayToHostBytes(arr)),
    );
    return {
      buffer,
      elementType,
      ...(opts.offset !== undefined ? { offset: opts.offset } : {}),
      ...(opts.stride !== undefined ? { stride: opts.stride } : {}),
      ...(opts.normalized !== undefined ? { normalized: opts.normalized } : {}),
    };
  },

  /**
   * Build a view that broadcasts a single adaptive value to every
   * vertex / instance. The runtime detects the `singleValue` field
   * and lowers it to a uniform binding instead of a per-element
   * attribute. The `buffer` aval still carries packed bytes for
   * backends that don't recognize the fast path.
   *
   * The `T` parameter on `ElementKind<T>` flows through to the
   * value type — passing `ElementType.V4f` requires `T = V4f` here.
   */
  ofValue<T>(
    v: aval<T> | T,
    elementType: ElementKind<T>,
  ): BufferView {
    const adaptive = isAval<T>(v) ? v : AVal.constant(v);
    const buffer = adaptive.map((value): IBuffer => {
      const bytes = elementType.byteSize;
      const ab = new ArrayBuffer(bytes);
      packSingle(value, elementType, ab);
      return IBuffer.fromHost(ab);
    });
    return {
      buffer,
      elementType,
      singleValue: adaptive,
    };
  },

  /**
   * Build a view backed by a user-supplied `GPUBuffer`. The
   * runtime binds it directly without an upload step.
   */
  ofGPU(
    buffer: aval<GPUBuffer> | GPUBuffer,
    elementType: ElementType,
    opts: OfBufferOptions = {},
  ): BufferView {
    const adaptive = isAval<GPUBuffer>(buffer) ? buffer : AVal.constant(buffer);
    const ibuffer = adaptive.map((b): IBuffer => IBuffer.fromGPU(b));
    return {
      buffer: ibuffer,
      elementType,
      ...(opts.offset !== undefined ? { offset: opts.offset } : {}),
      ...(opts.stride !== undefined ? { stride: opts.stride } : {}),
      ...(opts.normalized !== undefined ? { normalized: opts.normalized } : {}),
    };
  },

  /**
   * Wrap an already-built `aval<IBuffer>` with explicit element
   * kind. Used by callers that compute their own `IBuffer` (e.g.
   * compositing multiple attributes into one packed buffer).
   */
  ofBuffer(
    buffer: aval<IBuffer>,
    elementType: ElementType,
    opts: OfBufferOptions = {},
  ): BufferView {
    return {
      buffer,
      elementType,
      ...(opts.offset !== undefined ? { offset: opts.offset } : {}),
      ...(opts.stride !== undefined ? { stride: opts.stride } : {}),
      ...(opts.normalized !== undefined ? { normalized: opts.normalized } : {}),
    };
  },

  // ---------- accessors ----------

  /** Byte offset, defaulting to 0. */
  offsetOf(v: BufferView): number { return v.offset ?? 0; },
  /**
   * Stride in bytes — `v.stride` if set and non-zero, otherwise
   * tight from `elementType.byteSize`.
   */
  strideOf(v: BufferView): number {
    return v.stride !== undefined && v.stride > 0
      ? v.stride
      : v.elementType.byteSize;
  },
  /** Whether this view is a single-value broadcast. */
  isSingleValue(v: BufferView): boolean {
    return v.singleValue !== undefined;
  },
} as const;

// ---------------------------------------------------------------------------
// Single-value packing — write one element of `t` into a fresh
// ArrayBuffer. The runtime usually shortcuts this via `singleValue`
// (lifted to a uniform); this path runs only for backends that
// don't recognise the broadcast.
// ---------------------------------------------------------------------------

function packSingle(value: unknown, t: ElementType, dst: ArrayBuffer): void {
  const v = value as { _data?: ArrayLike<number> };
  if (v._data !== undefined) {
    const src = v._data;
    // Match by name on the singleton — concise and minifier-safe
    // since the names are baked at the singleton's construction.
    const name = t.name;
    if (name === "f32"
        || name.startsWith("v") && name.endsWith("f")
        || name.startsWith("m") && name.endsWith("f")) {
      const out = new Float32Array(dst);
      for (let i = 0; i < src.length; i++) out[i] = src[i]!;
      return;
    }
    if (name === "i32" || (name.startsWith("v") && name.endsWith("i"))) {
      const out = new Int32Array(dst);
      for (let i = 0; i < src.length; i++) out[i] = src[i]!;
      return;
    }
    if (name === "u32" || (name.startsWith("v") && name.endsWith("u"))) {
      const out = new Uint32Array(dst);
      for (let i = 0; i < src.length; i++) out[i] = src[i]!;
      return;
    }
  }
  // Scalars and unbranded values.
  switch (t.name) {
    case "f32": new Float32Array(dst)[0] = value as number; return;
    case "u32": new Uint32Array(dst)[0]  = value as number; return;
    case "i32": new Int32Array(dst)[0]   = value as number; return;
    case "u16": new Uint16Array(dst)[0]  = value as number; return;
    case "i16": new Int16Array(dst)[0]   = value as number; return;
    case "u8":  new Uint8Array(dst)[0]   = value as number; return;
    case "i8":  new Int8Array(dst)[0]    = value as number; return;
  }
  throw new Error(`BufferView.ofValue: cannot pack ${typeof value} as ${t.name}`);
}
