// BufferView — a typed slice of an `IBuffer` used as a vertex
// attribute, instance attribute, or index source. Mirrors
// Aardvark.Rendering's `BufferView`:
//
//   - `buffer` is `aval<IBuffer>` so per-frame content changes flow
//     through normal aval marking without reshaping the view.
//   - `elementType` carries the element kind; the backend derives
//     `GPUVertexFormat` / `GPUIndexFormat` and a default stride from
//     it.
//   - `singleValue` is set when the view was constructed from a
//     single adaptive value broadcast to all instances/vertices —
//     the runtime can lower it to a uniform binding instead of a
//     per-vertex/per-instance attribute.
//
// The view itself is plain (not wrapped in `aval`). Structural
// fields — `elementType`, `offset`, `stride`, `normalized` — are
// eager so the binding plan / pipeline layout can be computed
// without forcing.

import type { aval } from "@aardworx/wombat.adaptive";
import { AVal } from "@aardworx/wombat.adaptive";
import { IBuffer } from "./buffer.js";
import type { ElementType } from "./elementType.js";
import { ElementType as ElementTypeRegistry } from "./elementType.js";

export interface BufferView {
  /** Adaptive buffer source — content changes flow through this aval. */
  readonly buffer: aval<IBuffer>;
  /** Element kind. Stride defaults from this when `stride` is 0/undefined. */
  readonly elementType: ElementType;
  /** Byte offset into `buffer` where the view starts. Default 0. */
  readonly offset?: number;
  /** Stride between consecutive elements in bytes. 0 = tight (= `byteSize(elementType)`). */
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
// Inference helpers — discover ElementType from a wombat.base packed
// array or a TypedArray. Kept as a duck-type lookup so wombat.base
// doesn't need to expose explicit brands.
// ---------------------------------------------------------------------------

interface WombatArrayLike {
  readonly buffer: ArrayBufferLike;
  readonly length: number;
}

function inferElementType(arr: unknown): ElementType {
  // wombat.base packed types — discover via class name. These all
  // have a distinct `constructor.name` and a `buffer: ArrayBuffer`
  // backing field.
  const ctor = (arr as { constructor?: { name?: string } }).constructor;
  const name = ctor?.name;
  switch (name) {
    case "V2fArray": return "v2f";
    case "V3fArray": return "v3f";
    case "V4fArray": return "v4f";
    case "V2iArray": return "v2i";
    case "V3iArray": return "v3i";
    case "V4iArray": return "v4i";
    case "V2uiArray": return "v2u";
    case "V3uiArray": return "v3u";
    case "V4uiArray": return "v4u";
    // Native typed arrays.
    case "Float32Array": return "f32";
    case "Uint32Array":  return "u32";
    case "Int32Array":   return "i32";
    case "Uint16Array":  return "u16";
    case "Int16Array":   return "i16";
    case "Uint8Array":   return "u8";
    case "Int8Array":    return "i8";
  }
  throw new Error(
    `BufferView.ofArray: cannot infer ElementType from ${name ?? "unknown type"}; ` +
    `pass elementType explicitly`,
  );
}

function arrayToHostBytes(arr: unknown): ArrayBuffer | ArrayBufferView {
  // wombat.base packed types expose `.buffer: ArrayBuffer` covering
  // exactly the packed bytes.
  const a = arr as { buffer?: unknown };
  if (a.buffer instanceof ArrayBuffer) return a.buffer;
  // Native typed array — pass through.
  if (ArrayBuffer.isView(arr as ArrayBufferView)) return arr as ArrayBufferView;
  throw new Error("BufferView.ofArray: unsupported array source");
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

interface OfGPUOptions {
  readonly offset?: number;
  readonly stride?: number;
  readonly normalized?: boolean;
}

function isAval<T>(v: unknown): v is aval<T> {
  return typeof v === "object" && v !== null
    && typeof (v as { getValue?: unknown }).getValue === "function";
}

export const BufferView = {
  /**
   * Build a view from a wombat.base packed array (`V3fArray`,
   * `Uint16Array`, …) or an `aval<…>` of one. The element type is
   * inferred from the array constructor; pass it explicitly via
   * `opts.elementType` for unusual cases.
   */
  ofArray<T extends WombatArrayLike | ArrayBufferView>(
    a: aval<T> | T,
    opts: OfArrayOptions = {},
  ): BufferView {
    const adaptive = isAval<T>(a) ? a : AVal.constant(a);
    // Determine element type — eager. We force the aval ONCE here
    // to discover the array kind, which is structural; subsequent
    // changes are expected to keep the same kind.
    const sample = adaptive.force(/* allow-force */);
    const elementType = opts.elementType ?? inferElementType(sample);
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
   */
  ofValue<T>(
    v: aval<T> | T,
    elementType: ElementType,
  ): BufferView {
    const adaptive = isAval<T>(v) ? v : AVal.constant(v);
    const buffer = adaptive.map((value): IBuffer => {
      const bytes = ElementTypeRegistry.byteSize(elementType);
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
    opts: OfGPUOptions = {},
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
   * type. Used by callers that compute their own `IBuffer` (e.g.
   * compositing multiple attributes into one packed buffer).
   */
  ofBuffer(
    buffer: aval<IBuffer>,
    elementType: ElementType,
    opts: OfGPUOptions = {},
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
   * tight from `elementType`.
   */
  strideOf(v: BufferView): number {
    return v.stride !== undefined && v.stride > 0
      ? v.stride
      : ElementTypeRegistry.byteSize(v.elementType);
  },
  /** Whether this view is a single-value broadcast. */
  isSingleValue(v: BufferView): boolean {
    return v.singleValue !== undefined;
  },
} as const;

// ---------------------------------------------------------------------------
// Single-value packing — write one element of `elementType` into a
// fresh ArrayBuffer. The runtime usually shortcuts this via
// `singleValue` (lifted to a uniform), so this path runs only for
// backends that don't recognize the broadcast.
// ---------------------------------------------------------------------------

function packSingle(value: unknown, t: ElementType, dst: ArrayBuffer): void {
  // Lazy import to avoid a hard dep on wombat.base shapes here. We
  // peek at `_data` (the typed-array view exposed by every wombat
  // math type) and copy its bytes into the destination.
  const v = value as { _data?: ArrayLike<number> };
  if (v._data !== undefined) {
    const src = v._data;
    switch (t) {
      case "v2f": case "v3f": case "v4f":
      case "m22f": case "m33f": case "m44f":
      case "f32": {
        const out = new Float32Array(dst);
        for (let i = 0; i < src.length; i++) out[i] = src[i]!;
        return;
      }
      case "v2i": case "v3i": case "v4i": case "i32": {
        const out = new Int32Array(dst);
        for (let i = 0; i < src.length; i++) out[i] = src[i]!;
        return;
      }
      case "v2u": case "v3u": case "v4u": case "u32": {
        const out = new Uint32Array(dst);
        for (let i = 0; i < src.length; i++) out[i] = src[i]!;
        return;
      }
    }
  }
  // Scalars and unbranded values.
  switch (t) {
    case "f32": new Float32Array(dst)[0] = value as number; return;
    case "u32": new Uint32Array(dst)[0]  = value as number; return;
    case "i32": new Int32Array(dst)[0]   = value as number; return;
    case "u16": new Uint16Array(dst)[0]  = value as number; return;
    case "i16": new Int16Array(dst)[0]   = value as number; return;
    case "u8":  new Uint8Array(dst)[0]   = value as number; return;
    case "i8":  new Int8Array(dst)[0]    = value as number; return;
  }
  throw new Error(`BufferView.ofValue: cannot pack ${typeof value} as ${t}`);
}
