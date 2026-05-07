// ElementType — runtime token + structural metadata for the data
// type of a single element in a `BufferView`. The TypeScript-side
// equivalent of F#'s `typeof<T>` / `System.Type`: a value (not a
// string tag) carrying byte size, vertex / index format, and a
// phantom type parameter that flows through to `BufferView<T>`-
// generic call sites.
//
// All entries are exposed as singleton properties on `ElementType`:
//
//   ElementType.V3f         // ElementKind<V3f>
//   ElementType.U16         // ElementKind<number>
//   ElementType.M44f        // ElementKind<M44f>
//
// `BufferView.ofArray` infers the kind from the source array's
// constructor identity (V3fArray → ElementType.V3f, Float32Array →
// ElementType.F32, etc.) so the explicit second argument is rarely
// needed.

import type {
  V2f, V3f, V4f, M22f, M33f, M44f, V2i, V3i, V4i,
} from "@aardworx/wombat.base";
import {
  V2fArray, V3fArray, V4fArray,
  V2iArray, V3iArray, V4iArray,
  V2uiArray, V3uiArray, V4uiArray,
} from "@aardworx/wombat.base";

// ---------------------------------------------------------------------------
// ElementKind<T>
// ---------------------------------------------------------------------------

/**
 * Phantom-typed token describing a single element's data type. The
 * `T` parameter is a brand only — it carries the source-side type
 * (V3f, M44f, number, …) into the type system at zero runtime cost.
 *
 * Identity equality is the equality used everywhere in the renderer
 * (group keys, bind plans). All instances are singletons under the
 * `ElementType` namespace below.
 */
export class ElementKind<T> {
  constructor(
    /** Short human-readable tag (e.g. `"v3f"`, `"u16"`). */
    readonly name: string,
    /** Size of one element in bytes. */
    readonly byteSize: number,
    /** GPUVertexFormat for use as a vertex attribute. `undefined` for matrix and non-vertex types. */
    readonly vertexFormat: GPUVertexFormat | undefined,
    /** GPUIndexFormat when used as an index buffer. `undefined` unless the type is `u16` / `u32`. */
    readonly indexFormat: GPUIndexFormat | undefined,
  ) {}

  /** Phantom — never read at runtime. Forces TS to track T. */
  declare readonly _brand: T;

  toString(): string { return this.name; }
}

/**
 * `ElementType` doubles as a type alias (the structural shape) and
 * a value namespace (the singleton tokens). `ElementType` in type
 * position is `ElementKind<unknown>`; in value position it's the
 * collection of singletons below.
 */
export type ElementType = ElementKind<unknown>;

// ---------------------------------------------------------------------------
// Singleton tokens
// ---------------------------------------------------------------------------

const _V2f  = new ElementKind<V2f>("v2f",   8, "float32x2", undefined);
const _V3f  = new ElementKind<V3f>("v3f",  12, "float32x3", undefined);
const _V4f  = new ElementKind<V4f>("v4f",  16, "float32x4", undefined);
const _M22f = new ElementKind<M22f>("m22f", 16, undefined, undefined);
const _M33f = new ElementKind<M33f>("m33f", 36, undefined, undefined);
const _M44f = new ElementKind<M44f>("m44f", 64, undefined, undefined);
const _V2i  = new ElementKind<V2i>("v2i",   8, "sint32x2", undefined);
const _V3i  = new ElementKind<V3i>("v3i",  12, "sint32x3", undefined);
const _V4i  = new ElementKind<V4i>("v4i",  16, "sint32x4", undefined);
const _V2u  = new ElementKind<V2i>("v2u",   8, "uint32x2", undefined);
const _V3u  = new ElementKind<V3i>("v3u",  12, "uint32x3", undefined);
const _V4u  = new ElementKind<V4i>("v4u",  16, "uint32x4", undefined);
const _F32  = new ElementKind<number>("f32", 4, "float32", undefined);
const _U32  = new ElementKind<number>("u32", 4, "uint32",  "uint32");
const _I32  = new ElementKind<number>("i32", 4, "sint32",  undefined);
const _U16  = new ElementKind<number>("u16", 2, undefined, "uint16");
const _I16  = new ElementKind<number>("i16", 2, undefined, undefined);
const _U8   = new ElementKind<number>("u8",  1, undefined, undefined);
const _I8   = new ElementKind<number>("i8",  1, undefined, undefined);

/**
 * Construct a vertex view; throws if the kind has no vertex format
 * (matrix / unsupported type).
 */
function asVertexFormat(t: ElementType): GPUVertexFormat {
  if (t.vertexFormat === undefined) {
    throw new Error(
      `ElementType.${t.name}: no GPUVertexFormat (matrices must be split into column attributes; ` +
      `unsupported types must be repacked first)`,
    );
  }
  return t.vertexFormat;
}

function asIndexFormat(t: ElementType): GPUIndexFormat {
  if (t.indexFormat === undefined) {
    throw new Error(`ElementType.${t.name}: not a valid index type (use ElementType.U16 or ElementType.U32)`);
  }
  return t.indexFormat;
}

// ---------------------------------------------------------------------------
// Constructor → ElementType registry — drives `BufferView.ofArray`
// auto-inference. Keyed on class identity (V3fArray, Float32Array,
// …) so it survives minification.
// ---------------------------------------------------------------------------

const FROM_CTOR: ReadonlyMap<unknown, ElementType> = new Map<unknown, ElementType>([
  // wombat.base packed arrays
  [V2fArray,  _V2f], [V3fArray, _V3f], [V4fArray, _V4f],
  [V2iArray,  _V2i], [V3iArray, _V3i], [V4iArray, _V4i],
  [V2uiArray, _V2u], [V3uiArray, _V3u], [V4uiArray, _V4u],
  // native typed arrays (interpreted as scalars)
  [Float32Array, _F32],
  [Uint32Array,  _U32],
  [Int32Array,   _I32],
  [Uint16Array,  _U16],
  [Int16Array,   _I16],
  [Uint8Array,   _U8],
  [Int8Array,    _I8],
]);

function fromArray(arr: unknown): ElementType {
  const ctor = (arr as { constructor?: unknown }).constructor;
  if (ctor === undefined) {
    throw new Error("ElementType.fromArray: input has no constructor");
  }
  const et = FROM_CTOR.get(ctor);
  if (et === undefined) {
    const name = (ctor as { name?: string }).name ?? "<anonymous>";
    throw new Error(
      `ElementType.fromArray: cannot infer ElementType from ${name}; ` +
      `pass the kind explicitly (e.g. ElementType.V3f)`,
    );
  }
  return et;
}

// ---------------------------------------------------------------------------
// Public namespace
// ---------------------------------------------------------------------------

/**
 * Singleton tokens + helpers. Most call sites use the singleton
 * directly (`ElementType.V3f`); the helpers are kept on the
 * namespace so `import { ElementType }` is the only import you
 * need.
 */
export const ElementType = {
  // Float vectors
  V2f:  _V2f, V3f: _V3f, V4f: _V4f,
  // Float matrices (square)
  M22f: _M22f, M33f: _M33f, M44f: _M44f,
  // Signed-int vectors
  V2i: _V2i, V3i: _V3i, V4i: _V4i,
  // Unsigned-int vectors
  V2u: _V2u, V3u: _V3u, V4u: _V4u,
  // Scalars
  F32: _F32, U32: _U32, I32: _I32,
  U16: _U16, I16: _I16, U8: _U8, I8: _I8,

  /** Infer from a wombat.base packed array or a native TypedArray. */
  fromArray,
  /** GPUVertexFormat for `t`; throws if `t` has no vertex form. */
  toVertexFormat: asVertexFormat,
  /** GPUIndexFormat for `t`; throws if `t` is not `u16` / `u32`. */
  toIndexFormat:  asIndexFormat,
} as const;
