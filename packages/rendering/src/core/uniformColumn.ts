// Uniform columns — many uniform values that change together.
//
// A `UniformColumn` is one array holding the same uniform for many
// draws (20 000 model matrices, a tint per monster, …), behind ONE
// aval. A draw takes its value with `column.at(i)`, which is an
// ordinary `aval` and can be supplied wherever a uniform value is.
//
// What that buys depends on who reads it:
//   - the heap path recognises the cell, stores the WHOLE column once in
//     its arena (shared by every draw that references it) and points the
//     draw at its element. Changing all values is one array write, one
//     mark and one upload — no aval is evaluated or marked per draw.
//     The shader is unchanged: it reads a plain uniform.
//   - everything else (the classic path, a derived-uniform rule, user
//     code) reads the cell as the aval it is and gets the element's value.
//
// So batching stays a decision of whoever builds the scene, visible in
// what they supply, and never leaks into an effect.
//
// Values are stored the way the heap stores a uniform of that type: a
// mat4 is 16 floats ROW-major (as `M44d.copyTo` writes them), a vec3 is
// 3 floats. A plain column packs one value per row; a `uniformTable`
// packs SEVERAL fields per row (a struct per row) in one array, so one
// write updates every field of every draw.

import { AVal, type AdaptiveToken, type aval } from "@aardworx/wombat.adaptive";
import { M44d, Trafo3d, V2f, V3f, V4f } from "@aardworx/wombat.base";

export type UniformColumnData = Float32Array | Uint32Array | Int32Array;

/** Element type of a column, named like the WGSL type of the uniform it feeds. */
export type UniformColumnType =
  | "f32" | "u32" | "i32" | "vec2<f32>" | "vec3<f32>" | "vec4<f32>" | "mat4x4<f32>";

interface ColumnTypeInfo {
  /** Bytes of one value. */
  readonly bytes: number;
  /** Byte alignment the heap needs for a value of this type (its vec4 view serves vec4 and mat4). */
  readonly align: number;
  /** `f`/`u`/`i`: the value at 4-byte word `w` of the float / unsigned / signed view of the data. */
  readonly decode: (f: Float32Array, u: Uint32Array, i: Int32Array, w: number) => unknown;
}

const TYPES: Record<UniformColumnType, ColumnTypeInfo> = {
  "f32": { bytes: 4, align: 4, decode: (f, _u, _i, w) => f[w]! },
  "u32": { bytes: 4, align: 4, decode: (_f, u, _i, w) => u[w]! },
  "i32": { bytes: 4, align: 4, decode: (_f, _u, i, w) => i[w]! },
  "vec2<f32>": { bytes: 8, align: 4, decode: (f, _u, _i, w) => new V2f(f[w]!, f[w + 1]!) },
  "vec3<f32>": { bytes: 12, align: 4, decode: (f, _u, _i, w) => new V3f(f[w]!, f[w + 1]!, f[w + 2]!) },
  "vec4<f32>": { bytes: 16, align: 16, decode: (f, _u, _i, w) => new V4f(f[w]!, f[w + 1]!, f[w + 2]!, f[w + 3]!) },
  // A Trafo3d, so the value also serves where an inverse is wanted.
  "mat4x4<f32>": { bytes: 64, align: 16, decode: (f, _u, _i, w) => Trafo3d.fromMatrix(M44d.fromArray(f.subarray(w, w + 16))) },
};

interface ColumnValueOf {
  "f32": number; "u32": number; "i32": number;
  "vec2<f32>": V2f; "vec3<f32>": V3f; "vec4<f32>": V4f; "mat4x4<f32>": Trafo3d;
}

export interface UniformColumn<T = unknown> {
  readonly type: UniformColumnType;
  /** Bytes of one value. */
  readonly bytes: number;
  /** Bytes from one row to the next. */
  readonly stride: number;
  /** Byte offset of this column's value within a row (0 for a plain column). */
  readonly offset: number;
  /** The whole array; replacing or marking it changes every cell at once. */
  readonly data: aval<UniformColumnData>;
  /** The value in row `index`, as a uniform value. */
  at(index: number | aval<number>): UniformColumnCell<T>;
}

/** One value of a column: an aval of it that also names where it comes from. */
export interface UniformColumnCell<T = unknown> extends aval<T> {
  readonly column: UniformColumn<T>;
  readonly index: number | aval<number>;
}

function makeColumn<K extends UniformColumnType>(
  type: K, data: aval<UniformColumnData>, stride: number, offset: number,
): UniformColumn<ColumnValueOf[K]> {
  const info = TYPES[type];
  if (info === undefined) throw new Error(`uniform column: unsupported element type '${type}'`);
  if (stride % info.align !== 0 || offset % info.align !== 0) {
    throw new Error(
      `uniform column: a '${type}' value must sit on a ${info.align}-byte boundary ` +
      `(row stride ${stride}, offset ${offset})`,
    );
  }
  // views over the current array (rebuilt when the array is replaced)
  let viewsOf: UniformColumnData | undefined;
  let f!: Float32Array, u!: Uint32Array, i!: Int32Array;
  const column: UniformColumn<ColumnValueOf[K]> = {
    type, bytes: info.bytes, stride, offset, data,
    at(index) {
      const cell = AVal.custom((token: AdaptiveToken) => {
        const d = data.getValue(token);
        if (d !== viewsOf) {
          viewsOf = d;
          const words = d.byteLength >>> 2;
          f = new Float32Array(d.buffer, d.byteOffset, words);
          u = new Uint32Array(d.buffer, d.byteOffset, words);
          i = new Int32Array(d.buffer, d.byteOffset, words);
        }
        const row = typeof index === "number" ? index : index.getValue(token);
        return info.decode(f, u, i, (row * stride + offset) >>> 2) as ColumnValueOf[K];
      }) as unknown as { column: unknown; index: unknown };
      cell.column = column;
      cell.index = index;
      return cell as unknown as UniformColumnCell<ColumnValueOf[K]>;
    },
  };
  return column;
}

const asAval = (data: aval<UniformColumnData> | UniformColumnData): aval<UniformColumnData> =>
  (typeof (data as { getValue?: unknown }).getValue === "function" ? data : AVal.constant(data)) as aval<UniformColumnData>;

/**
 * A column of uniform values: `data` holds one tightly packed value per
 * row. Pass an aval of the array to replace it, or to mark it changed
 * after writing into it.
 */
export function uniformColumn<K extends UniformColumnType>(
  type: K, data: aval<UniformColumnData> | UniformColumnData,
): UniformColumn<ColumnValueOf[K]> {
  return makeColumn(type, asAval(data), TYPES[type]?.bytes ?? 0, 0);
}

/** A table: rows of several values each, in ONE array (a struct per row). */
export interface UniformTable<F extends Record<string, UniformColumnType>> {
  /** Bytes per row. */
  readonly stride: number;
  /** Byte offset of each field within a row. */
  readonly offsets: { readonly [N in keyof F]: number };
  readonly columns: { readonly [N in keyof F]: UniformColumn<ColumnValueOf[F[N]]> };
  readonly data: aval<UniformColumnData>;
  /** Every field of row `index`, by name — ready to supply as uniforms. */
  row(index: number | aval<number>): { readonly [N in keyof F]: UniformColumnCell<ColumnValueOf[F[N]]> };
}

/**
 * Lay out rows holding the given fields, in order. A vec4 or mat4 field
 * starts on a 16-byte boundary and everything else on a 4-byte one; a
 * row with such a field is padded to a multiple of 16 bytes.
 * `tableLayout(fields)` gives the same numbers without any data, to size
 * and fill the array.
 */
export function uniformTable<F extends Record<string, UniformColumnType>>(
  fields: F, data: aval<UniformColumnData> | UniformColumnData,
): UniformTable<F> {
  const { stride, offsets } = tableLayout(fields);
  const dataAval = asAval(data);
  const columns: Record<string, UniformColumn> = {};
  for (const name of Object.keys(fields)) {
    columns[name] = makeColumn(fields[name]!, dataAval, stride, offsets[name]!);
  }
  return {
    stride, offsets, data: dataAval,
    columns: columns as UniformTable<F>["columns"],
    row(index) {
      const out: Record<string, UniformColumnCell> = {};
      for (const name of Object.keys(columns)) out[name] = columns[name]!.at(index);
      return out as ReturnType<UniformTable<F>["row"]>;
    },
  };
}

/** Row stride and field offsets (bytes) of a table with these fields. */
export function tableLayout<F extends Record<string, UniformColumnType>>(
  fields: F,
): { stride: number; offsets: { [N in keyof F]: number } } {
  let off = 0, align = 4;
  const offsets: Record<string, number> = {};
  for (const name of Object.keys(fields)) {
    const info = TYPES[fields[name]!];
    if (info === undefined) throw new Error(`uniform table: unsupported type '${fields[name]}' for field '${name}'`);
    off = Math.ceil(off / info.align) * info.align;
    offsets[name] = off;
    off += info.bytes;
    if (info.align > align) align = info.align;
  }
  return { stride: Math.ceil(off / align) * align, offsets: offsets as { [N in keyof F]: number } };
}

/** Whether a uniform value is a cell of a `UniformColumn`. */
export function isUniformColumnCell(v: unknown): v is UniformColumnCell {
  if (typeof v !== "object" || v === null) return false;
  const c = (v as { column?: { data?: unknown; stride?: unknown } }).column;
  return typeof c === "object" && c !== null && typeof c.stride === "number" && c.data !== undefined
    && (v as { index?: unknown }).index !== undefined;
}
