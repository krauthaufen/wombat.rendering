// WGSL-type → JS-value packers used by heapScene to write per-draw
// uniforms into the arena's Float32Array staging mirror.
//
// One packer per supported WGSL type. The `pack` callback takes the
// aval's `.getValue(tok)` result and writes its bytes to `dst` at
// float offset `off`. Coerces between Trafo3d / M44d / V4f / V3d /
// V3f / number as needed.
//
// Integer scalar / vector packers go through a same-buffer Uint32/
// Int32 view to avoid the lossy i32 → f32 coercion you'd get from a
// direct `dst[off] = ...` assignment.

import type { M44d, V4f } from "@aardworx/wombat.base";

export interface WgslPacker {
  /** Tightly-packed size in bytes of one value (mat4 = 64, vec3 = 12, …). */
  readonly dataBytes: number;
  readonly typeId: number;
  /**
   * Pack `val` (the aval's `.getValue(tok)` result) into `dst` at
   * float offset `off`.
   */
  readonly pack: (val: unknown, dst: Float32Array, off: number) => void;
}

function packMat44(m: M44d, dst: Float32Array, off: number): void {
  // Zero-alloc flat copy (row-major) straight into the f32 staging
  // buffer — `copyTo` does `dst.set(m._data, off)` which narrows f64→f32
  // on store, no throwaway `number[]` per call.
  m.copyTo(dst, off);
}

export const PACKER_MAT4: WgslPacker = {
  dataBytes: 64, typeId: 0,
  pack: (val, dst, off) => {
    // Accept Trafo3d (uses .forward) or M44d directly.
    const m = (val as { forward?: M44d }).forward !== undefined
      ? (val as { forward: M44d }).forward
      : (val as M44d);
    packMat44(m, dst, off);
  },
};

export const PACKER_VEC4: WgslPacker = {
  dataBytes: 16, typeId: 0,
  pack: (val, dst, off) => {
    const v = val as V4f;
    dst[off + 0] = v.x; dst[off + 1] = v.y;
    dst[off + 2] = v.z; dst[off + 3] = v.w;
  },
};

export const PACKER_VEC3: WgslPacker = {
  dataBytes: 12, typeId: 0,
  pack: (val, dst, off) => {
    // V3f or V3d both expose .x/.y/.z; cast through a common shape.
    const v = val as { x: number; y: number; z: number };
    dst[off + 0] = v.x; dst[off + 1] = v.y; dst[off + 2] = v.z;
  },
};

export const PACKER_VEC2: WgslPacker = {
  dataBytes: 8, typeId: 0,
  pack: (val, dst, off) => {
    const v = val as { x: number; y: number };
    dst[off + 0] = v.x; dst[off + 1] = v.y;
  },
};

export const PACKER_F32: WgslPacker = {
  dataBytes: 4, typeId: 0,
  pack: (val, dst, off) => { dst[off] = val as number; },
};

function makeIntPacker(
  ctor: typeof Uint32Array | typeof Int32Array,
  dim: 1 | 2 | 3 | 4,
): WgslPacker {
  const bytes = dim * 4;
  if (dim === 1) {
    return {
      dataBytes: bytes, typeId: 0,
      pack: (val, dst, off) => {
        new ctor(dst.buffer as ArrayBuffer, dst.byteOffset + off * 4, 1)[0] = val as number;
      },
    };
  }
  return {
    dataBytes: bytes, typeId: 0,
    pack: (val, dst, off) => {
      const view = new ctor(dst.buffer as ArrayBuffer, dst.byteOffset + off * 4, dim);
      const v = val as { x: number; y: number; z?: number; w?: number };
      view[0] = v.x; view[1] = v.y;
      if (dim >= 3) view[2] = v.z!;
      if (dim >= 4) view[3] = v.w!;
    },
  };
}

export const PACKER_U32     = makeIntPacker(Uint32Array, 1);
export const PACKER_UVEC2   = makeIntPacker(Uint32Array, 2);
export const PACKER_UVEC3   = makeIntPacker(Uint32Array, 3);
export const PACKER_UVEC4   = makeIntPacker(Uint32Array, 4);
export const PACKER_I32     = makeIntPacker(Int32Array, 1);
export const PACKER_IVEC2   = makeIntPacker(Int32Array, 2);
export const PACKER_IVEC3   = makeIntPacker(Int32Array, 3);
export const PACKER_IVEC4   = makeIntPacker(Int32Array, 4);

export function packerForWgslType(wgslType: string): WgslPacker {
  switch (wgslType) {
    case "mat4x4<f32>": return PACKER_MAT4;
    case "vec4<f32>":   return PACKER_VEC4;
    case "vec3<f32>":   return PACKER_VEC3;
    case "vec2<f32>":   return PACKER_VEC2;
    case "f32":         return PACKER_F32;
    case "u32":         return PACKER_U32;
    case "vec2<u32>":   return PACKER_UVEC2;
    case "vec3<u32>":   return PACKER_UVEC3;
    case "vec4<u32>":   return PACKER_UVEC4;
    case "i32":         return PACKER_I32;
    case "vec2<i32>":   return PACKER_IVEC2;
    case "vec3<i32>":   return PACKER_IVEC3;
    case "vec4<i32>":   return PACKER_IVEC4;
    default:
      throw new Error(`heapScene: no JS-side packer for WGSL type '${wgslType}'`);
  }
}
