// CPU-side df32 helpers: pack/unpack and reference math.
//
// JS numbers are f64. To split an f64 into a (hi: f32, lo: f32) df32 pair
// we round `hi` to the nearest f32, then represent the remainder also in
// f32. As long as the remainder fits in f32 (true for all "f64 with a
// reasonable f32 magnitude"), the pair is exact: hi + lo = original f64.

export function f64ToDf32(x: number): [number, number] {
  const hi = Math.fround(x);
  const lo = Math.fround(x - hi);
  return [hi, lo];
}

export function df32ToF64(hi: number, lo: number): number {
  // Same arithmetic the kernel performs at f32; we just do it at f64 to
  // recover the value the kernel meant to represent.
  return hi + lo;
}

// 4×4 matrix layout in JS: row-major Float64Array(16), m[r*4 + c].
// On the GPU we store column-major vec2<f32>(hi, lo); helper functions
// translate at the buffer boundary.

export function packDf32Mat4(rowMajorF64: Float64Array, out: Float32Array, offset: number): void {
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      const v = rowMajorF64[r * 4 + c]!;
      const hi = Math.fround(v);
      const lo = Math.fround(v - hi);
      const i = offset + (c * 4 + r) * 2;
      out[i]     = hi;
      out[i + 1] = lo;
    }
  }
}

export function unpackDf32Mat4(buf: Float32Array, offset: number): Float64Array {
  const m = new Float64Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      const i = offset + (c * 4 + r) * 2;
      m[r * 4 + c] = buf[i]! + buf[i + 1]!;
    }
  }
  return m;
}

// Reference f64 mat4×mat4 (row-major). C = A · B.
export function mat4MulF64(a: Float64Array, b: Float64Array): Float64Array {
  const c = new Float64Array(16);
  for (let r = 0; r < 4; r++) {
    for (let col = 0; col < 4; col++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[r * 4 + k]! * b[k * 4 + col]!;
      c[r * 4 + col] = s;
    }
  }
  return c;
}

// Naive f32 mat4×mat4 — what the renderer does today. Used as a baseline
// to show what df32 buys us.
export function mat4MulF32(a: Float64Array, b: Float64Array): Float64Array {
  const af = new Float32Array(16); for (let i = 0; i < 16; i++) af[i] = Math.fround(a[i]!);
  const bf = new Float32Array(16); for (let i = 0; i < 16; i++) bf[i] = Math.fround(b[i]!);
  const cf = new Float32Array(16);
  for (let r = 0; r < 4; r++) {
    for (let col = 0; col < 4; col++) {
      let s = Math.fround(0);
      for (let k = 0; k < 4; k++) {
        s = Math.fround(s + Math.fround(af[r * 4 + k]! * bf[k * 4 + col]!));
      }
      cf[r * 4 + col] = s;
    }
  }
  const out = new Float64Array(16);
  for (let i = 0; i < 16; i++) out[i] = cf[i]!;
  return out;
}
