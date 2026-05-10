// Pure-JS simulation of the WGSL kernel using Math.fround for every
// f32 op. If this achieves f64-level precision, the algorithm is right
// and the loss is purely compiler-side. If it gives f32-level errors,
// there's a bug in the algorithm itself.

const f = Math.fround;

// Single-rounded fma: compute a*b+c in f64, then round to f32 once.
function fma(a: number, b: number, c: number): number {
  return f(a * b + c);
}

// Knuth's two_sum — captures err via two sub-cancellations, NOT via
// `b + (a - s)` which loses precision when |err| << ulp(b).
function two_sum(a: number, b: number): [number, number] {
  const s  = f(a + b);
  const bb = f(s - a);
  const t1 = f(s - bb);
  const t2 = f(a - t1);
  const t3 = f(b - bb);
  const err = f(t2 + t3);
  return [s, err];
}

function quick_two_sum(a: number, b: number): [number, number] {
  const s   = f(a + b);
  const t   = f(s - a);
  const err = f(b - t);
  return [s, err];
}

function two_prod(a: number, b: number): [number, number] {
  const p   = fma(a, b, 0);
  const err = fma(a, b, -p);
  return [p, err];
}

function df_add(a: [number, number], b: [number, number]): [number, number] {
  const [sH, sL] = two_sum(a[0], b[0]);
  const [tH, tL] = two_sum(a[1], b[1]);
  const m1 = f(sL + tH);
  const [s3H, s3L] = quick_two_sum(sH, m1);
  const m2 = f(s3L + tL);
  return quick_two_sum(s3H, m2);
}

function df_mul(a: [number, number], b: [number, number]): [number, number] {
  const [pH, pL] = two_prod(a[0], b[0]);
  const c1 = fma(a[0], b[1], pL);
  const c  = fma(a[1], b[0], c1);
  return quick_two_sum(pH, c);
}

export function df32MatMulSim(A: Float32Array, B: Float32Array): Float32Array {
  // A, B are 32-float buffers (16 entries × (hi, lo)) column-major.
  const C = new Float32Array(32);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let acc: [number, number] = [0, 0];
      for (let k = 0; k < 4; k++) {
        const aIdx = (k * 4 + r) * 2;
        const bIdx = (c * 4 + k) * 2;
        const ai: [number, number] = [A[aIdx]!, A[aIdx + 1]!];
        const bi: [number, number] = [B[bIdx]!, B[bIdx + 1]!];
        const prod = df_mul(ai, bi);
        acc = df_add(acc, prod);
      }
      const oIdx = (c * 4 + r) * 2;
      C[oIdx]     = acc[0];
      C[oIdx + 1] = acc[1];
    }
  }
  return C;
}
