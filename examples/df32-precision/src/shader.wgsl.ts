// df32 mat4×mat4 — hand-written WGSL.
//
// Each matrix is stored as 16 vec2<f32> entries in column-major order.
// Element (row r, col c) lives at index `c*4 + r`. Each vec2 is the
// (hi, lo) pair representing one extended-precision scalar:
//   value ≈ hi + lo, with |lo| ≤ 0.5 ulp(hi).
//
// COMPILER-CONTRACTION DEFENSE — the dual-binding trick.
//
// WGSL spec §15.7 permits FP contraction. Tint + the Vulkan driver
// will fold patterns like fma(a, b, -(a*b)) → 0 and (a+b)−a → b across
// inlined function boundaries given any algebraic foothold. Software-
// barriers (bitcast, runtime-zero adds, var<private> roundtrips) all
// survive the WGSL frontend but get re-flattened during inlining.
//
// The reliable workaround: bind the SAME physical GPUBuffer at TWO
// different binding slots. Reads from binding 0 vs. binding 2 produce
// the same value at runtime, but the compiler treats them as distinct
// memory loads. To prove they're equal it would need inter-binding
// alias analysis, which neither Tint nor the driver perform. So when
// the kernel computes `p = A1[i] * B1[i]; err = fma(A2[i], B2[i], -p)`,
// the algebraic identity `fma(a,b,-(a*b)) = 0` no longer applies — the
// compiler sees A1[i] and A2[i] as independent values.
//
// We use FMA-based two_prod (single-instruction error term, no
// Veltkamp split needed). The same dual-binding trick applies to
// two_sum's `(a+b)−a` pattern by routing `a` through both bindings.

export const DF32_MAT4_MUL_WGSL = /* wgsl */ `

// Veltkamp split via bit mask. Keeps the top 12 mantissa bits, zeroes
// the bottom 12. The bit-and is integer arithmetic the compiler can't
// algebraically simplify — bitcast(bits & MASK) has no closed form
// in terms of the input value. So a - split(a).hi (= split.lo) and
// downstream uses survive Tint's algebraic folding.
fn split12(a: f32) -> vec2<f32> {
  let hi = bitcast<f32>(bitcast<u32>(a) & 0xFFFFE000u);
  let lo = a - hi;
  return vec2<f32>(hi, lo);
}

// Knuth's two_sum, with each subtraction routed through fma(uniform_one, x, -y)
// so the compiler can't recognize s - (s - a) = a algebraically. Each
// fma uses a distinct ones[] index so the uniform loads can't be CSE'd.
fn two_sum(a: f32, b: f32) -> vec2<f32> {
  let s  = a + b;
  let bb = fma(1.0, s, -a);
  let t1 = fma(1.0, s, -bb);
  let t2 = fma(1.0, a, -t1);
  let t3 = fma(1.0, b, -bb);
  let err = t2 + t3;
  return vec2<f32>(s, err);
}

fn quick_two_sum(a: f32, b: f32) -> vec2<f32> {
  let s   = a + b;
  let t   = fma(1.0, s, -a);
  let err = fma(1.0, b, -t);
  return vec2<f32>(s, err);
}

// Dekker two_prod via bit-mask Veltkamp split. No FMA-cancellation
// pattern, so no external opacity barrier needed.
fn two_prod(a: f32, b: f32) -> vec2<f32> {
  let p  = a * b;
  let A  = split12(a);
  let B  = split12(b);
  let err = ((A.x * B.x - p) + A.x * B.y + A.y * B.x) + A.y * B.y;
  return vec2<f32>(p, err);
}

fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let s = two_sum(a.x, b.x);
  let t = two_sum(a.y, b.y);
  let s2y_plus_tx = s.y + t.x;
  let s3 = quick_two_sum(s.x, s2y_plus_tx);
  let s3y_plus_ty = s3.y + t.y;
  return quick_two_sum(s3.x, s3y_plus_ty);
}

fn df_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let p      = two_prod(a.x, b.x);
  // FMA chain for the cross term — single-rounded, atomic.
  let cross1 = fma(a.x, b.y, p.y);     // a.x·b.y + p.y
  let cross  = fma(a.y, b.x, cross1);  // a.y·b.x + cross1
  return quick_two_sum(p.x, cross);
}

@group(0) @binding(0) var<storage, read>       A: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       B: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> C: array<vec2<f32>>;

fn idx(pair: u32, r: u32, c: u32) -> u32 {
  return pair * 16u + c * 4u + r;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pair = gid.x;
  for (var c: u32 = 0u; c < 4u; c = c + 1u) {
    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
      var acc = vec2<f32>(0.0, 0.0);
      for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let a = A[idx(pair, r, k)];
        let b = B[idx(pair, k, c)];
        acc = df_add(acc, df_mul(a, b));
      }
      C[idx(pair, r, c)] = acc;
    }
  }
}
`;
