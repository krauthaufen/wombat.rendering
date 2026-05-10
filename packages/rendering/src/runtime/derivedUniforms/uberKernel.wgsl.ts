// §7 derived-uniforms uber kernel — single flat dispatch over leaves.
//
// Records are static-ish (only edited on RO add/remove). Each thread
// processes one record:
//   1. Load record (recipe_id, in0, in1, in2, out_byte).
//   2. Switch on recipe_id, compute inline (df_mul + collapse), write
//      f32 mat4/mat3 to MainHeap at out_byte.
//
// No dirty test on the kernel: when constituents don't change, the
// kernel rewrites identical bits. Saves a CPU writeBuffer + a binding
// per dispatch. GPU does ~64 FLOPs per record × N records — trivial at
// the scales this runs at.
//
// df32 primitives mirror examples/df32-precision/. Two opacity hooks:
//   - bitcast<f32>(bits & MASK) for Veltkamp split (integer ops, no
//     algebraic identity).
//   - fma(1.0, x, -y) for compensated subtractions.

export const DERIVED_UBER_KERNEL_WGSL = /* wgsl */ `

// ─── Bindings ─────────────────────────────────────────────────────
@group(0) @binding(0) var<storage, read>       Constituents: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> MainHeap:     array<f32>;

struct Record {
  recipe_id: u32,
  in0:       u32,    // constituent slot index
  in1:       u32,    // constituent slot index, 0 if unused
  in2:       u32,    // constituent slot index, 0 if unused
  out_byte:  u32,    // byte offset in MainHeap
  _pad:      u32,    // bumps stride to 24 (defensively 16-aligned)
}
@group(0) @binding(2) var<storage, read>       Records: array<Record>;

// Live record count. arrayLength(Records) reports the GPU buffer's
// capacity (rounded up by allocator growth), not the live count —
// trailing slots hold stale data from swap-removed records. Threads
// past Count.count would read stale records and trample random
// MainHeap bytes.
struct CountUniform { count: u32 }
@group(0) @binding(3) var<uniform>             Count: CountUniform;

// ─── df32 primitives ──────────────────────────────────────────────

fn split12(a: f32) -> vec2<f32> {
  let hi = bitcast<f32>(bitcast<u32>(a) & 0xFFFFE000u);
  let lo = a - hi;
  return vec2<f32>(hi, lo);
}

fn two_sum(a: f32, b: f32) -> vec2<f32> {
  let s  = a + b;
  let bb = fma(1.0, s, -a);
  let t1 = fma(1.0, s, -bb);
  let t2 = fma(1.0, a, -t1);
  let t3 = fma(1.0, b, -bb);
  return vec2<f32>(s, t2 + t3);
}

fn quick_two_sum(a: f32, b: f32) -> vec2<f32> {
  let s = a + b;
  let t = fma(1.0, s, -a);
  return vec2<f32>(s, fma(1.0, b, -t));
}

fn two_prod(a: f32, b: f32) -> vec2<f32> {
  let p = a * b;
  let A = split12(a);
  let B = split12(b);
  let err = ((A.x * B.x - p) + A.x * B.y + A.y * B.x) + A.y * B.y;
  return vec2<f32>(p, err);
}

fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let s = two_sum(a.x, b.x);
  let t = two_sum(a.y, b.y);
  let s3 = quick_two_sum(s.x, s.y + t.x);
  return quick_two_sum(s3.x, s3.y + t.y);
}

fn df_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let p      = two_prod(a.x, b.x);
  let cross1 = fma(a.x, b.y, p.y);
  let cross  = fma(a.y, b.x, cross1);
  return quick_two_sum(p.x, cross);
}

// ─── Constituents access ──────────────────────────────────────────
//
// One slot = 16 vec2<f32> = one df32 mat4 in row-major order
// (matching Aardvark M44d convention). Element (row, col) of slot k
// lives at vec2 index k*16 + row*4 + col.

fn read_entry(slot: u32, row: u32, col: u32) -> vec2<f32> {
  return Constituents[slot * 16u + row * 4u + col];
}

// (A · B) entry (r, c) in df32, both A and B in constituents.
fn df_mul_entry(a_slot: u32, b_slot: u32, r: u32, c: u32) -> vec2<f32> {
  var acc = vec2<f32>(0.0, 0.0);
  for (var k: u32 = 0u; k < 4u; k = k + 1u) {
    acc = df_add(acc, df_mul(read_entry(a_slot, r, k), read_entry(b_slot, k, c)));
  }
  return acc;
}

// Compute one full row of (A · B), returns 4 df32 entries.
fn df_mul_row(a_slot: u32, b_slot: u32, r: u32) -> array<vec2<f32>, 4> {
  var out: array<vec2<f32>, 4>;
  for (var c: u32 = 0u; c < 4u; c = c + 1u) {
    out[c] = df_mul_entry(a_slot, b_slot, r, c);
  }
  return out;
}

// Triple product (A · B · C) entry (r, c). Uses 4 df_muls per entry
// — recomputes the inner row each call. Acceptable: at most one
// such recipe per RO and only on dirty frames.
fn df_mul3_entry(a_slot: u32, b_slot: u32, c_slot: u32, r: u32, c: u32) -> vec2<f32> {
  // (A·B·C)[r,c] = sum_k (A·B)[r,k] · C[k,c]
  var acc = vec2<f32>(0.0, 0.0);
  for (var k: u32 = 0u; k < 4u; k = k + 1u) {
    let ab_rk = df_mul_entry(a_slot, b_slot, r, k);
    acc = df_add(acc, df_mul(ab_rk, read_entry(c_slot, k, c)));
  }
  return acc;
}

// ─── Writers ──────────────────────────────────────────────────────
//
// Mat4: 16 contiguous f32s, row-major.
// Mat3: 3 rows × 4-stride (std140 mat3<f32>), 12 f32s; last column
//       per row left untouched.

fn write_mat4_entry(out_byte: u32, row: u32, col: u32, val: f32) {
  MainHeap[(out_byte >> 2u) + row * 4u + col] = val;
}

fn write_mat3_entry(out_byte: u32, row: u32, col: u32, val: f32) {
  MainHeap[(out_byte >> 2u) + row * 4u + col] = val;
}

// ─── Recipe arms ──────────────────────────────────────────────────

fn run_collapse_mat4(in0: u32, out_byte: u32) {
  for (var r: u32 = 0u; r < 4u; r = r + 1u) {
    for (var c: u32 = 0u; c < 4u; c = c + 1u) {
      let e = read_entry(in0, r, c);
      write_mat4_entry(out_byte, r, c, e.x + e.y);
    }
  }
}

// NormalMatrix: (M⁻¹)ᵀ upper-3×3.
//   NM[i, j] = MInv[j, i]
fn run_collapse_mat3_normal(in0: u32, out_byte: u32) {
  for (var i: u32 = 0u; i < 3u; i = i + 1u) {
    for (var j: u32 = 0u; j < 3u; j = j + 1u) {
      let e = read_entry(in0, j, i);
      write_mat3_entry(out_byte, i, j, e.x + e.y);
    }
  }
}

fn run_dfmul2_collapse(in0: u32, in1: u32, out_byte: u32) {
  for (var r: u32 = 0u; r < 4u; r = r + 1u) {
    for (var c: u32 = 0u; c < 4u; c = c + 1u) {
      let e = df_mul_entry(in0, in1, r, c);
      write_mat4_entry(out_byte, r, c, e.x + e.y);
    }
  }
}

fn run_dfmul3_collapse(in0: u32, in1: u32, in2: u32, out_byte: u32) {
  for (var r: u32 = 0u; r < 4u; r = r + 1u) {
    for (var c: u32 = 0u; c < 4u; c = c + 1u) {
      let e = df_mul3_entry(in0, in1, in2, r, c);
      write_mat4_entry(out_byte, r, c, e.x + e.y);
    }
  }
}

// ─── Entry ────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= Count.count) { return; }
  let r = Records[gid.x];
  switch r.recipe_id {
    case 0u: { run_collapse_mat4(r.in0, r.out_byte); }            // ModelTrafo
    case 1u: { run_collapse_mat4(r.in0, r.out_byte); }            // ModelTrafoInv
    case 2u: { run_collapse_mat3_normal(r.in0, r.out_byte); }     // NormalMatrix
    case 3u: { run_dfmul2_collapse(r.in0, r.in1, r.out_byte); }   // ModelViewTrafo
    case 4u: { run_dfmul2_collapse(r.in0, r.in1, r.out_byte); }   // ModelViewTrafoInv
    case 5u: { run_dfmul3_collapse(r.in0, r.in1, r.in2, r.out_byte); } // ModelViewProjTrafo
    case 6u: { run_dfmul3_collapse(r.in0, r.in1, r.in2, r.out_byte); } // ModelViewProjTrafoInv
    case 7u: { run_collapse_mat4(r.in0, r.out_byte); }            // ViewTrafo
    case 8u: { run_collapse_mat4(r.in0, r.out_byte); }            // ViewTrafoInv
    case 9u: { run_dfmul2_collapse(r.in0, r.in1, r.out_byte); }   // ViewProjTrafo
    case 10u:{ run_dfmul2_collapse(r.in0, r.in1, r.out_byte); }   // ViewProjTrafoInv
    case 11u:{ run_collapse_mat4(r.in0, r.out_byte); }            // ProjTrafo
    case 12u:{ run_collapse_mat4(r.in0, r.out_byte); }            // ProjTrafoInv
    default: { return; }
  }
}

`;

/** Records are 5× u32 = 20 bytes. WGSL storage struct alignment of
 *  4 means the array stride is 20 bytes. */
export const RECORD_STRIDE_BYTES = 24;
export const RECORD_STRIDE_U32 = 6;

export const RECORD_FIELD = {
  recipe_id: 0,
  in0:       1,
  in1:       2,
  in2:       3,
  out_byte:  4,
} as const;
