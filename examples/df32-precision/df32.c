// df32 mat4×mat4 — C reference implementation.
//
// Same math as the WGSL kernel. Uses `volatile float` and
// `#pragma STDC FP_CONTRACT OFF` to prevent the compiler from
// algebraically simplifying away the precision-recovery operations.
//
// Build:
//   cc -O2 -fno-fast-math df32.c -o df32 -lm
// Run:
//   ./df32

#pragma STDC FP_CONTRACT OFF

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ─── df32 primitives ──────────────────────────────────────────────────
// `volatile` on each computed intermediate denies the compiler the
// ability to fold (a+b)−a → b. The volatile qualifier forces the
// store/load roundtrip to be observable.

static inline void two_sum(float a, float b, float* s_out, float* e_out) {
    volatile float s  = a + b;
    volatile float bb = s - a;
    volatile float t1 = s - bb;
    volatile float t2 = a - t1;
    volatile float t3 = b - bb;
    *s_out = s;
    *e_out = t2 + t3;
}

static inline void quick_two_sum(float a, float b, float* s_out, float* e_out) {
    volatile float s  = a + b;
    volatile float t  = s - a;
    *s_out = s;
    *e_out = b - t;
}

static inline void two_prod(float a, float b, float* p_out, float* e_out) {
    volatile float p = a * b;
    // FMA-based exact error term. fmaf is single-rounding per C99.
    *p_out = p;
    *e_out = fmaf(a, b, -p);
}

// df + df → df
static inline void df_add(float a_hi, float a_lo, float b_hi, float b_lo,
                          float* r_hi, float* r_lo) {
    float s_hi, s_lo, t_hi, t_lo;
    two_sum(a_hi, b_hi, &s_hi, &s_lo);
    two_sum(a_lo, b_lo, &t_hi, &t_lo);
    volatile float s2y_plus_tx = s_lo + t_hi;
    float s3_hi, s3_lo;
    quick_two_sum(s_hi, s2y_plus_tx, &s3_hi, &s3_lo);
    volatile float s3y_plus_ty = s3_lo + t_lo;
    quick_two_sum(s3_hi, s3y_plus_ty, r_hi, r_lo);
}

// df * df → df
static inline void df_mul(float a_hi, float a_lo, float b_hi, float b_lo,
                          float* r_hi, float* r_lo) {
    float p_hi, p_lo;
    two_prod(a_hi, b_hi, &p_hi, &p_lo);
    volatile float cross = a_hi * b_lo + a_lo * b_hi + p_lo;
    quick_two_sum(p_hi, cross, r_hi, r_lo);
}

// ─── Pack / unpack ────────────────────────────────────────────────────
// Matrices are 4×4 f64 row-major in source. df32 storage holds 16
// (hi, lo) pairs in column-major order.

static void pack_df32(const double* src, float* out) {
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            double v = src[r * 4 + c];
            float hi = (float)v;
            float lo = (float)(v - (double)hi);
            int i = (c * 4 + r) * 2;
            out[i]     = hi;
            out[i + 1] = lo;
        }
    }
}

static void unpack_df32(const float* buf, double* out) {
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            int i = (c * 4 + r) * 2;
            out[r * 4 + c] = (double)buf[i] + (double)buf[i + 1];
        }
    }
}

// ─── Mat-mul kernels ──────────────────────────────────────────────────

// df32 mat-mul (column-major buf, but indexing matches WGSL kernel).
static void df32_mat4_mul(const float* A, const float* B, float* C) {
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            float acc_hi = 0.0f, acc_lo = 0.0f;
            for (int k = 0; k < 4; k++) {
                int aIdx = (k * 4 + r) * 2;
                int bIdx = (c * 4 + k) * 2;
                float prod_hi, prod_lo;
                df_mul(A[aIdx], A[aIdx + 1],
                       B[bIdx], B[bIdx + 1],
                       &prod_hi, &prod_lo);
                float new_hi, new_lo;
                df_add(acc_hi, acc_lo, prod_hi, prod_lo, &new_hi, &new_lo);
                acc_hi = new_hi;
                acc_lo = new_lo;
            }
            int outIdx = (c * 4 + r) * 2;
            C[outIdx]     = acc_hi;
            C[outIdx + 1] = acc_lo;
        }
    }
}

// f64 reference (row-major).
static void f64_mat4_mul(const double* A, const double* B, double* C) {
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            double s = 0.0;
            for (int k = 0; k < 4; k++) {
                s += A[r * 4 + k] * B[k * 4 + c];
            }
            C[r * 4 + c] = s;
        }
    }
}

// Naive f32 mat-mul (what the GPU does without df32).
static void f32_mat4_mul(const double* A, const double* B, double* C) {
    float Af[16], Bf[16], Cf[16];
    for (int i = 0; i < 16; i++) Af[i] = (float)A[i];
    for (int i = 0; i < 16; i++) Bf[i] = (float)B[i];
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            volatile float s = 0.0f;
            for (int k = 0; k < 4; k++) {
                s = s + Af[r * 4 + k] * Bf[k * 4 + c];
            }
            Cf[r * 4 + c] = s;
        }
    }
    for (int i = 0; i < 16; i++) C[i] = (double)Cf[i];
}

// ─── Test harness ─────────────────────────────────────────────────────

static uint32_t rng_state = 0xa110ce;
static double rnd(double lo, double hi) {
    rng_state = rng_state * 1664525u + 1013904223u;
    double u = (double)rng_state / (double)0x100000000ull;
    return lo + (hi - lo) * u;
}

static void random_mat(double scale, double* m) {
    for (int i = 0; i < 16; i++) m[i] = rnd(-scale, scale);
}

static void identity(double* m) {
    memset(m, 0, 16 * sizeof(double));
    m[0] = m[5] = m[10] = m[15] = 1.0;
}

static void translation(double tx, double ty, double tz, double* m) {
    memset(m, 0, 16 * sizeof(double));
    m[0] = m[5] = m[10] = m[15] = 1.0;
    m[3] = tx; m[7] = ty; m[11] = tz;
}

static void rot_y(double th, double* m) {
    double c = cos(th), s = sin(th);
    memset(m, 0, 16 * sizeof(double));
    m[0] = c;  m[2] = s;
    m[5] = 1.0;
    m[8] = -s; m[10] = c;
    m[15] = 1.0;
}

static void mul_rm(const double* a, const double* b, double* c) {
    for (int r = 0; r < 4; r++) {
        for (int col = 0; col < 4; col++) {
            double s = 0.0;
            for (int k = 0; k < 4; k++) s += a[r * 4 + k] * b[k * 4 + col];
            c[r * 4 + col] = s;
        }
    }
}

typedef struct {
    const char* name;
    int n;
    double pairs[64][2][16];   // n × (A, B) f64 row-major
} Case;

static void run_case(const Case* c) {
    double df32_max_abs = 0, df32_max_rel = 0, df32_sum_rel = 0;
    double f32_max_abs  = 0, f32_max_rel  = 0, f32_sum_rel  = 0;
    int count = 0;

    for (int i = 0; i < c->n; i++) {
        const double* A = c->pairs[i][0];
        const double* B = c->pairs[i][1];
        double truth[16], f32_result[16];
        f64_mat4_mul(A, B, truth);
        f32_mat4_mul(A, B, f32_result);

        // df32 path: pack, multiply in df32, unpack.
        float Adf[32], Bdf[32], Cdf[32];
        pack_df32(A, Adf);
        pack_df32(B, Bdf);
        df32_mat4_mul(Adf, Bdf, Cdf);
        double df32_result[16];
        unpack_df32(Cdf, df32_result);

        for (int k = 0; k < 16; k++) {
            double t = truth[k];
            double dD = fabs(df32_result[k] - t);
            double dF = fabs(f32_result[k]  - t);
            double denom = fabs(t) > 1e-30 ? fabs(t) : 1e-30;
            if (dD > df32_max_abs) df32_max_abs = dD;
            if (dF > f32_max_abs)  f32_max_abs  = dF;
            double rD = dD / denom, rF = dF / denom;
            if (rD > df32_max_rel) df32_max_rel = rD;
            if (rF > f32_max_rel)  f32_max_rel  = rF;
            df32_sum_rel += rD;
            f32_sum_rel  += rF;
            count++;
        }
    }
    printf("\n%-12s pairs=%d  scalars=%d\n", c->name, c->n, count);
    printf("  df32 (C)   max abs=%-10.3e  max rel=%-10.3e  mean rel=%-10.3e\n",
           df32_max_abs, df32_max_rel, df32_sum_rel / count);
    printf("  f32 (C)    max abs=%-10.3e  max rel=%-10.3e  mean rel=%-10.3e\n",
           f32_max_abs, f32_max_rel, f32_sum_rel / count);
}

int main(void) {
    static Case sanity = { .name = "sanity", .n = 32 };
    rng_state = 0xa110ce;
    for (int i = 0; i < sanity.n; i++) {
        identity((double*)sanity.pairs[i][0]);
        random_mat(1.0, (double*)sanity.pairs[i][1]);
    }

    static Case small = { .name = "small", .n = 64 };
    rng_state = 0x5ba11;
    for (int i = 0; i < small.n; i++) {
        random_mat(1.0, (double*)small.pairs[i][0]);
        random_mat(1.0, (double*)small.pairs[i][1]);
    }

    static Case geo = { .name = "geodetic", .n = 32 };
    rng_state = 0xea27e;
    {
        const double R = 6.378137e6;
        double deltas[4] = {1e3, 1.0, 1e-3, 1e-6};
        int idx = 0;
        for (int d = 0; d < 4; d++) {
            for (int i = 0; i < 8 && idx < geo.n; i++, idx++) {
                double r1[16], r2[16], tF[16], tB[16], M[16], V[16];
                rot_y(rnd(0, 2.0 * M_PI), r1);
                rot_y(rnd(0, 2.0 * M_PI), r2);
                translation(R + deltas[d], rnd(-10, 10), rnd(-10, 10), tF);
                translation(-R, 0, 0, tB);
                mul_rm(r1, tF, M);
                mul_rm(tB, r2, V);
                memcpy(geo.pairs[idx][0], M, 16 * sizeof(double));
                memcpy(geo.pairs[idx][1], V, 16 * sizeof(double));
            }
        }
    }

    static Case composed = { .name = "composed", .n = 16 };
    rng_state = 0xc0490;
    for (int i = 0; i < composed.n; i++) {
        double A[16], B[16], R[16], tmp[16];
        rot_y(rnd(0, 2.0 * M_PI), A);
        rot_y(rnd(0, 2.0 * M_PI), B);
        for (int k = 0; k < 50; k++) {
            rot_y(rnd(0, 0.1), R);
            mul_rm(A, R, tmp); memcpy(A, tmp, sizeof(A));
            rot_y(rnd(0, 0.1), R);
            mul_rm(B, R, tmp); memcpy(B, tmp, sizeof(B));
        }
        memcpy(composed.pairs[i][0], A, 16 * sizeof(double));
        memcpy(composed.pairs[i][1], B, 16 * sizeof(double));
    }

    printf("df32 mat4×mat4 — C reference (volatile + FP_CONTRACT OFF)\n");
    run_case(&sanity);
    run_case(&small);
    run_case(&geo);
    run_case(&composed);
    return 0;
}
