/* SPDX-License-Identifier: MIT
 * base_q4_gemv — fused BaseQ4 GEMV for POWER8 VSX (dequant-in-registers)
 *
 * Elyan Labs — Phase 3 of the .base-on-POWER8 plan.
 *
 * y[m] = sum_k W[m,k] * x[k],  W stored base_q4 (asymmetric, gs=64):
 *   W = q*scale + bias  =>  row dot = sum_g ( scale_g * sum_i q_i*x_i
 *                                           + bias_g  * sum_i x_i )
 * The activation group-sums (sum_i x_i per group) are computed ONCE per
 * GEMV and shared across all rows — the bias term becomes ~free.
 * Weights are never materialized as f32 in memory.
 *
 * Build (POWER8): gcc -O3 -mcpu=power8 -maltivec -mvsx -fopenmp \
 *                     -o base_q4_gemv base_q4_gemv.c -lm
 * Usage:
 *   ./base_q4_gemv selftest
 *   ./base_q4_gemv probe model.base     # real lm_head rows vs reference
 *   ./base_q4_gemv bench [rows] [cols]  # default 8192 x 4096
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef NO_VSX
#include <altivec.h>
#endif

#define GS 64  /* base_q4 canonical group size */

static float bf16_to_f32(uint16_t b){uint32_t u=(uint32_t)b<<16;float f;memcpy(&f,&u,4);return f;}
static float f16_to_f32(uint16_t h){
    uint32_t s=(uint32_t)(h&0x8000)<<16,e=(h>>10)&0x1F,m=h&0x3FF,b;
    if(e==0){ if(m==0)b=s; else{e=113;while(!(m&0x400)){m<<=1;e--;}m&=0x3FF;b=s|(e<<23)|(m<<13);} }
    else if(e==31)b=s|0x7F800000|(m<<13);
    else b=s|((e+112)<<23)|(m<<13);
    float f;memcpy(&f,&b,4);return f;
}
static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

/* ---------- reference: dequant row then dot ---------- */
static void gemv_ref(const uint8_t *W, const float *sc, const float *bi,
                     const float *x, float *y, size_t M, size_t K) {
    size_t gpr = K / GS;                 /* groups per row */
    for (size_t m = 0; m < M; m++) {
        const uint8_t *row = W + m * (K / 2);
        double acc = 0;
        for (size_t g = 0; g < gpr; g++) {
            float s = sc[m * gpr + g], b = bi[m * gpr + g];
            const uint8_t *p = row + g * (GS / 2);
            const float *xg = x + g * GS;
            for (int i = 0; i < GS / 2; i++) {
                acc += ((float)(p[i] & 0xF) * s + b) * xg[2*i];
                acc += ((float)(p[i] >> 4)  * s + b) * xg[2*i+1];
            }
        }
        y[m] = (float)acc;
    }
}

/* ---------- fused scalar (algebraic split, group-sum trick) ---------- */
static void xgroupsums(const float *x, size_t K, float *xsum) {
    for (size_t g = 0; g < K / GS; g++) {
        float s = 0;
        for (int i = 0; i < GS; i++) s += x[g * GS + i];
        xsum[g] = s;
    }
}
static void gemv_fused_scalar(const uint8_t *W, const float *sc, const float *bi,
                              const float *x, const float *xsum, float *y,
                              size_t M, size_t K) {
    size_t gpr = K / GS;
    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; m++) {
        const uint8_t *row = W + m * (K / 2);
        float acc = 0;
        for (size_t g = 0; g < gpr; g++) {
            const uint8_t *p = row + g * (GS / 2);
            const float *xg = x + g * GS;
            float qdot = 0;
            for (int i = 0; i < GS / 2; i++) {
                qdot += (float)(p[i] & 0xF) * xg[2*i]
                      + (float)(p[i] >> 4)  * xg[2*i+1];
            }
            acc += sc[m * gpr + g] * qdot + bi[m * gpr + g] * xsum[g];
        }
        y[m] = acc;
    }
}

#ifndef NO_VSX
/* ---------- fused VSX: unpack nibbles in-register, vmadd with x ---------- */
static void gemv_fused_vsx(const uint8_t *W, const float *sc, const float *bi,
                           const float *x, const float *xsum, float *y,
                           size_t M, size_t K) {
    size_t gpr = K / GS;
    const vector unsigned char vmask = vec_splats((unsigned char)0x0F);
    const vector unsigned char vzero = vec_splats((unsigned char)0);
    const vector unsigned char v4    = vec_splats((unsigned char)4);
    const vector unsigned short z16  = vec_splats((unsigned short)0);
    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; m++) {
        const uint8_t *row = W + m * (K / 2);
        float acc = 0;
        for (size_t g = 0; g < gpr; g++) {
            const uint8_t *p = row + g * (GS / 2);
            const float *xg = x + g * GS;
            vector float vacc = vec_splats(0.0f);
            for (int half = 0; half < 2; half++) {     /* 16 bytes -> 32 vals */
                vector unsigned char v  = vec_xl(half * 16, p);
                vector unsigned char lo = vec_and(v, vmask);
                vector unsigned char hi = vec_sr(v, v4);
                vector unsigned char q01 = vec_mergeh(lo, hi);
                vector unsigned char q23 = vec_mergel(lo, hi);
                vector unsigned short u16[4];
                u16[0] = (vector unsigned short)vec_mergeh(q01, vzero);
                u16[1] = (vector unsigned short)vec_mergel(q01, vzero);
                u16[2] = (vector unsigned short)vec_mergeh(q23, vzero);
                u16[3] = (vector unsigned short)vec_mergel(q23, vzero);
                const float *xb = xg + half * 32;
                for (int k = 0; k < 4; k++) {
                    vector unsigned int a = (vector unsigned int)vec_mergeh(u16[k], z16);
                    vector unsigned int b = (vector unsigned int)vec_mergel(u16[k], z16);
                    vector float qa = vec_ctf(a, 0), qb = vec_ctf(b, 0);
                    vacc = vec_madd(qa, vec_xl(k * 32 + 0,  xb), vacc);
                    vacc = vec_madd(qb, vec_xl(k * 32 + 16, xb), vacc);
                }
            }
            float qdot = vacc[0] + vacc[1] + vacc[2] + vacc[3];
            acc += sc[m * gpr + g] * qdot + bi[m * gpr + g] * xsum[g];
        }
        y[m] = acc;
    }
}
#endif

/* ---------- utilities ---------- */
static float max_rel_err(const float *a, const float *b, size_t n) {
    float worst = 0;
    for (size_t i = 0; i < n; i++) {
        float denom = fabsf(a[i]) > 1e-3f ? fabsf(a[i]) : 1e-3f;
        float e = fabsf(a[i] - b[i]) / denom;
        if (e > worst) worst = e;
    }
    return worst;
}
static void rand_fill_q4(uint8_t *W, float *sc, float *bi, size_t M, size_t K) {
    for (size_t i = 0; i < M * K / 2; i++) W[i] = rand() & 0xFF;
    for (size_t i = 0; i < M * (K / GS); i++) {
        sc[i] = 0.005f + 0.01f * ((float)rand() / RAND_MAX);
        bi[i] = -0.08f + 0.16f * ((float)rand() / RAND_MAX);
    }
}

static int selftest(void) {
    const size_t M = 256, K = 1024;
    uint8_t *W = malloc(M * K / 2);
    float *sc = malloc(M * (K/GS) * 4), *bi = malloc(M * (K/GS) * 4);
    float *x = malloc(K * 4), *xsum = malloc((K/GS) * 4);
    float *yr = malloc(M * 4), *yf = malloc(M * 4), *yv = malloc(M * 4);
    srand(42);
    rand_fill_q4(W, sc, bi, M, K);
    for (size_t i = 0; i < K; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    xgroupsums(x, K, xsum);

    gemv_ref(W, sc, bi, x, yr, M, K);
    gemv_fused_scalar(W, sc, bi, x, xsum, yf, M, K);
    float e1 = max_rel_err(yr, yf, M);
    printf("fused-scalar vs ref: max_rel_err = %.3e\n", e1);
#ifndef NO_VSX
    gemv_fused_vsx(W, sc, bi, x, xsum, yv, M, K);
    float e2 = max_rel_err(yr, yv, M);
    printf("fused-vsx    vs ref: max_rel_err = %.3e\n", e2);
    if (e2 > 1e-4f) { printf("FAIL\n"); return 1; }
#endif
    if (e1 > 1e-4f) { printf("FAIL\n"); return 1; }
    printf("SELFTEST PASS\n");
    return 0;
}

/* ---------- probe real lm_head: logits = lm_head . x ----------
 * NOTE: header scale/bias offsets are TENSOR-relative; lm_head is at
 * tensor offset 0 so they coincide with blob-relative here ONLY. */
static int probe(const char *path) {
    const uint64_t BLOB = 0x540000, W_OFF = 0, S_OFF = 77791232, B_OFF = 82653184;
    const size_t K = 1024, M = 4096;      /* first 4096 vocab rows */
    const size_t gpr = K / GS;
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    uint8_t *W = malloc(M * K / 2);
    uint8_t *sraw = malloc(M * gpr * 2), *braw = malloc(M * gpr * 2);
    fseek(f, BLOB + W_OFF, SEEK_SET); if (fread(W, 1, M*K/2, f) != M*K/2) return 1;
    fseek(f, BLOB + S_OFF, SEEK_SET); if (fread(sraw, 1, M*gpr*2, f) != M*gpr*2) return 1;
    fseek(f, BLOB + B_OFF, SEEK_SET); if (fread(braw, 1, M*gpr*2, f) != M*gpr*2) return 1;
    fclose(f);
    float *sc = malloc(M * gpr * 4), *bi = malloc(M * gpr * 4);
    for (size_t i = 0; i < M * gpr; i++) {
        uint16_t su = (uint16_t)sraw[2*i] | ((uint16_t)sraw[2*i+1] << 8);
        uint16_t bu = (uint16_t)braw[2*i] | ((uint16_t)braw[2*i+1] << 8);
        sc[i] = bf16_to_f32(su); bi[i] = bf16_to_f32(bu);
    }
    float *x = malloc(K * 4), *xsum = malloc(gpr * 4);
    srand(1234);
    for (size_t i = 0; i < K; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f);
    xgroupsums(x, K, xsum);
    float *yr = malloc(M * 4), *yv = malloc(M * 4);
    gemv_ref(W, sc, bi, x, yr, M, K);
#ifndef NO_VSX
    gemv_fused_vsx(W, sc, bi, x, xsum, yv, M, K);
    printf("real lm_head rows: max_rel_err = %.3e\n", max_rel_err(yr, yv, M));
#endif
    printf("logits[0..5]: ");
    for (int i = 0; i < 6; i++) printf("% .4f ", yr[i]);
    printf("\n");
    return 0;
}

static int bench(size_t M, size_t K) {
    size_t gpr = K / GS;
    uint8_t *W = malloc(M * K / 2);
    float *sc = malloc(M * gpr * 4), *bi = malloc(M * gpr * 4);
    float *x = malloc(K * 4), *xsum = malloc(gpr * 4);
    float *y = malloc(M * 4);
    srand(7);
    rand_fill_q4(W, sc, bi, M, K);
    for (size_t i = 0; i < K; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f);
    xgroupsums(x, K, xsum);

    double bytes = (double)M * K / 2 + (double)M * gpr * 8;  /* packed + sc/bi */
    double flops = 2.0 * M * K;
    int reps = 20;
#ifdef _OPENMP
    printf("threads: %d\n", omp_get_max_threads());
#endif
    gemv_fused_scalar(W, sc, bi, x, xsum, y, M, K);           /* warm */
    double t0 = now_s();
    for (int r = 0; r < reps; r++) gemv_fused_scalar(W, sc, bi, x, xsum, y, M, K);
    double ts = (now_s() - t0) / reps;
    printf("fused-scalar: %7.2f GB/s  %7.2f GFLOP/s  (%.3f ms)\n",
           bytes/ts/1e9, flops/ts/1e9, ts*1e3);
#ifndef NO_VSX
    gemv_fused_vsx(W, sc, bi, x, xsum, y, M, K);              /* warm */
    t0 = now_s();
    for (int r = 0; r < reps; r++) gemv_fused_vsx(W, sc, bi, x, xsum, y, M, K);
    double tv = (now_s() - t0) / reps;
    printf("fused-vsx:    %7.2f GB/s  %7.2f GFLOP/s  (%.3f ms)  speedup %.2fx\n",
           bytes/tv/1e9, flops/tv/1e9, tv*1e3, ts/tv);
#endif
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "selftest")) return selftest();
    if (argc >= 3 && !strcmp(argv[1], "probe"))    return probe(argv[2]);
    if (argc >= 2 && !strcmp(argv[1], "bench"))
        return bench(argc >= 3 ? atoll(argv[2]) : 8192,
                     argc >= 4 ? atoll(argv[3]) : 4096);
    fprintf(stderr, "usage: %s selftest | probe model.base | bench [M] [K]\n", argv[0]);
    return 2;
}
