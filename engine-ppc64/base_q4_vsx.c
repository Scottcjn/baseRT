/* SPDX-License-Identifier: MIT
 * base_q4_vsx — BaseQ4 (.base) dequantization kernels for POWER8 VSX
 *
 * Elyan Labs — Phase 2 of the .base-on-POWER8 plan.
 *
 * Layout (from basecompute/baseRT base-quant/src/base_q4.rs):
 *   - group of 64 values along last dim
 *   - q in [0,15], x = q * scale + bias  (scale=(max-min)/15, bias=min)
 *   - two per byte, LOW NIBBLE FIRST: byte = (q[2i+1] << 4) | q[2i]
 *   - scales/biases: contiguous f16 (converter) or bf16 (catalog bundles),
 *     separate regions at scale_offset / bias_offset
 *   - scale & bias round-tripped through f16 at pack time, so dequant
 *     can match pack bit-for-bit
 *
 * Build (POWER8): gcc -O3 -mcpu=power8 -maltivec -mvsx -o base_q4_vsx base_q4_vsx.c
 * Build (x86 ref-only): gcc -O3 -DNO_VSX -o base_q4_vsx base_q4_vsx.c
 *
 * Usage:
 *   ./base_q4_vsx selftest          # pack/dequant roundtrip + VSX vs ref
 *   ./base_q4_vsx probe model.base  # dequant real lm_head rows from the bundle
 *   ./base_q4_vsx bench             # scalar vs VSX throughput
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef NO_VSX
#include <altivec.h>
#endif

#define GROUP_SIZE 64

/* ---------- f16 / bf16 decode (endian-explicit LE) ---------- */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) { bits = sign; }
        else { /* subnormal */
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}
static float bf16_to_f32(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f; memcpy(&f, &bits, 4); return f;
}
/* f32 -> f16, round-to-nearest-even (for the pack reference) */
static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man  = x & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000;
        uint32_t shift = 14 - exp;
        uint32_t half = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1);
        uint32_t mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1))) half++;
        return (uint16_t)(sign | half);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    uint32_t half = (uint32_t)(exp << 10) | (man >> 13);
    uint32_t rem = man & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) half++;
    return (uint16_t)(sign | half);
}

/* ---------- pack reference (port of base_q4.rs pack()) ---------- */
static void pack_ref(const float *w, size_t n, uint8_t *packed,
                     uint16_t *scales_f16, uint16_t *biases_f16) {
    size_t n_groups = n / GROUP_SIZE;
    memset(packed, 0, n / 2);
    for (size_t g = 0; g < n_groups; g++) {
        const float *grp = w + g * GROUP_SIZE;
        float mn = INFINITY, mx = -INFINITY;
        for (int i = 0; i < GROUP_SIZE; i++) {
            if (grp[i] < mn) mn = grp[i];
            if (grp[i] > mx) mx = grp[i];
        }
        float raw_scale = (mx - mn) / 15.0f;
        float scale_f32 = (raw_scale == 0.0f) ? 1.0f : raw_scale;
        uint16_t sh = f32_to_f16(scale_f32), bh = f32_to_f16(mn);
        float scale = f16_to_f32(sh), bias = f16_to_f32(bh);
        scales_f16[g] = sh; biases_f16[g] = bh;
        float inv = 1.0f / scale;
        for (int i = 0; i < GROUP_SIZE; i++) {
            float qf = roundf((grp[i] - bias) * inv);
            if (qf < 0) qf = 0; if (qf > 15) qf = 15;
            uint8_t q = (uint8_t)qf;
            size_t bi = (g * GROUP_SIZE + i) / 2;
            if (i % 2 == 0) packed[bi] = (packed[bi] & 0xF0) | q;
            else            packed[bi] = (packed[bi] & 0x0F) | (q << 4);
        }
    }
}

/* ---------- scalar dequant (port of base_q4.rs unpack()) ---------- */
static void dequant_ref(const uint8_t *packed, const float *scales,
                        const float *biases, float *out, size_t n) {
    size_t n_groups = n / GROUP_SIZE;
    for (size_t g = 0; g < n_groups; g++) {
        float scale = scales[g], bias = biases[g];
        const uint8_t *p = packed + g * (GROUP_SIZE / 2);
        float *o = out + g * GROUP_SIZE;
        for (int i = 0; i < GROUP_SIZE / 2; i++) {
            o[2*i]   = (float)(p[i] & 0x0F) * scale + bias;
            o[2*i+1] = (float)(p[i] >> 4)   * scale + bias;
        }
    }
}

#ifndef NO_VSX
/* ---------- VSX dequant ----------
 * Per group of 64: 32 packed bytes = two 16-byte vector loads.
 * Each 16-byte load expands to 32 f32 outputs:
 *   lo = v & 0xF   -> q[0],q[2],..,q[30] (even logical indices)
 *   hi = v >> 4    -> q[1],q[3],..,q[31] (odd logical indices)
 *   merge lo/hi pairwise to restore logical order, widen u8->u32,
 *   convert to f32, fma with splatted scale/bias.
 */
static void dequant_vsx(const uint8_t *packed, const float *scales,
                        const float *biases, float *out, size_t n) {
    const vector unsigned char vmask = vec_splats((unsigned char)0x0F);
    const vector unsigned char vzero = vec_splats((unsigned char)0);
    size_t n_groups = n / GROUP_SIZE;
    for (size_t g = 0; g < n_groups; g++) {
        vector float vs = vec_splats(scales[g]);
        vector float vb = vec_splats(biases[g]);
        const uint8_t *p = packed + g * (GROUP_SIZE / 2);
        float *o = out + g * GROUP_SIZE;
        for (int half = 0; half < 2; half++) {          /* 16 bytes -> 32 vals */
            vector unsigned char v = vec_xl(half * 16, p);
            vector unsigned char lo = vec_and(v, vmask);
            vector unsigned char hi = vec_sr(v, vec_splats((unsigned char)4));
            /* interleave nibbles back to logical order: q0,q1,q2,q3,... */
            vector unsigned char q01 = vec_mergeh(lo, hi); /* q0..q15  */
            vector unsigned char q23 = vec_mergel(lo, hi); /* q16..q31 */
            /* widen u8 -> u16 -> u32, convert, fma */
            vector unsigned short u16a = (vector unsigned short)vec_mergeh(q01, vzero);
            vector unsigned short u16b = (vector unsigned short)vec_mergel(q01, vzero);
            vector unsigned short u16c = (vector unsigned short)vec_mergeh(q23, vzero);
            vector unsigned short u16d = (vector unsigned short)vec_mergel(q23, vzero);
            vector unsigned short z16 = vec_splats((unsigned short)0);
            vector unsigned int u32[8];
            u32[0] = (vector unsigned int)vec_mergeh(u16a, z16);
            u32[1] = (vector unsigned int)vec_mergel(u16a, z16);
            u32[2] = (vector unsigned int)vec_mergeh(u16b, z16);
            u32[3] = (vector unsigned int)vec_mergel(u16b, z16);
            u32[4] = (vector unsigned int)vec_mergeh(u16c, z16);
            u32[5] = (vector unsigned int)vec_mergel(u16c, z16);
            u32[6] = (vector unsigned int)vec_mergeh(u16d, z16);
            u32[7] = (vector unsigned int)vec_mergel(u16d, z16);
            float *dst = o + half * 32;
            for (int k = 0; k < 8; k++) {
                vector float vf = vec_ctf(u32[k], 0);
                vec_xst(vec_madd(vf, vs, vb), k * 16, dst);
            }
        }
    }
}
#endif

/* ---------- helpers ---------- */
static void decode_scales(const uint8_t *raw, size_t n_groups, int is_bf16,
                          float *out) {
    for (size_t g = 0; g < n_groups; g++) {
        uint16_t u = (uint16_t)raw[2*g] | ((uint16_t)raw[2*g+1] << 8); /* LE */
        out[g] = is_bf16 ? bf16_to_f32(u) : f16_to_f32(u);
    }
}
static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmp_exact(const float *a, const float *b, size_t n, const char *tag) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("MISMATCH [%s] at %zu: %g vs %g\n", tag, i, a[i], b[i]);
            return 1;
        }
    }
    printf("OK [%s]: %zu values bit-identical\n", tag, n);
    return 0;
}

/* ---------- selftest ---------- */
static int selftest(void) {
    const size_t N = 64 * 1024;  /* 1024 groups */
    float *w = malloc(N * 4), *ref = malloc(N * 4), *vsx = malloc(N * 4);
    float *rt = malloc(N * 4);
    uint8_t *packed = malloc(N / 2);
    uint16_t *sh = malloc(N / GROUP_SIZE * 2), *bh = malloc(N / GROUP_SIZE * 2);
    float *sf = malloc(N / GROUP_SIZE * 4), *bf = malloc(N / GROUP_SIZE * 4);
    srand(42);
    for (size_t i = 0; i < N; i++)
        w[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;

    pack_ref(w, N, packed, sh, bh);
    for (size_t g = 0; g < N / GROUP_SIZE; g++) {
        sf[g] = f16_to_f32(sh[g]); bf[g] = f16_to_f32(bh[g]);
    }
    dequant_ref(packed, sf, bf, ref, N);

    /* roundtrip sanity: |x - dq(q(x))| <= scale/2 + f16 slack */
    double max_err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fabs(w[i] - ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("roundtrip max_err = %g (expect ~scale/2 ~ 0.14)\n", max_err);
    if (max_err > 0.3) { printf("FAIL: roundtrip error too large\n"); return 1; }

    memcpy(rt, ref, N * 4);
#ifndef NO_VSX
    dequant_vsx(packed, sf, bf, vsx, N);
    if (cmp_exact(ref, vsx, N, "vsx-vs-ref")) return 1;
#else
    printf("(VSX disabled in this build)\n");
#endif
    (void)rt; (void)vsx;
    printf("SELFTEST PASS\n");
    return 0;
}

/* ---------- probe: dequant real lm_head rows from model.base ----------
 * tensor[0] from the Qwen3-0.6B default-q4 bundle (Phase 1 recon):
 *   lm_head.weight [151936,1024] base_q4 gs=64 scale_dtype=bf16
 *   offset 0, scale_offset 77791232, bias_offset 82653184
 *   weights blob at file offset 0x540000
 * NOTE: scale/bias offsets in the header are TENSOR-relative. lm_head
 * sits at tensor offset 0, so tensor-relative == blob-relative here
 * ONLY. Do not copy these constants for any other tensor.
 */
static int probe(const char *path) {
    const uint64_t BLOB = 0x540000;
    const uint64_t W_OFF = 0, S_OFF = 77791232, B_OFF = 82653184;
    const size_t ROW = 1024, NROWS = 4;    /* dequant first 4 rows */
    const size_t NG = ROW * NROWS / GROUP_SIZE;

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    uint8_t *packed = malloc(ROW * NROWS / 2);
    uint8_t *sraw = malloc(NG * 2), *braw = malloc(NG * 2);
    fseek(f, BLOB + W_OFF, SEEK_SET);  fread(packed, 1, ROW * NROWS / 2, f);
    fseek(f, BLOB + S_OFF, SEEK_SET);  fread(sraw, 1, NG * 2, f);
    fseek(f, BLOB + B_OFF, SEEK_SET);  fread(braw, 1, NG * 2, f);
    fclose(f);

    float *sf = malloc(NG * 4), *bf = malloc(NG * 4);
    decode_scales(sraw, NG, 1 /* bf16 */, sf);
    decode_scales(braw, NG, 1, bf);

    float *ref = malloc(ROW * NROWS * 4), *vsx = malloc(ROW * NROWS * 4);
    dequant_ref(packed, sf, bf, ref, ROW * NROWS);
#ifndef NO_VSX
    dequant_vsx(packed, sf, bf, vsx, ROW * NROWS);
    if (cmp_exact(ref, vsx, ROW * NROWS, "lm_head-real-tensor")) return 1;
#endif
    printf("lm_head row0[0..7]: ");
    for (int i = 0; i < 8; i++) printf("% .5f ", ref[i]);
    printf("\ngroup0 scale=%g bias=%g\n", sf[0], bf[0]);
    double ss = 0; for (size_t i = 0; i < ROW; i++) ss += (double)ref[i]*ref[i];
    printf("row0 L2 norm = %.4f (sane embedding-ish magnitude expected)\n", sqrt(ss));
    return 0;
}

/* ---------- bench ---------- */
static int bench(void) {
    const size_t N = 64ull * 1024 * 1024;   /* 64M values = 32MB packed */
    const size_t NG = N / GROUP_SIZE;
    uint8_t *packed = malloc(N / 2);
    float *sf = malloc(NG * 4), *bf = malloc(NG * 4), *out = malloc(N * 4);
    srand(7);
    for (size_t i = 0; i < N / 2; i++) packed[i] = rand() & 0xFF;
    for (size_t g = 0; g < NG; g++) { sf[g] = 0.01f; bf[g] = -0.07f; }

    double t0 = now_s();
    dequant_ref(packed, sf, bf, out, N);
    double t1 = now_s();
    double scalar_s = t1 - t0;
    printf("scalar: %6.1f Mval/s (%.3fs)\n", N / scalar_s / 1e6, scalar_s);
#ifndef NO_VSX
    t0 = now_s();
    dequant_vsx(packed, sf, bf, out, N);
    t1 = now_s();
    double vsx_s = t1 - t0;
    printf("vsx:    %6.1f Mval/s (%.3fs)  speedup %.2fx\n",
           N / vsx_s / 1e6, vsx_s, scalar_s / vsx_s);
#endif
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "selftest")) return selftest();
    if (argc >= 3 && !strcmp(argv[1], "probe"))    return probe(argv[2]);
    if (argc >= 2 && !strcmp(argv[1], "bench"))    return bench();
    fprintf(stderr, "usage: %s selftest | probe model.base | bench\n", argv[0]);
    return 2;
}
