/* SPDX-License-Identifier: MIT
 * qwen3_base — minimal Qwen3 forward pass reading a .base bundle directly.
 *
 * Elyan Labs — Phase 3b of the .base-on-POWER8 plan: FIRST TOKEN.
 *
 * Arch (from bundle config): Qwen3-0.6B — hidden 1024, 28 layers,
 * 16 Q heads / 8 KV heads (GQA), head_dim 128, per-head q/k RMSNorm,
 * neox RoPE theta 1e6, SwiGLU ffn 3072, vocab 151936, tied embeddings
 * (embed f16 for lookup, lm_head base_q4 for logits).
 *
 * Weights stay q4-packed in the mmap; GEMVs dequant in-registers
 * (Phase 3 fused kernel). Scales/biases pre-decoded bf16->f32 at load.
 *
 * Build: gcc -O3 -mcpu=power8 -maltivec -mvsx -fopenmp -o qwen3_base \
 *            qwen3_base.c -lm
 * Run:   ./qwen3_base model.base 16 785 6722 315 9625 374
 *        (16 = max new tokens, rest = prompt ids; greedy decode)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#ifndef NO_VSX
#include <altivec.h>
#endif
#include "tensors.h"

#define HIDDEN   1024
#define NLAYERS  28
#define NQH      16
#define NKVH     8
#define HDIM     128
#define FFN      3072
#define VOCAB    151936
#define GS       64
#define MAXCTX   256
#define EPS      1e-6f
#define ROPE_THETA 1000000.0f

static const uint64_t BLOB = 0x540000;

/* ---------- decode helpers ---------- */
static float bf16f(uint16_t b){uint32_t u=(uint32_t)b<<16;float f;memcpy(&f,&u,4);return f;}
static float f16f(uint16_t h){
    uint32_t s=(uint32_t)(h&0x8000)<<16,e=(h>>10)&0x1F,m=h&0x3FF,b;
    if(e==0){ if(m==0)b=s; else{e=113;while(!(m&0x400)){m<<=1;e--;}m&=0x3FF;b=s|(e<<23)|(m<<13);} }
    else if(e==31)b=s|0x7F800000|(m<<13);
    else b=s|((e+112)<<23)|(m<<13);
    float f;memcpy(&f,&b,4);return f;
}
static uint16_t rd16(const uint8_t*p){return (uint16_t)p[0]|((uint16_t)p[1]<<8);}

/* ---------- tensor access ---------- */
static const uint8_t *g_map;

typedef struct {
    const uint8_t *w;      /* packed q4 or f16 data */
    float *sc, *bi;        /* decoded f32 scales/biases (q4 only) */
    int rows, cols, is_q4;
} T;

static T get_tensor(const char *name) {
    for (int i = 0; i < BT_COUNT; i++) {
        if (!strcmp(BT[i].name, name)) {
            T t; const uint8_t *base = g_map + BLOB + BT[i].off;
            t.w = base; t.rows = BT[i].rows; t.cols = BT[i].cols;
            t.is_q4 = BT[i].is_q4; t.sc = t.bi = NULL;
            if (t.is_q4) {   /* scale/bias offsets are TENSOR-RELATIVE */
                long ng = (long)t.rows * t.cols / GS;
                t.sc = malloc(ng * 4); t.bi = malloc(ng * 4);
                const uint8_t *sr = base + BT[i].soff, *br = base + BT[i].boff;
                for (long g = 0; g < ng; g++) {
                    t.sc[g] = bf16f(rd16(sr + 2*g));
                    t.bi[g] = bf16f(rd16(br + 2*g));
                }
            }
            return t;
        }
    }
    fprintf(stderr, "tensor not found: %s\n", name); exit(1);
}

/* ---------- fused q4 GEMV (Phase 3 kernel) ---------- */
static void xgsums(const float *x, int K, float *xs) {
    for (int g = 0; g < K / GS; g++) {
        float s = 0; for (int i = 0; i < GS; i++) s += x[g*GS+i];
        xs[g] = s;
    }
}
#ifndef NO_VSX
static void gemv_q4(const T *t, const float *x, float *y) {
    int K = t->cols, gpr = K / GS;
    float xs[FFN / GS];
    if (K > FFN || K % GS != 0) { fprintf(stderr, "gemv_q4: bad K=%d\n", K); exit(1); }
    xgsums(x, K, xs);
    const vector unsigned char vmask = vec_splats((unsigned char)0x0F);
    const vector unsigned char vzero = vec_splats((unsigned char)0);
    const vector unsigned char v4    = vec_splats((unsigned char)4);
    const vector unsigned short z16  = vec_splats((unsigned short)0);
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < t->rows; m++) {
        const uint8_t *row = t->w + (long)m * (K / 2);
        float acc = 0;
        for (int g = 0; g < gpr; g++) {
            const uint8_t *p = row + g * (GS / 2);
            const float *xg = x + g * GS;
            vector float vacc = vec_splats(0.0f);
            for (int half = 0; half < 2; half++) {
                vector unsigned char v  = vec_xl(half * 16, p);
                vector unsigned char lo = vec_and(v, vmask);
                vector unsigned char hi = vec_sr(v, v4);
                vector unsigned char q01 = vec_mergeh(lo, hi);
                vector unsigned char q23 = vec_mergel(lo, hi);
                vector unsigned short u16[4];
                u16[0]=(vector unsigned short)vec_mergeh(q01,vzero);
                u16[1]=(vector unsigned short)vec_mergel(q01,vzero);
                u16[2]=(vector unsigned short)vec_mergeh(q23,vzero);
                u16[3]=(vector unsigned short)vec_mergel(q23,vzero);
                const float *xb = xg + half * 32;
                for (int k = 0; k < 4; k++) {
                    vector unsigned int a=(vector unsigned int)vec_mergeh(u16[k],z16);
                    vector unsigned int b=(vector unsigned int)vec_mergel(u16[k],z16);
                    vacc = vec_madd(vec_ctf(a,0), vec_xl(k*32,    xb), vacc);
                    vacc = vec_madd(vec_ctf(b,0), vec_xl(k*32+16, xb), vacc);
                }
            }
            float qdot = vacc[0]+vacc[1]+vacc[2]+vacc[3];
            long gi = (long)m * gpr + g;
            acc += t->sc[gi] * qdot + t->bi[gi] * xs[g];
        }
        y[m] = acc;
    }
}
#else
static void gemv_q4(const T *t, const float *x, float *y) {
    int K = t->cols, gpr = K / GS;
    float xs[FFN / GS];
    if (K > FFN || K % GS != 0) { fprintf(stderr, "gemv_q4: bad K=%d\n", K); exit(1); }
    xgsums(x, K, xs);
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < t->rows; m++) {
        const uint8_t *row = t->w + (long)m * (K / 2);
        float acc = 0;
        for (int g = 0; g < gpr; g++) {
            const uint8_t *p = row + g * (GS / 2);
            const float *xg = x + g * GS;
            float qd = 0;
            for (int i = 0; i < GS/2; i++) {
                qd += (float)(p[i]&0xF)*xg[2*i] + (float)(p[i]>>4)*xg[2*i+1];
            }
            long gi = (long)m * gpr + g;
            acc += t->sc[gi]*qd + t->bi[gi]*xs[g];
        }
        y[m] = acc;
    }
}
#endif

/* ---------- small ops ---------- */
static void rmsnorm(const float *x, const float *w, float *o, int n) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i]*x[i];
    float inv = 1.0f / sqrtf((float)(ss / n) + EPS);
    for (int i = 0; i < n; i++) o[i] = x[i] * inv * w[i];
}
static void decode_f16_vec(const uint8_t *src, float *dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = f16f(rd16(src + 2*i));
}
/* neox rope: halves rotation, applied in place to one 128-dim head */
static void rope(float *h, int pos) {
    for (int i = 0; i < HDIM/2; i++) {
        float freq = powf(ROPE_THETA, -(float)(2*i) / HDIM);
        float a = pos * freq, c = cosf(a), s = sinf(a);
        float x1 = h[i], x2 = h[i + HDIM/2];
        h[i]          = x1*c - x2*s;
        h[i + HDIM/2] = x2*c + x1*s;
    }
}

/* ---------- model ---------- */
typedef struct {
    T q, k, v, o, gate, up, down;
    float in_norm[HIDDEN], post_norm[HIDDEN], qn[HDIM], kn[HDIM];
} Layer;

static Layer L[NLAYERS];
static T lm_head; static const uint8_t *embed;
static float final_norm[HIDDEN];
static float kcache[NLAYERS][MAXCTX][NKVH*HDIM];
static float vcache[NLAYERS][MAXCTX][NKVH*HDIM];

static void load_model(void) {
    char nm[128];
    for (int l = 0; l < NLAYERS; l++) {
        #define GET(field, fmt) snprintf(nm,sizeof nm,fmt,l); L[l].field = get_tensor(nm)
        GET(q, "layers.%d.self_attn.q_proj.weight");
        GET(k, "layers.%d.self_attn.k_proj.weight");
        GET(v, "layers.%d.self_attn.v_proj.weight");
        GET(o, "layers.%d.self_attn.o_proj.weight");
        GET(gate, "layers.%d.mlp.gate_proj.weight");
        GET(up,   "layers.%d.mlp.up_proj.weight");
        GET(down, "layers.%d.mlp.down_proj.weight");
        #undef GET
        snprintf(nm,sizeof nm,"layers.%d.input_norm.weight",l);
        decode_f16_vec(get_tensor(nm).w, L[l].in_norm, HIDDEN);
        snprintf(nm,sizeof nm,"layers.%d.post_attn_norm.weight",l);
        decode_f16_vec(get_tensor(nm).w, L[l].post_norm, HIDDEN);
        snprintf(nm,sizeof nm,"layers.%d.self_attn.q_norm.weight",l);
        decode_f16_vec(get_tensor(nm).w, L[l].qn, HDIM);
        snprintf(nm,sizeof nm,"layers.%d.self_attn.k_norm.weight",l);
        decode_f16_vec(get_tensor(nm).w, L[l].kn, HDIM);
    }
    lm_head = get_tensor("lm_head.weight");
    embed = get_tensor("embed_tokens.weight").w;
    decode_f16_vec(get_tensor("final_norm.weight").w, final_norm, HIDDEN);
}

/* forward one token at position pos; returns logits into `logits`
 * (only computed when want_logits) */
static void forward(int tok, int pos, float *logits, int want_logits) {
    static float x[HIDDEN], h[HIDDEN], q[NQH*HDIM], k[NKVH*HDIM], v[NKVH*HDIM];
    static float attn[NQH*HDIM], ff_g[FFN], ff_u[FFN], ff_d[HIDDEN], o[HIDDEN];
    decode_f16_vec(embed + (long)tok * HIDDEN * 2, x, HIDDEN);

    for (int l = 0; l < NLAYERS; l++) {
        rmsnorm(x, L[l].in_norm, h, HIDDEN);
        gemv_q4(&L[l].q, h, q);
        gemv_q4(&L[l].k, h, k);
        gemv_q4(&L[l].v, h, v);
        for (int hd = 0; hd < NQH; hd++) {          /* per-head q norm + rope */
            float tmp[HDIM];
            rmsnorm(q + hd*HDIM, L[l].qn, tmp, HDIM);
            memcpy(q + hd*HDIM, tmp, sizeof tmp);
            rope(q + hd*HDIM, pos);
        }
        for (int hd = 0; hd < NKVH; hd++) {
            float tmp[HDIM];
            rmsnorm(k + hd*HDIM, L[l].kn, tmp, HDIM);
            memcpy(k + hd*HDIM, tmp, sizeof tmp);
            rope(k + hd*HDIM, pos);
        }
        memcpy(kcache[l][pos], k, sizeof k);
        memcpy(vcache[l][pos], v, sizeof v);

        /* GQA attention: q head hd -> kv head hd/2 */
        float scale = 1.0f / sqrtf((float)HDIM);
        #pragma omp parallel for schedule(static)
        for (int hd = 0; hd < NQH; hd++) {
            int kv = hd / (NQH / NKVH);
            float scores[MAXCTX];
            float mx = -1e30f;
            for (int t = 0; t <= pos; t++) {
                float s = 0;
                const float *kt = kcache[l][t] + kv*HDIM;
                const float *qh = q + hd*HDIM;
                for (int i = 0; i < HDIM; i++) s += qh[i]*kt[i];
                s *= scale;
                scores[t] = s; if (s > mx) mx = s;
            }
            float sum = 0;
            for (int t = 0; t <= pos; t++) { scores[t] = expf(scores[t]-mx); sum += scores[t]; }
            float *ao = attn + hd*HDIM;
            memset(ao, 0, HDIM*4);
            for (int t = 0; t <= pos; t++) {
                float p = scores[t] / sum;
                const float *vt = vcache[l][t] + kv*HDIM;
                for (int i = 0; i < HDIM; i++) ao[i] += p * vt[i];
            }
        }
        gemv_q4(&L[l].o, attn, o);
        for (int i = 0; i < HIDDEN; i++) x[i] += o[i];

        rmsnorm(x, L[l].post_norm, h, HIDDEN);
        gemv_q4(&L[l].gate, h, ff_g);
        gemv_q4(&L[l].up,   h, ff_u);
        for (int i = 0; i < FFN; i++) {
            float g = ff_g[i];
            ff_g[i] = (g / (1.0f + expf(-g))) * ff_u[i];   /* silu(g)*u */
        }
        gemv_q4(&L[l].down, ff_g, ff_d);
        for (int i = 0; i < HIDDEN; i++) x[i] += ff_d[i];
    }
    if (want_logits) {
        rmsnorm(x, final_norm, h, HIDDEN);
        gemv_q4(&lm_head, h, logits);
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.base max_new id0 id1 ...\n", argv[0]);
        return 2;
    }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror(argv[1]); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (long)BLOB) {
        fprintf(stderr, "%s: not a plausible .base file\n", argv[1]); return 1;
    }
    g_map = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (g_map == MAP_FAILED) { perror("mmap"); return 1; }
    if (memcmp(g_map, "BASE", 4) != 0) {
        fprintf(stderr, "%s: bad magic (not a .base file)\n", argv[1]); return 1;
    }

    load_model();
    fprintf(stderr, "model loaded (28 layers, scales decoded)\n");

    int max_new = atoi(argv[2]);
    int prompt[MAXCTX], plen = 0;
    for (int i = 3; i < argc && plen < MAXCTX; i++) {
        int id = atoi(argv[i]);
        if (id < 0 || id >= VOCAB) {
            fprintf(stderr, "token id %d out of range [0,%d)\n", id, VOCAB);
            return 2;
        }
        prompt[plen++] = id;
    }
    if (plen == 0) { fprintf(stderr, "empty prompt\n"); return 2; }

    static float logits[VOCAB];
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int p = 0; p < plen; p++)
        forward(prompt[p], p, logits, p == plen - 1);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double prefill = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9;
    fprintf(stderr, "prefill %d tokens in %.2fs (%.2f tok/s)\n",
            plen, prefill, plen/prefill);

    /* top-5 logits after prefill (non-destructive; diagnostic only) */
    int top[5] = {-1,-1,-1,-1,-1};
    for (int r = 0; r < 5; r++) {
        int best = -1; float bv = -1e30f;
        for (int i = 0; i < VOCAB; i++) {
            int seen = 0;
            for (int s = 0; s < r; s++) if (top[s] == i) seen = 1;
            if (!seen && logits[i] > bv) { bv = logits[i]; best = i; }
        }
        top[r] = best;
        fprintf(stderr, "top%d: id=%d logit=%.4f\n", r+1, best, logits[best]);
    }
    printf("generated:");
    int pos = plen;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int n_gen = 0;
    for (int n = 0; n < max_new && pos < MAXCTX; n++) {
        int best = 0; float bv = logits[0];
        for (int i = 1; i < VOCAB; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        printf(" %d", best); fflush(stdout);
        n_gen++;
        if (best == 151645 || best == 151643) break;   /* eos */
        forward(best, pos++, logits, 1);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gen = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9;
    printf("\n");
    fprintf(stderr, "decode: %d tokens in %.2fs (%.2f tok/s)\n",
            n_gen, gen, n_gen/gen);
    return 0;
}
