/* SPDX-License-Identifier: MIT
 * base_inspect — minimal .base (BaseRT v1) file inspector
 *
 * Elyan Labs — Phase 1 of the .base-on-POWER8 plan.
 * Reads the container: magic, version, header JSON, tensor table.
 * Endian-explicit: works on LE (ppc64le, x86) and BE (G4/G5) hosts.
 *
 * Format ref: basecompute/baseRT base-convert/FORMAT.md
 *   0x00  4  magic "BASE"
 *   0x04  4  format_version u32 LE
 *   0x08  8  header_len u64 LE (canonical JSON)
 *   0x10  N  header_json
 *   pad to 64 KiB boundary, then weights blob
 *
 * Build: gcc -O2 -o base_inspect base_inspect.c
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd_u32le(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint64_t rd_u64le(const uint8_t *p) {
    uint64_t lo = rd_u32le(p), hi = rd_u32le(p + 4);
    return lo | hi << 32;
}

/* Naive scanners for canonical JSON (keys sorted, no stray whitespace).
 * Good enough for Phase 1; a real parser lands in Phase 2. */
static int json_str(const char *j, const char *key, char *out, size_t cap) {
    char pat[128];
    snprintf(pat, sizeof pat, "\"%s\":\"", key);
    const char *p = strstr(j, pat);
    if (!p) return 0;
    p += strlen(pat);
    size_t n = strcspn(p, "\"");
    if (n >= cap) n = cap - 1;
    memcpy(out, p, n); out[n] = 0;
    return 1;
}
static long long json_int(const char *j, const char *key) {
    char pat[128];
    snprintf(pat, sizeof pat, "\"%s\":", key);
    const char *p = strstr(j, pat);
    if (!p) return -1;
    p += strlen(pat);
    if (*p == '"') p++;   /* numbers sometimes arrive as JSON strings */
    return atoll(p);
}
static long count_occurrences(const char *j, const char *pat) {
    long n = 0;
    for (const char *p = strstr(j, pat); p; p = strstr(p + 1, pat)) n++;
    return n;
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s model.base\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }

    uint8_t pre[16];
    if (fread(pre, 1, 16, f) != 16) { fprintf(stderr, "short read\n"); return 1; }
    if (memcmp(pre, "BASE", 4) != 0) { fprintf(stderr, "bad magic\n"); return 1; }
    uint32_t version   = rd_u32le(pre + 4);
    uint64_t header_len = rd_u64le(pre + 8);

    char *hdr = malloc(header_len + 1);
    if (!hdr || fread(hdr, 1, header_len, f) != header_len) {
        fprintf(stderr, "header read failed\n"); return 1;
    }
    hdr[header_len] = 0;

    /* weights blob begins at the next 64 KiB boundary */
    uint64_t blob_off = (16 + header_len + 0xFFFF) & ~0xFFFFULL;

    fseek(f, 0, SEEK_END);
    uint64_t file_size = (uint64_t)ftell(f);

    char arch[64] = "?", scheme[64] = "?", min_hw[64] = "?",
         backend[64] = "?", ver_rt[64] = "?", profile[64] = "?";
    json_str(hdr, "arch", arch, sizeof arch);
    json_str(hdr, "quant_scheme", scheme, sizeof scheme);
    json_str(hdr, "min_hw", min_hw, sizeof min_hw);
    json_str(hdr, "target_backend", backend, sizeof backend);
    json_str(hdr, "baserT_version", ver_rt, sizeof ver_rt);
    json_str(hdr, "quant_profile", profile, sizeof profile);

    long long created  = json_int(hdr, "created");
    long long n_layers = json_int(hdr, "num_hidden_layers");
    long long hidden   = json_int(hdr, "hidden_size");
    long long heads    = json_int(hdr, "num_attention_heads");
    long long kv_heads = json_int(hdr, "num_key_value_heads");
    long n_tensors     = count_occurrences(hdr, "\"checksum_xxh64\":");
    long n_q4          = count_occurrences(hdr, "\"dtype\":\"base_q4\"");
    long n_f16         = count_occurrences(hdr, "\"dtype\":\"f16\"");

    printf("file:          %s (%llu bytes)\n", argv[1], (unsigned long long)file_size);
    printf("magic:         BASE  version: %u\n", version);
    printf("header_len:    %llu\n", (unsigned long long)header_len);
    printf("arch:          %s\n", arch);
    printf("quant_scheme:  %s  (profile %s)\n", scheme, profile);
    printf("min_hw:        %s  (ignored by this loader)\n", min_hw);
    printf("target_backend:%s\n", backend);
    printf("base_rt:       %s\n", ver_rt);
    printf("created:       %lld\n", created);
    printf("model:         layers=%lld hidden=%lld heads=%lld/%lld\n",
           n_layers, hidden, heads, kv_heads);
    printf("n_tensors:     %ld  (base_q4=%ld, f16=%ld)\n", n_tensors, n_q4, n_f16);
    printf("weights blob:  offset 0x%llx, %llu bytes to EOF\n",
           (unsigned long long)blob_off,
           (unsigned long long)(file_size - blob_off));
    printf("host endian:   %s (readers are endian-explicit)\n",
           (*(const uint16_t *)"\x01\x00" == 1) ? "little" : "BIG");

    free(hdr); fclose(f);
    return 0;
}
