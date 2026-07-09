# engine-ppc64 — .base format tools and inference for IBM POWER8

Experimental PowerPC (ppc64le) implementation of the BaseRT `.base` model
format: a container parser, BaseQ4 dequantization kernels for POWER8 VSX,
a fused q4 GEMV, and a minimal Qwen3 forward pass that reads a `.base`
bundle directly. Built against the published format spec
(`base-convert/FORMAT.md`, `CANONICAL_QUANT_SPEC.md`) and validated with
the `base-format` / `base-quant` reference implementation in this repo.

The upstream BaseRT engine is Metal-only by design. This directory is an
independent engine experiment for non-Apple hardware. It is not a port of
the engine binary and shares no code with it.

Tested on an IBM Power System S824 (POWER8, 16 cores / SMT8, ppc64le,
Ubuntu 20.04) against the catalog `Qwen/Qwen3-0.6B` default-q4 bundle.

## Status

| Component | File | Status |
|-----------|------|--------|
| Container parser | `base_inspect.c` | Matches `basert inspect` field for field on x86-64 and ppc64le |
| BaseQ4 dequant (VSX) | `base_q4_vsx.c` | Bit-identical to scalar reference on synthetic roundtrip and real tensors; 3.3 Gval/s single thread |
| Fused q4 GEMV | `base_q4_gemv.c` | rel err ~1e-4 vs reference (fp ordering); 38 GFLOP/s at 64 threads |
| Qwen3 forward pass | `qwen3_base.c` | Runs end to end at ~18 tok/s decode (32 threads). KNOWN ISSUE below |

### Known issue: top-1 logit mismatch

The forward pass produces grammatical output but the greedy argmax does
not yet match the reference runtime. For the prompt "The capital of
France is", the expected token " Paris" ranks 5th, about 1.2 logits below
our top-1. All structural components (weight orientation, attention, GQA
mapping, RoPE, SwiGLU) produce coherent English, and the f16 decoder is
exhaustively verified against numpy (0/65536 mismatches), so the residual
defect is a small numeric term. Differential debugging against the HF
reference model is the next step. Treat generation output as
demonstrative, not correct, until this is closed.

### Scaffolding disclosure

`qwen3_base.c` loads tensor geometry from a generated header
(`tensors.h`, produced by `gen_tensors.py` from the bundle's JSON header)
rather than parsing the header at runtime, and the file offsets in the
two probe tools are hardcoded for the Qwen3-0.6B default-q4 bundle
specifically. A general runtime loader is the next milestone. The
inference CLI validates magic and rejects out-of-range token ids, but do
not point these tools at arbitrary bundles and trust the output.

Scale and bias offsets in the tensor header are tensor-relative.
`lm_head` sits at tensor offset 0, so the probe tools' constants coincide
with blob-relative values for that tensor only.

## Build and run

```sh
# on a POWER8 (ppc64le) host
gcc -O3 -mcpu=power8 -maltivec -mvsx -o base_inspect base_inspect.c
gcc -O3 -mcpu=power8 -maltivec -mvsx -o base_q4_vsx base_q4_vsx.c -lm
gcc -O3 -mcpu=power8 -maltivec -mvsx -fopenmp -o base_q4_gemv base_q4_gemv.c -lm
gcc -O3 -mcpu=power8 -maltivec -mvsx -fopenmp -o qwen3_base qwen3_base.c -lm

./base_inspect model.base
./base_q4_vsx selftest && ./base_q4_vsx probe model.base && ./base_q4_vsx bench
./base_q4_gemv selftest && ./base_q4_gemv bench 8192 4096

# regenerate tensors.h for your bundle, then run inference
python3 gen_tensors.py model.base > tensors.h
gcc -O3 -mcpu=power8 -maltivec -mvsx -fopenmp -o qwen3_base qwen3_base.c -lm
./qwen3_base model.base 16 785 6722 315 9625 374
```

The scalar reference paths build on any host with `-DNO_VSX`.

## Spec observations from independent implementation

Noted while implementing from FORMAT.md against a real catalog bundle:

- `created` is serialized as a JSON string, not a number.
- Header JSON keys are not sorted, although FORMAT.md describes the
  header as canonical JSON with sorted keys.
- The v0.1.5 engine tarball installs a binary that reports `basert 0.1.0`.
- The README's `model:variant` and `--variant` pull syntaxes are not
  accepted by the shipped CLI; `--target base-q8` style works.

## License

Files in this directory are MIT licensed (SPDX headers in each file).
The surrounding repository is Apache-2.0.
