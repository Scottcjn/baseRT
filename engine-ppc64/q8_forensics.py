# SPDX-License-Identifier: MIT
# Q8-MoE forensics: same tensor, two quantizations of the same source model.
# If Q8 decodes ~= Q4 decode -> bundle data fine -> engine kernel bug.
# If Q8 decodes to garbage   -> converter/bundle bug.
import json
import struct
import subprocess
import numpy as np

URL = "https://huggingface.co/basecompute/gemma-4-26B-A4B-it/resolve/main/gemma-4-26B-A4B-it-{v}.base"
TENSOR = "layers.0.ffn_down_exps.weight"
NCHECK = 2_000_000   # ~1 expert worth of values


def header(path):
    f = open(path, "rb")
    assert f.read(4) == b"BASE"
    struct.unpack("<I", f.read(4))
    (hlen,) = struct.unpack("<Q", f.read(8))
    h = json.loads(f.read(hlen))
    blob = (16 + hlen + 0xFFFF) & ~0xFFFF
    return h, blob


def fetch(variant, t, blob, out):
    a = blob + t["offset"]
    b = a + t["length"] - 1
    subprocess.run(["curl", "-sL", "-r", f"{a}-{b}", "-o", out,
                    URL.format(v=variant)], check=True)
    return np.memmap(out, dtype=np.uint8, mode="r")


def f16_arr(u8):
    return u8.view(np.float16).astype(np.float32)


def bf16_arr(u8):
    return (u8.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


h4, blob4 = header("g26_head.bin")
h8, blob8 = header("g26q8_head.bin")
t4 = next(t for t in h4["tensors"] if t["name"] == TENSOR)
t8 = next(t for t in h8["tensors"] if t["name"] == TENSOR)
print("q4 entry:", {k: t4[k] for k in ("dtype", "group_size", "scale_dtype")})
print("q8 entry:", {k: t8[k] for k in ("dtype", "group_size", "scale_dtype")})

print("fetching q4 slice...")
r4 = fetch("Q4", t4, blob4, "t4.bin")
print("fetching q8 slice...")
r8 = fetch("Q8", t8, blob8, "t8.bin")

# ---- decode q4 (proven decoder: unsigned nibbles, low first, x=q*s+b) ----
gs4 = t4["group_size"]
n = NCHECK - (NCHECK % (gs4 * 128))
packed = r4[: n // 2]
q = np.empty(n, np.float32)
q[0::2] = packed & 0x0F
q[1::2] = packed >> 4
sd4 = t4.get("scale_dtype")
dec = bf16_arr if sd4 == "bf16" else f16_arr
sc4 = dec(r4[t4["scale_offset"]: t4["scale_offset"] + (n // gs4) * 2])
bi4 = dec(r4[t4["bias_offset"]: t4["bias_offset"] + (n // gs4) * 2])
w4 = q * sc4.repeat(gs4) + bi4.repeat(gs4)

# ---- decode q8 candidates ----
gs8 = t8["group_size"]
sc8 = f16_arr(r8[t8["scale_offset"]: t8["scale_offset"] + (n // gs8) * 2])
bi8 = f16_arr(r8[t8["bias_offset"]: t8["bias_offset"] + (n // gs8) * 2])
raw = r8[:n]
cands = {
    "unsigned q*s+b": raw.astype(np.float32) * sc8.repeat(gs8) + bi8.repeat(gs8),
    "signed  q*s+b": raw.view(np.int8).astype(np.float32) * sc8.repeat(gs8) + bi8.repeat(gs8),
    "signed  q*s   ": raw.view(np.int8).astype(np.float32) * sc8.repeat(gs8),
}
print(f"\nscale stats q8: min={sc8.min():.3e} max={sc8.max():.3e} zeros={int((sc8==0).sum())}/{len(sc8)} nan={int(np.isnan(sc8).sum())}")
print(f"bias  stats q8: min={bi8.min():.3e} max={bi8.max():.3e} nan={int(np.isnan(bi8).sum())}")
print(f"q4 reference:   mean|w|={np.abs(w4).mean():.5f} std={w4.std():.5f}")
for name, w8 in cands.items():
    corr = float(np.corrcoef(w4, w8)[0, 1])
    print(f"q8 [{name}] mean|w|={np.abs(w8).mean():.5f} std={w8.std():.5f} corr_vs_q4={corr:+.4f}")
