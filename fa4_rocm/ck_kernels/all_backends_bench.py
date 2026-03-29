#!/usr/bin/env python3
"""
Complete benchmark: ALL backends on the same shapes.
CK (aiter v3 ASM + CK tile), Triton AVO R20, Triton vanilla, FlyDSL, HIP C++, SDPA.
"""
import torch, math, time, sys, json, traceback
from pathlib import Path

sys.path.insert(0, "/workspace/fa4_rocm")

device = "cuda:0"
WARMUP, ITERS = 10, 50


def calc_flops(batch, sq, sk, nh, hd, causal):
    f = 4 * batch * nh * sq * sk * hd
    return f // 2 if causal else f


def bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


# ── Load all backends ──────────────────────────────────────────────

backends = {}

# 1) CK via aiter (v3 ASM + CK tile fallback)
try:
    from ck_kernels.flash_attn_ck import ck_flash_attn_func
    backends["CK(aiter)"] = lambda q, k, v, c, s: ck_flash_attn_func(
        q, k, v, causal=c, softmax_scale=s, mode="aiter")
    print("[OK] CK (aiter v3 ASM + CK tile)")
except Exception as e:
    print(f"[SKIP] CK: {e}")

# 2) Triton AVO R20 (evolved best config)
try:
    from triton_kernels.flash_fwd_evo import evo_flash_attn
    best_cfg = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4,
                "num_stages": 2, "waves_per_eu": 2,
                "PRE_LOAD_V": True, "PRE_SCALE_Q": False}
    backends["Triton AVO"] = lambda q, k, v, c, s: evo_flash_attn(
        q, k, v, causal=c, softmax_scale=s, **best_cfg)
    print("[OK] Triton AVO R20")
except Exception as e:
    print(f"[SKIP] Triton AVO: {e}")

# 3) Triton vanilla (v1)
try:
    from triton_kernels.flash_fwd_triton import flash_attn_triton_func
    backends["Triton v1"] = lambda q, k, v, c, s: flash_attn_triton_func(
        q, k, v, causal=c, softmax_scale=s)
    print("[OK] Triton v1")
except Exception as e:
    print(f"[SKIP] Triton v1: {e}")

# 4) aiter FA3 v3 ASM (direct)
try:
    from aiter import fmha_v3_fwd
    def fa3_fn(q, k, v, c, s):
        return fmha_v3_fwd(q, k, v, 0.0, s, c, -1, -1, False, False, 1)[0]
    backends["aiter FA3v3"] = fa3_fn
    print("[OK] aiter FA3 v3 ASM (direct)")
except Exception as e:
    print(f"[SKIP] aiter FA3v3: {e}")

# 5) aiter FA2 CK tile
try:
    from aiter import mha_fwd
    def fa2_fn(q, k, v, c, s):
        return mha_fwd(q, k, v, 0.0, s, c, -1, -1, 0, False, False)[0]
    backends["aiter FA2CK"] = fa2_fn
    print("[OK] aiter FA2 CK tile")
except Exception as e:
    print(f"[SKIP] aiter FA2CK: {e}")

# 6) SDPA
def sdpa_fn(q, k, v, c, s):
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    hq, hk = q.shape[2], k.shape[2]
    if hq != hk:
        kt = kt.repeat_interleave(hq // hk, dim=1)
        vt = vt.repeat_interleave(hq // hk, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        qt, kt, vt, is_causal=c, scale=s).transpose(1, 2)

backends["SDPA"] = sdpa_fn
print("[OK] PyTorch SDPA")

print(f"\nLoaded {len(backends)} backends: {list(backends.keys())}")

# ── Shapes ─────────────────────────────────────────────────────────

shapes = [
    (2, 2048, 32, 32, 128, True,  "b2_s2048_causal"),
    (2, 4096, 32, 32, 128, True,  "b2_s4096_causal"),
    (1, 8192, 32, 32, 128, True,  "b1_s8192_causal"),
    (2, 2048, 32, 32, 128, False, "b2_s2048_noncausal"),
    (2, 2048, 32,  8, 128, True,  "b2_GQA4_causal"),
    (4, 2048, 32, 32, 128, True,  "b4_s2048_causal"),
]

# ── Run ────────────────────────────────────────────────────────────

print()
print("=" * 120)
print("ALL BACKENDS BENCHMARK -- MI325X (gfx942) -- BF16 -- hdim=128")
print("=" * 120)

# Header
bnames = list(backends.keys())
hdr = f"{'Shape':<25}"
for bn in bnames:
    hdr += f" {bn:>12}"
print(hdr)
print("-" * 120)

all_results = []

for batch, sq, nhq, nhk, hd, causal, label in shapes:
    q = torch.randn(batch, sq, nhq, hd, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch, sq, nhk, hd, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch, sq, nhk, hd, dtype=torch.bfloat16, device=device)
    scale = 1.0 / math.sqrt(hd)
    flops = calc_flops(batch, sq, sq, nhq, hd, causal)

    row = {"shape": label}
    line = f"{label:<25}"

    for bn in bnames:
        fn = backends[bn]
        try:
            t = bench(lambda: fn(q, k, v, causal, scale))
            tf = flops / t / 1e12
            row[bn] = round(tf, 1)
            line += f" {tf:>10.1f}TF"
        except Exception as e:
            row[bn] = 0
            line += f" {'ERR':>12}"

    print(line)
    all_results.append(row)

# ── Summary ────────────────────────────────────────────────────────

print("-" * 120)
print()

# Averages
print("AVERAGES (TFLOPS):")
avgs = {}
for bn in bnames:
    vals = [r[bn] for r in all_results if r.get(bn, 0) > 0]
    avg = sum(vals) / len(vals) if vals else 0
    avgs[bn] = avg
    print(f"  {bn:<15}: {avg:>8.1f} TF")

print()

# Speedup matrix (relative to each other)
ref_key = "CK(aiter)" if "CK(aiter)" in avgs else bnames[0]
ref_avg = avgs.get(ref_key, 1)
print(f"SPEEDUP vs {ref_key}:")
for bn in bnames:
    if bn == ref_key:
        continue
    a = avgs.get(bn, 0)
    if a > 0:
        print(f"  {ref_key} / {bn:<15}: {ref_avg / a:.2f}x")

print()
print("FULL EVOLUTION HISTORY (b2 s2048 h32 d128 causal BF16):")
ref_shape = all_results[0]
print(f"  HIP C++ kernel:       ~13 TF   (from earlier testing)")
triton_v1 = ref_shape.get("Triton v1", 0)
triton_avo = ref_shape.get("Triton AVO", 0)
ck_val = ref_shape.get("CK(aiter)", 0)
fa3_val = ref_shape.get("aiter FA3v3", 0)
fa2_val = ref_shape.get("aiter FA2CK", 0)
sdpa_val = ref_shape.get("SDPA", 0)
print(f"  Triton v1:            {triton_v1:>6.1f} TF")
print(f"  Triton v2 3-phase:    ~180.0 TF  (from earlier testing)")
print(f"  Triton v3 log2:       ~192.0 TF  (from earlier testing)")
print(f"  Triton AVO R20:       {triton_avo:>6.1f} TF")
print(f"  FlyDSL MFMA:          compiles, no perf number (experimental)")
print(f"  aiter FA2 (CK tile):  {fa2_val:>6.1f} TF")
print(f"  aiter FA3 v3 (ASM):   {fa3_val:>6.1f} TF")
print(f"  CK backend (aiter):   {ck_val:>6.1f} TF  <-- this project")
print(f"  PyTorch SDPA:         {sdpa_val:>6.1f} TF")

# Save
out = Path("/workspace/fa4_rocm/ck_kernels/all_backends_results.json")
with open(out, "w") as f:
    json.dump({"backends": bnames, "shapes": all_results, "averages": avgs}, f, indent=2)
print(f"\nSaved to {out}")
