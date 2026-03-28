#!/usr/bin/env python3
"""
Full comparison of all FA backends on MI325X (gfx942).
CK (aiter v3 ASM + CK tile), Triton AVO, PyTorch SDPA.
"""
import torch
import sys
import math
import time
import json
from pathlib import Path

sys.path.insert(0, "/workspace/fa4_rocm")


def calc_flops(batch, seqlen_q, seqlen_k, nheads, hdim, causal):
    flops = 4 * batch * nheads * seqlen_q * seqlen_k * hdim
    if causal:
        flops //= 2
    return flops


def bench_fn(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    times.sort()
    return times[len(times) // 2]


def main():
    from ck_kernels.flash_attn_ck import ck_flash_attn_func

    # Try loading triton
    triton_ok = False
    try:
        from triton_kernels.flash_fwd_triton import flash_attn_triton_func
        triton_ok = True
    except Exception:
        pass

    # Try loading aiter directly for separate v3 vs CK tile measurement
    aiter_v3_ok = False
    try:
        from aiter import fmha_v3_fwd, mha_fwd
        aiter_v3_ok = True
    except Exception:
        pass

    shapes = [
        # (batch, seqlen, nheads_q, nheads_k, hdim, causal, label)
        (2,  2048, 32, 32, 128, True,  "b2_s2048_h32_d128_causal"),
        (2,  4096, 32, 32, 128, True,  "b2_s4096_h32_d128_causal"),
        (1,  8192, 32, 32, 128, True,  "b1_s8192_h32_d128_causal"),
        (2,  2048, 32, 32, 128, False, "b2_s2048_h32_d128_noncausal"),
        (2,  2048, 32,  8, 128, True,  "b2_s2048_h32_d128_GQA4_causal"),
        (4,  2048, 32, 32, 128, True,  "b4_s2048_h32_d128_causal"),
        (8,  1024, 32, 32, 128, True,  "b8_s1024_h32_d128_causal"),
        (1, 16384, 32, 32, 128, True,  "b1_s16k_h32_d128_causal"),
    ]

    all_results = []

    header = f"{'Shape':<38}"
    backends = ["CK(aiter)"]
    if triton_ok:
        backends.append("Triton")
    backends.append("SDPA")
    for b in backends:
        header += f" {b+' TF':>12}"
    header += f" {'CK/SDPA':>10}"
    if triton_ok:
        header += f" {'CK/Triton':>10}"

    print("=" * len(header))
    print("FA4 ROCm Full Comparison -- MI325X (gfx942) -- BF16")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for batch, seqlen, nhq, nhk, hdim, causal, label in shapes:
        q = torch.randn(batch, seqlen, nhq, hdim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, seqlen, nhk, hdim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, seqlen, nhk, hdim, dtype=torch.bfloat16, device="cuda")
        scale = 1.0 / math.sqrt(hdim)
        flops = calc_flops(batch, seqlen, seqlen, nhq, hdim, causal)

        result = {"shape": label, "batch": batch, "seqlen": seqlen,
                  "nheads_q": nhq, "nheads_k": nhk, "hdim": hdim, "causal": causal}

        # CK via aiter (v3 ASM for bf16 hdim128, CK tile fallback)
        ck_t = bench_fn(lambda: ck_flash_attn_func(q, k, v, causal=causal, softmax_scale=scale, mode="aiter"))
        ck_tf = flops / ck_t / 1e12
        result["ck_tflops"] = round(ck_tf, 1)
        result["ck_us"] = round(ck_t * 1e6, 1)

        # Triton
        triton_tf = 0.0
        if triton_ok:
            try:
                triton_t = bench_fn(lambda: flash_attn_triton_func(q, k, v, causal=causal, softmax_scale=scale))
                triton_tf = flops / triton_t / 1e12
                result["triton_tflops"] = round(triton_tf, 1)
                result["triton_us"] = round(triton_t * 1e6, 1)
            except Exception as e:
                result["triton_tflops"] = 0
                result["triton_error"] = str(e)

        # SDPA
        qt = q.transpose(1, 2).contiguous()
        kt_orig = k.transpose(1, 2).contiguous()
        vt_orig = v.transpose(1, 2).contiguous()
        if nhq != nhk:
            ratio = nhq // nhk
            kt = kt_orig.repeat_interleave(ratio, dim=1).contiguous()
            vt = vt_orig.repeat_interleave(ratio, dim=1).contiguous()
        else:
            kt, vt = kt_orig, vt_orig

        sdpa_t = bench_fn(lambda: torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=causal))
        sdpa_tf = flops / sdpa_t / 1e12
        result["sdpa_tflops"] = round(sdpa_tf, 1)
        result["sdpa_us"] = round(sdpa_t * 1e6, 1)

        # Print row
        row = f"{label:<38} {ck_tf:>11.1f}"
        if triton_ok:
            row += f" {triton_tf:>11.1f}"
        row += f" {sdpa_tf:>11.1f}"
        row += f" {ck_tf/sdpa_tf:>9.2f}x"
        if triton_ok and triton_tf > 0:
            row += f" {ck_tf/triton_tf:>9.2f}x"
        print(row)

        all_results.append(result)

    print("-" * len(header))

    # Summary
    ck_avg = sum(r["ck_tflops"] for r in all_results) / len(all_results)
    sdpa_avg = sum(r["sdpa_tflops"] for r in all_results) / len(all_results)
    print(f"\nAverage CK: {ck_avg:.1f} TF  |  Average SDPA: {sdpa_avg:.1f} TF  |  Avg speedup: {ck_avg/sdpa_avg:.2f}x")

    if triton_ok:
        triton_results = [r for r in all_results if r.get("triton_tflops", 0) > 0]
        if triton_results:
            triton_avg = sum(r["triton_tflops"] for r in triton_results) / len(triton_results)
            ck_avg_match = sum(r["ck_tflops"] for r in triton_results) / len(triton_results)
            print(f"Average Triton: {triton_avg:.1f} TF  |  CK/Triton avg speedup: {ck_avg_match/triton_avg:.2f}x")

    # Historical comparison
    print(f"\n--- Historical Comparison (b2 s2048 h32 d128 causal BF16) ---")
    ref = all_results[0]
    print(f"  Triton AVO R20 best:  197.5 TF  (from evolution_v2_log.json)")
    print(f"  CK (aiter v3 ASM):    {ref['ck_tflops']:.1f} TF  (this run)")
    print(f"  aiter baseline:       347.6 TF  (from evolution_v2_log.json)")
    print(f"  SDPA:                 {ref['sdpa_tflops']:.1f} TF  (this run)")
    if ref.get("triton_tflops"):
        print(f"  Triton (this run):    {ref['triton_tflops']:.1f} TF")
    print(f"\n  CK vs Triton AVO R20: {ref['ck_tflops']/197.5:.2f}x improvement")
    print(f"  CK vs SDPA:           {ref['ck_tflops']/ref['sdpa_tflops']:.2f}x improvement")

    # Save
    out_path = Path("/workspace/fa4_rocm/ck_kernels/full_comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
