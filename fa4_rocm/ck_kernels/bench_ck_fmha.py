#!/usr/bin/env python3
"""
Benchmark CK FMHA (via aiter) and compare with Triton, SDPA baselines.
Reports TFLOPS for standard FA shapes.
"""
import torch
import sys
import math
import time
import json
from pathlib import Path

sys.path.insert(0, "/workspace/fa4_rocm")


def calc_flops(batch, seqlen_q, seqlen_k, nheads, hdim, causal):
    """4 * batch * heads * seqlen_q * seqlen_k * hdim (fwd only)."""
    flops = 4 * batch * nheads * seqlen_q * seqlen_k * hdim
    if causal:
        flops //= 2
    return flops


def bench_fn(fn, q, k, v, warmup=5, iters=20):
    """Time a function with warmup, return median latency in seconds."""
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn(q, k, v)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    times.sort()
    return times[len(times) // 2]


def run_benchmarks():
    from ck_kernels.flash_attn_ck import ck_flash_attn_func

    shapes = [
        # (batch, seqlen, nheads_q, nheads_k, hdim, causal, name)
        (2, 2048, 32, 32, 128, True,  "prefill_b2s2k_causal"),
        (2, 4096, 32, 32, 128, True,  "prefill_b2s4k_causal"),
        (1, 8192, 32, 32, 128, True,  "prefill_b1s8k_causal"),
        (2, 2048, 32, 32, 128, False, "prefill_b2s2k_noncausal"),
        (2, 2048, 32, 8,  128, True,  "gqa4_b2s2k_causal"),
        (4, 2048, 32, 32, 128, True,  "prefill_b4s2k_causal"),
    ]

    results = []
    print(f"{'Shape':<35} {'CK (TF)':<12} {'SDPA (TF)':<12} {'Speedup':<10}")
    print("-" * 75)

    for batch, seqlen, nhq, nhk, hdim, causal, name in shapes:
        q = torch.randn(batch, seqlen, nhq, hdim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, seqlen, nhk, hdim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, seqlen, nhk, hdim, dtype=torch.bfloat16, device="cuda")
        scale = 1.0 / math.sqrt(hdim)
        flops = calc_flops(batch, seqlen, seqlen, nhq, hdim, causal)

        # CK via aiter
        def ck_fn(q, k, v):
            return ck_flash_attn_func(q, k, v, causal=causal, softmax_scale=scale, mode="aiter")

        ck_time = bench_fn(ck_fn, q, k, v)
        ck_tflops = flops / ck_time / 1e12

        # SDPA reference
        qt = q.transpose(1, 2).contiguous()
        kt = k.transpose(1, 2).contiguous()
        vt = v.transpose(1, 2).contiguous()
        if nhq != nhk:
            gqa_ratio = nhq // nhk
            kt = kt.repeat_interleave(gqa_ratio, dim=1).contiguous()
            vt = vt.repeat_interleave(gqa_ratio, dim=1).contiguous()

        def sdpa_fn(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)

        sdpa_time = bench_fn(sdpa_fn, q, k, v)
        sdpa_tflops = flops / sdpa_time / 1e12

        speedup = ck_tflops / sdpa_tflops if sdpa_tflops > 0 else float("inf")
        print(f"{name:<35} {ck_tflops:<12.1f} {sdpa_tflops:<12.1f} {speedup:<10.2f}x")

        results.append({
            "name": name,
            "batch": batch, "seqlen": seqlen,
            "nheads_q": nhq, "nheads_k": nhk,
            "hdim": hdim, "causal": causal,
            "ck_tflops": round(ck_tflops, 1),
            "ck_latency_us": round(ck_time * 1e6, 1),
            "sdpa_tflops": round(sdpa_tflops, 1),
            "sdpa_latency_us": round(sdpa_time * 1e6, 1),
            "speedup": round(speedup, 2),
        })

    # Also try Triton if available
    try:
        from triton_kernels.flash_fwd_triton import flash_attn_triton_func

        print("\n--- Triton comparison (b2 s2048 h32 d128 causal bf16) ---")
        q = torch.randn(2, 2048, 32, 128, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(2, 2048, 32, 128, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(2, 2048, 32, 128, dtype=torch.bfloat16, device="cuda")
        scale = 1.0 / math.sqrt(128)
        flops = calc_flops(2, 2048, 2048, 32, 128, True)

        def triton_fn(q, k, v):
            return flash_attn_triton_func(q, k, v, causal=True, softmax_scale=scale)

        triton_time = bench_fn(triton_fn, q, k, v)
        triton_tf = flops / triton_time / 1e12
        ck_tf = results[0]["ck_tflops"]
        print(f"Triton: {triton_tf:.1f} TF  |  CK: {ck_tf:.1f} TF  |  CK/Triton: {ck_tf/triton_tf:.2f}x")
        results.append({"name": "triton_b2s2k_causal", "triton_tflops": round(triton_tf, 1)})
    except Exception as e:
        print(f"\nTriton not available: {e}")

    # Save results
    out_path = Path("/workspace/fa4_rocm/ck_kernels/bench_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_benchmarks()
