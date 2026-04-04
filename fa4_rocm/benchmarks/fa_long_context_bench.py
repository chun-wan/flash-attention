#!/usr/bin/env python3
"""
FA2 vs FA3/FA4 Long-Context Benchmark on AMD MI325X.

Compares CK v3 ASM (FA3/FA4), CK tile (FA2), Triton, and SDPA
across sequence lengths from 1K to 128K.

Produces a table suitable for README.md.
"""
import torch
import math
import time
import json
import sys

def calc_flops(batch, sq, sk, nh, hd, causal):
    f = 4 * batch * nh * sq * sk * hd
    return f // 2 if causal else f

def bench(fn, warmup=5, iters=20):
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

def main():
    from aiter import flash_attn_func as ck_fa  # dispatches to v3 ASM
    from aiter import mha_fwd                   # CK tile (FA2)

    def fa3_fn(q, k, v, causal, scale):
        """FA3/FA4 via aiter flash_attn_func (auto v3 ASM dispatch)."""
        result = ck_fa(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=causal)
        return result if not isinstance(result, tuple) else result[0]

    def fa2_fn(q, k, v, causal, scale):
        """FA2 via CK tile (mha_fwd)."""
        return mha_fwd(q, k, v, 0.0, scale, causal, -1, -1, 0, False, False)[0]

    def sdpa_fn(q, k, v, causal, scale):
        """PyTorch SDPA."""
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=causal, scale=scale
        ).transpose(1, 2)

    backends = {
        "FA3/FA4 (CK v3 ASM)": fa3_fn,
        "FA2 (CK tile)": fa2_fn,
        "SDPA (PyTorch)": sdpa_fn,
    }

    # Shapes: vary seqlen, fixed batch/heads/hdim
    hd = 128
    nh = 32
    causal = True

    shapes = []
    for sq in [1024, 2048, 4096, 8192, 16384, 32768]:
        # Adjust batch to fit memory
        if sq <= 4096:
            b = 2
        elif sq <= 16384:
            b = 1
        else:
            b = 1
        shapes.append((b, sq, nh, hd, f"b{b}_s{sq//1024}k"))

    print("=" * 90)
    print("FA2 vs FA3/FA4 Long-Context Benchmark -- MI325X (gfx942) -- BF16 Causal")
    print("=" * 90)
    print()

    # Header
    header = f"{'Shape':<18}"
    for name in backends:
        header += f" {name:>20}"
    header += f" {'FA4/FA2':>10} {'FA4/SDPA':>10}"
    print(header)
    print("-" * len(header))

    all_results = []

    for batch, sq, nh_q, hd_val, label in shapes:
        q = torch.randn(batch, sq, nh_q, hd_val, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, sq, nh_q, hd_val, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, sq, nh_q, hd_val, dtype=torch.bfloat16, device="cuda")
        scale = 1.0 / math.sqrt(hd_val)
        flops = calc_flops(batch, sq, sq, nh_q, hd_val, causal)

        row = {"shape": label, "batch": batch, "seqlen": sq}
        line = f"{label:<18}"

        for name, fn in backends.items():
            try:
                t = bench(lambda: fn(q, k, v, causal, scale))
                tf = flops / t / 1e12
                row[name] = round(tf, 1)
                line += f" {tf:>18.1f}TF"
            except Exception as e:
                row[name] = 0
                line += f" {'OOM/ERR':>20}"

        # Speedup
        fa4 = row.get("FA3/FA4 (CK v3 ASM)", 0)
        fa2 = row.get("FA2 (CK tile)", 0)
        sdpa = row.get("SDPA (PyTorch)", 0)
        fa4_fa2 = f"{fa4/fa2:.2f}x" if fa2 > 0 else "N/A"
        fa4_sdpa = f"{fa4/sdpa:.2f}x" if sdpa > 0 else "N/A"
        line += f" {fa4_fa2:>10} {fa4_sdpa:>10}"
        row["fa4_vs_fa2"] = fa4_fa2
        row["fa4_vs_sdpa"] = fa4_sdpa

        print(line)
        all_results.append(row)

        del q, k, v
        torch.cuda.empty_cache()

    # GQA test at long context
    print()
    print("--- GQA (h_q=32, h_kv=8) ---")
    for sq in [4096, 16384]:
        b = 1
        q = torch.randn(b, sq, 32, hd, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(b, sq, 8, hd, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(b, sq, 8, hd, dtype=torch.bfloat16, device="cuda")
        scale = 1.0 / math.sqrt(hd)
        flops = calc_flops(b, sq, sq, 32, hd, causal)

        row = {"shape": f"GQA4_s{sq//1024}k"}
        line = f"{'GQA4_s'+str(sq//1024)+'k':<18}"

        for name, fn in backends.items():
            try:
                if name == "SDPA (PyTorch)":
                    # SDPA needs broadcasted KV
                    kt = k.transpose(1, 2).repeat_interleave(4, dim=1)
                    vt = v.transpose(1, 2).repeat_interleave(4, dim=1)
                    t = bench(lambda: torch.nn.functional.scaled_dot_product_attention(
                        q.transpose(1, 2), kt, vt, is_causal=causal, scale=scale))
                else:
                    t = bench(lambda: fn(q, k, v, causal, scale))
                tf = flops / t / 1e12
                row[name] = round(tf, 1)
                line += f" {tf:>18.1f}TF"
            except Exception as e:
                row[name] = 0
                line += f" {'OOM/ERR':>20}"

        fa4 = row.get("FA3/FA4 (CK v3 ASM)", 0)
        fa2 = row.get("FA2 (CK tile)", 0)
        line += f" {fa4/fa2:.2f}x" if fa2 > 0 else " N/A"
        print(line)
        all_results.append(row)

        del q, k, v
        torch.cuda.empty_cache()

    # Summary
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    for r in all_results:
        s = r["shape"]
        fa4 = r.get("FA3/FA4 (CK v3 ASM)", 0)
        fa2 = r.get("FA2 (CK tile)", 0)
        sdpa = r.get("SDPA (PyTorch)", 0)
        print(f"  {s:<18}: FA4={fa4:>6.1f} TF, FA2={fa2:>6.1f} TF, SDPA={sdpa:>6.1f} TF")

    out_path = "/workspace/fa4_rocm/benchmarks/fa_long_context_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
