#!/usr/bin/env python3
"""
AVO Rolling Kernel Evolution: 20-round iterative optimization.
Each round: apply optimization -> correctness check -> benchmark -> log.
"""
import json, math, os, sys, time, torch
sys.path.insert(0, "/workspace/fa4_rocm")

DEVICE = "cuda:0"
SHAPE = (2, 2048, 32, 128)  # batch, seqlen, heads, hdim
DTYPE = torch.bfloat16
WARMUP, ITERS = 20, 100
SCALE = 1.0 / math.sqrt(128)
FLOPS = 4 * 2 * 2048 * 2048 * 32 * 128 * 0.5  # causal

def get_ref(q, k, v):
    qt = q.transpose(1, 2); kt = k.transpose(1, 2); vt = v.transpose(1, 2)
    return torch.nn.functional.scaled_dot_product_attention(
        qt, kt, vt, is_causal=True, scale=SCALE).transpose(1, 2)

def check_correct(fn, q, k, v, ref, atol=0.01):
    out = fn(q, k, v, causal=True, softmax_scale=SCALE)
    err = (out.float() - ref.float()).abs().max().item()
    return err < atol, err

def benchmark(fn, q, k, v):
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        fn(q, k, v, causal=True, softmax_scale=SCALE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn(q, k, v, causal=True, softmax_scale=SCALE)
    torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / ITERS * 1e6
    return FLOPS / (lat * 1e-6) / 1e12, lat

# Define all 20 rounds
ROUNDS = [
    # Round 1-5: Tile/config tuning
    {"name": "R01_BN32_stages2", "BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": False, "PRE_SCALE_Q": False},
    {"name": "R02_BM64", "BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": False, "PRE_SCALE_Q": False},
    {"name": "R03_BN128", "BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 4, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": False, "PRE_SCALE_Q": False},
    {"name": "R04_warps8", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 8, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": False, "PRE_SCALE_Q": False},
    {"name": "R05_preload_v", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": False},
    # Round 6-10: Algorithm optimizations
    {"name": "R06_prescale_q", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": False, "PRE_SCALE_Q": True},
    {"name": "R07_preload+prescale", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": True},
    {"name": "R08_BN32_preload", "BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": False},
    {"name": "R09_BN32_prescale", "BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": False, "PRE_SCALE_Q": True},
    {"name": "R10_BN32_both", "BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": True},
    # Round 11-15: Memory/occupancy optimizations
    {"name": "R11_waves1", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1,
     "waves_per_eu": 1, "PRE_LOAD_V": False, "PRE_SCALE_Q": False},
    {"name": "R12_waves3", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1,
     "waves_per_eu": 3, "PRE_LOAD_Q": False, "PRE_LOAD_V": False, "PRE_SCALE_Q": False},
    {"name": "R13_BM64_preload", "BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": False},
    {"name": "R14_BM64_prescale", "BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": False, "PRE_SCALE_Q": True},
    {"name": "R15_BM64_both", "BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": True},
    # Round 16-20: Combined best configs
    {"name": "R16_BN32_stages2_w8", "BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 8, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": False, "PRE_SCALE_Q": False},
    {"name": "R17_BM64_BN32_stg2", "BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": True},
    {"name": "R18_BN64_stg2", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": True},
    {"name": "R19_BN64_w8_preload", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 8, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": True},
    {"name": "R20_best_combo", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": False},
]


def run_evolution():
    from triton_kernels.flash_fwd_evo import evo_flash_attn

    torch.manual_seed(42)
    q = torch.randn(*SHAPE, dtype=DTYPE, device=DEVICE)
    k = torch.randn(SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[3], dtype=DTYPE, device=DEVICE)
    v = torch.randn(SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[3], dtype=DTYPE, device=DEVICE)
    ref = get_ref(q, k, v)

    # Baseline: aiter FA2
    from aiter import flash_attn_func as aiter_fa
    aiter_tf, aiter_lat = benchmark(aiter_fa, q, k, v)
    print(f"Baseline aiter FA2: {aiter_tf:.0f} TF ({aiter_lat:.0f} us)")
    print()

    results = []
    best_tf = 0
    best_name = "none"

    header = f"{'Round':<28s} {'TF':>6s} {'us':>7s} {'Err':>8s} {'OK':>4s} {'vs FA3':>7s} {'Best':>6s} {'Config'}"
    print("=" * 110)
    print("AVO ROLLING KERNEL EVOLUTION: 20 ROUNDS")
    print("=" * 110)
    print(header)
    print("-" * 110)

    for i, cfg in enumerate(ROUNDS):
        name = cfg.pop("name")
        # Filter valid kwargs for evo_flash_attn
        valid_keys = {"BLOCK_M", "BLOCK_N", "num_warps", "num_stages",
                       "waves_per_eu", "PRE_LOAD_V", "PRE_SCALE_Q"}
        evo_cfg = {k: v for k, v in cfg.items() if k in valid_keys}

        fn = lambda q, k, v, causal=True, softmax_scale=None: \
            evo_flash_attn(q, k, v, causal=causal, softmax_scale=softmax_scale, **evo_cfg)

        try:
            ok, err = check_correct(fn, q, k, v, ref)
            if ok:
                tf, lat = benchmark(fn, q, k, v)
            else:
                tf, lat = 0.0, 99999.0
        except Exception as e:
            ok, err, tf, lat = False, -1.0, 0.0, 99999.0

        is_best = tf > best_tf and ok
        if is_best:
            best_tf = tf
            best_name = name

        ratio = f"{tf/aiter_tf:.0%}" if tf > 0 else "N/A"
        star = " *" if is_best else "  "
        cfg_str = " ".join(f"{k}={v}" for k, v in evo_cfg.items())
        err_s = f"{err:.4f}" if err >= 0 else "ERR"
        ok_s = "Y" if ok else "N"

        print(f"{name:<28s} {tf:>6.0f} {lat:>7.0f} {err_s:>8s} {ok_s:>4s} {ratio:>7s} {star:>6s} {cfg_str}")

        results.append({
            "round": i + 1, "name": name, "tflops": round(tf, 1),
            "latency_us": round(lat, 1), "max_err": round(err, 6) if err >= 0 else None,
            "correct": ok, "is_best": is_best, "config": evo_cfg,
        })
        cfg["name"] = name  # restore

    print("=" * 110)
    print(f"\nBest: {best_name} = {best_tf:.0f} TF ({best_tf/aiter_tf:.0%} of aiter FA2 {aiter_tf:.0f} TF)")

    # Save results
    log_path = "/workspace/fa4_rocm/tools/evolution_log.json"
    with open(log_path, "w") as f:
        json.dump({"baseline_aiter_tf": round(aiter_tf, 1), "rounds": results,
                    "best_name": best_name, "best_tf": round(best_tf, 1)}, f, indent=2)
    print(f"Results saved to {log_path}")

    # Print evolution trajectory
    print("\nEvolution trajectory:")
    running_best = 0
    for r in results:
        if r["correct"] and r["tflops"] > running_best:
            running_best = r["tflops"]
        bar = "#" * int(r["tflops"] / 10) if r["correct"] else "x" * 3
        mark = " <-- NEW BEST" if r["is_best"] else ""
        print(f"  R{r['round']:02d} {r['tflops']:>6.0f} TF  {bar}{mark}")


if __name__ == "__main__":
    run_evolution()
