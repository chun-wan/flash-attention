#!/usr/bin/env python3
"""
AVO Rolling Kernel Evolution v2: 20 rounds with Triton + FlyDSL.
Includes profiling via rocprofv3 for each round.
"""
import json, math, os, sys, time, subprocess, csv, glob
import torch
sys.path.insert(0, "/workspace/fa4_rocm")

DEVICE = "cuda:0"
SHAPE = (2, 2048, 32, 128)
DTYPE = torch.bfloat16
WARMUP, ITERS = 20, 100
SCALE = 1.0 / math.sqrt(128)
FLOPS = 4 * 2 * 2048 * 2048 * 32 * 128 * 0.5

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

def profile_kernel(fn, q, k, v, round_name):
    """Quick profile: run under rocprofv3, extract kernel trace."""
    script = f"""
import torch, math, sys
sys.path.insert(0, "/workspace/fa4_rocm")
from triton_kernels.flash_fwd_evo import evo_flash_attn
q = torch.randn(2,2048,32,128, dtype=torch.bfloat16, device="cuda:0")
k = torch.randn(2,2048,32,128, dtype=torch.bfloat16, device="cuda:0")
v = torch.randn(2,2048,32,128, dtype=torch.bfloat16, device="cuda:0")
# warmup
for _ in range(3): evo_flash_attn(q,k,v,causal=True,softmax_scale={SCALE})
torch.cuda.synchronize()
# profiled
for _ in range(3): evo_flash_attn(q,k,v,causal=True,softmax_scale={SCALE})
torch.cuda.synchronize()
"""
    script_path = f"/tmp/prof_{round_name}.py"
    with open(script_path, "w") as f:
        f.write(script)

    db_path = f"/tmp/prof_{round_name}"
    try:
        result = subprocess.run(
            ["rocprofv3", "--kernel-trace", "-o", db_path, "--", "python3", script_path],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "HIP_VISIBLE_DEVICES": "0"}
        )
        # Parse SQLite for kernel info
        import sqlite3
        db_file = f"{db_path}_results.db"
        if os.path.exists(db_file):
            db = sqlite3.connect(db_file)
            cur = db.cursor()
            # Find kernel symbol table
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%kernel_symbol%'")
            sym_table = cur.fetchone()
            if sym_table:
                cur.execute(f"SELECT kernel_name, arch_vgpr_count, accum_vgpr_count, sgpr_count, group_segment_size FROM {sym_table[0]} WHERE kernel_name LIKE '%evo%' OR kernel_name LIKE '%fa4%'")
                for row in cur.fetchall():
                    return {"vgpr": row[1], "agpr": row[2], "sgpr": row[3], "lds": row[4]}
            db.close()
    except Exception:
        pass
    return {}


# Define 20 rounds: mix of Triton configs
ROUNDS = [
    # Round 1-5: Tile tuning
    {"name": "R01_baseline_BN64", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": False, "PRE_SCALE_Q": False},
    {"name": "R02_BN32_stg2", "BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": False, "PRE_SCALE_Q": False},
    {"name": "R03_preload_v", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": False},
    {"name": "R04_BN64_stg2", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": False, "PRE_SCALE_Q": False},
    {"name": "R05_stg2+preload", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": False},
    # Round 6-10: Occupancy/warp tuning
    {"name": "R06_BM64_BN64", "BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": False, "PRE_SCALE_Q": False},
    {"name": "R07_BM64_preload", "BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": False},
    {"name": "R08_BM64_stg2", "BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": False, "PRE_SCALE_Q": False},
    {"name": "R09_BM64_stg2_pre", "BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": False},
    {"name": "R10_warps8_BN64", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 8, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": False, "PRE_SCALE_Q": False},
    # Round 11-15: Wave/stage combos
    {"name": "R11_waves1_stg2", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 1, "PRE_LOAD_V": True, "PRE_SCALE_Q": False},
    {"name": "R12_BN32_preload", "BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": False},
    {"name": "R13_BN128", "BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 4, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": False, "PRE_SCALE_Q": False},
    {"name": "R14_BM256_BN64", "BLOCK_M": 256, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1,
     "waves_per_eu": 2, "PRE_LOAD_V": False, "PRE_SCALE_Q": False},
    {"name": "R15_BM64_BN32_stg2", "BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": False},
    # Round 16-20: Best-of combos + fine tuning
    {"name": "R16_best+waves1", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 1, "PRE_LOAD_V": True, "PRE_SCALE_Q": False},
    {"name": "R17_best+waves3", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 3, "PRE_LOAD_V": True, "PRE_SCALE_Q": False},
    {"name": "R18_BN48_stg2", "BLOCK_M": 128, "BLOCK_N": 48, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": False},
    {"name": "R19_warps2_stg2", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 2, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": False},
    {"name": "R20_FINAL_BEST", "BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2,
     "waves_per_eu": 2, "PRE_LOAD_V": True, "PRE_SCALE_Q": False},
]


def run_evolution():
    from triton_kernels.flash_fwd_evo import evo_flash_attn

    torch.manual_seed(42)
    b, sq, hq, hd = SHAPE
    q = torch.randn(b, sq, hq, hd, dtype=DTYPE, device=DEVICE)
    k = torch.randn(b, sq, hq, hd, dtype=DTYPE, device=DEVICE)
    v = torch.randn(b, sq, hq, hd, dtype=DTYPE, device=DEVICE)
    ref = get_ref(q, k, v)

    # Baseline
    from aiter import flash_attn_func as aiter_fa
    aiter_tf, aiter_lat = benchmark(aiter_fa, q, k, v)

    results = []
    best_tf = 0
    best_name = "none"
    best_cfg = {}

    print("=" * 120)
    print(f"AVO ROLLING EVOLUTION v2 | Baseline: aiter FA2 = {aiter_tf:.0f} TF ({aiter_lat:.0f} us)")
    print("=" * 120)
    print(f"{'#':>3s} {'Name':<24s} {'TF':>6s} {'us':>7s} {'Err':>8s} {'OK':>3s} {'%FA3':>6s} {'VGPRs':>6s} {'Best':>5s} {'Config'}")
    print("-" * 120)

    for i, cfg in enumerate(ROUNDS):
        name = cfg.pop("name")
        valid_keys = {"BLOCK_M", "BLOCK_N", "num_warps", "num_stages",
                       "waves_per_eu", "PRE_LOAD_V", "PRE_SCALE_Q"}
        evo_cfg = {k: v for k, v in cfg.items() if k in valid_keys}

        fn = lambda q, k, v, causal=True, softmax_scale=None, _c=evo_cfg: \
            evo_flash_attn(q, k, v, causal=causal, softmax_scale=softmax_scale, **_c)

        try:
            ok, err = check_correct(fn, q, k, v, ref)
            if ok:
                tf, lat = benchmark(fn, q, k, v)
                # Quick profile for VGPRs
                prof = profile_kernel(fn, q, k, v, f"r{i+1:02d}")
                vgpr = prof.get("vgpr", "?")
            else:
                tf, lat, vgpr = 0.0, 99999.0, "?"
        except Exception as e:
            ok, err, tf, lat, vgpr = False, -1.0, 0.0, 99999.0, "?"

        is_best = tf > best_tf and ok
        if is_best:
            best_tf = tf
            best_name = name
            best_cfg = evo_cfg.copy()

        ratio = f"{tf/aiter_tf:.0%}" if tf > 0 else "N/A"
        star = " *" if is_best else "  "
        cfg_str = f"BM={evo_cfg.get('BLOCK_M',128)} BN={evo_cfg.get('BLOCK_N',64)} w={evo_cfg.get('num_warps',4)} stg={evo_cfg.get('num_stages',1)} pre={evo_cfg.get('PRE_LOAD_V',False)}"
        err_s = f"{err:.4f}" if err >= 0 else "ERR"
        ok_s = "Y" if ok else "N"

        print(f"{i+1:>3d} {name:<24s} {tf:>6.0f} {lat:>7.0f} {err_s:>8s} {ok_s:>3s} {ratio:>6s} {str(vgpr):>6s} {star:>5s} {cfg_str}")

        results.append({
            "round": i + 1, "name": name, "tflops": round(tf, 1),
            "latency_us": round(lat, 1), "max_err": round(err, 6) if err >= 0 else None,
            "correct": ok, "is_best": is_best, "vgpr": vgpr, "config": evo_cfg,
        })
        cfg["name"] = name

    print("=" * 120)
    print(f"\nBEST: {best_name} = {best_tf:.0f} TF ({best_tf/aiter_tf:.0%} of aiter)")
    print(f"Config: {best_cfg}")

    # Save
    log_path = "/workspace/fa4_rocm/tools/evolution_v2_log.json"
    with open(log_path, "w") as f:
        json.dump({"baseline_aiter_tf": round(aiter_tf, 1), "rounds": results,
                    "best_name": best_name, "best_tf": round(best_tf, 1),
                    "best_cfg": best_cfg}, f, indent=2)
    print(f"Saved to {log_path}")

    # Trajectory
    print("\nTrajectory:")
    rb = 0
    for r in results:
        if r["correct"] and r["tflops"] > rb:
            rb = r["tflops"]
        bar = "#" * int(r["tflops"] / 10) if r["correct"] else "x" * 3
        m = " <-- BEST" if r["is_best"] else ""
        print(f"  R{r['round']:02d} {r['tflops']:>6.0f} TF  {bar}{m}")


if __name__ == "__main__":
    run_evolution()
