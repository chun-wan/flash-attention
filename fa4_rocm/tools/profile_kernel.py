#!/usr/bin/env python3
"""
FA4 ROCm Kernel Profiler using rocprofv3.

Profiles the Triton and HIP kernels, collects hardware counters,
and identifies bottlenecks (MFMA utilization, LDS stalls, VMEM stalls).

Usage:
    python tools/profile_kernel.py                    # profile all
    python tools/profile_kernel.py --backend triton   # profile triton only
    python tools/profile_kernel.py --counters         # detailed counter collection
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_bench_script(backend: str, shape: dict, output_file: str) -> str:
    """Generate a Python script that runs the kernel for profiling."""
    return f'''
import torch, math, sys, time
sys.path.insert(0, "{Path(__file__).parent.parent}")

torch.manual_seed(42)
device = "cuda:0"
dtype = torch.bfloat16

b, sq, sk, hq, hk, hd = {shape["b"]}, {shape["sq"]}, {shape["sk"]}, {shape["hq"]}, {shape["hk"]}, {shape["hd"]}
causal = {shape["causal"]}
scale = 1.0 / math.sqrt(hd)

q = torch.randn(b, sq, hq, hd, dtype=dtype, device=device)
k = torch.randn(b, sk, hk, hd, dtype=dtype, device=device)
v = torch.randn(b, sk, hk, hd, dtype=dtype, device=device)

if "{backend}" == "triton":
    from triton_kernels.flash_fwd_triton import flash_attn_triton_func as fn
elif "{backend}" == "hip":
    from flash_attn_rocm import flash_attn_func
    fn = lambda q,k,v,causal=False,softmax_scale=None: flash_attn_func(q,k,v,causal=causal,softmax_scale=softmax_scale,backend="hip")
else:
    raise ValueError(f"Unknown backend: {backend}")

# Warmup
for _ in range(5):
    _ = fn(q, k, v, causal=causal, softmax_scale=scale)
torch.cuda.synchronize()

# Profiled run
for _ in range(10):
    _ = fn(q, k, v, causal=causal, softmax_scale=scale)
torch.cuda.synchronize()
'''


def run_rocprofv3(backend: str, shape: dict):
    """Run rocprofv3 with kernel trace."""
    script = create_bench_script(backend, shape, "")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
        f.write(script)
        script_path = f.name

    output_dir = f"/tmp/fa4_profile_{backend}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "rocprofv3",
        "--kernel-trace",
        "-o", f"{output_dir}/trace",
        "--", sys.executable, script_path,
    ]

    print(f"\n{'='*60}")
    print(f"Profiling {backend} with rocprofv3...")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = "0"

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    os.unlink(script_path)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:300]}")
        return

    # Parse kernel trace
    trace_file = f"{output_dir}/trace_kernel_trace.csv"
    if os.path.exists(trace_file):
        print(f"\n  Kernel trace ({trace_file}):")
        with open(trace_file) as f:
            lines = f.readlines()
            if len(lines) > 1:
                header = lines[0].strip().split(',')
                for line in lines[1:6]:
                    fields = line.strip().split(',')
                    if len(fields) >= 4:
                        kernel_name = fields[-1] if len(fields) > 4 else fields[3]
                        duration = int(fields[2]) - int(fields[1]) if len(fields) >= 3 else 0
                        print(f"    {kernel_name[:60]:60s} {duration/1000:.1f} us")
    else:
        print(f"  No kernel trace found at {trace_file}")
        for f in Path(output_dir).glob("*"):
            print(f"    Found: {f}")

    return output_dir


def run_rocprof_compute(backend: str, shape: dict):
    """Run rocprof-compute for roofline analysis."""
    script = create_bench_script(backend, shape, "")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
        f.write(script)
        script_path = f.name

    output_name = f"fa4_{backend}"
    print(f"\n{'='*60}")
    print(f"Profiling {backend} with rocprof-compute...")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = "0"

    # Profile
    cmd_profile = [
        "rocprof-compute", "profile",
        "-n", output_name,
        "--", sys.executable, script_path,
    ]
    result = subprocess.run(cmd_profile, capture_output=True, text=True,
                            timeout=300, env=env, cwd="/tmp")
    os.unlink(script_path)

    if result.returncode != 0:
        print(f"  Profile ERROR: {result.stderr[:300]}")
        return

    # Analyze
    workload_dir = f"/tmp/{output_name}"
    if os.path.isdir(workload_dir):
        cmd_analyze = [
            "rocprof-compute", "analyze",
            "-p", workload_dir,
        ]
        result = subprocess.run(cmd_analyze, capture_output=True, text=True,
                                timeout=120, env=env)
        if result.returncode == 0:
            print(result.stdout[:2000])
        else:
            print(f"  Analyze ERROR: {result.stderr[:300]}")


def quick_benchmark(backend: str, shape: dict):
    """Quick TFLOPS measurement without external profiling tools."""
    import torch
    device = "cuda:0"
    dtype = torch.bfloat16

    q = torch.randn(shape["b"], shape["sq"], shape["hq"], shape["hd"],
                     dtype=dtype, device=device)
    k = torch.randn(shape["b"], shape["sk"], shape["hk"], shape["hd"],
                     dtype=dtype, device=device)
    v = torch.randn(shape["b"], shape["sk"], shape["hk"], shape["hd"],
                     dtype=dtype, device=device)
    scale = 1.0 / math.sqrt(shape["hd"])

    if backend == "triton":
        from triton_kernels.flash_fwd_triton import flash_attn_triton_func as fn
    elif backend == "hip":
        from flash_attn_rocm import flash_attn_func
        fn = lambda q,k,v,causal=False,softmax_scale=None: \
            flash_attn_func(q,k,v,causal=causal,softmax_scale=softmax_scale,backend="hip")

    cf = 0.5 if shape["causal"] else 1.0
    flops = 4 * shape["b"] * shape["sq"] * shape["sk"] * shape["hq"] * shape["hd"] * cf

    # Warmup
    torch.cuda.synchronize()
    for _ in range(20):
        fn(q, k, v, causal=shape["causal"], softmax_scale=scale)
    torch.cuda.synchronize()

    # Timed
    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(q, k, v, causal=shape["causal"], softmax_scale=scale)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tflops = flops / (elapsed / iters) / 1e12
    latency_us = elapsed / iters * 1e6
    print(f"  {backend:10s}: {tflops:.1f} TFLOPS  {latency_us:.1f} us")
    return tflops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["triton", "hip", "all"], default="all")
    parser.add_argument("--counters", action="store_true")
    parser.add_argument("--rocprof-compute", action="store_true")
    args = parser.parse_args()

    shape = {"b": 2, "sq": 2048, "sk": 2048, "hq": 32, "hk": 32, "hd": 128, "causal": True}

    backends = ["triton", "hip"] if args.backend == "all" else [args.backend]

    print("="*60)
    print("FA4 ROCm Kernel Profiling")
    print(f"Shape: b={shape['b']} sq={shape['sq']} hq={shape['hq']} hd={shape['hd']} causal={shape['causal']}")
    print("="*60)

    # Quick TFLOPS
    print("\nQuick TFLOPS measurement:")
    for b in backends:
        try:
            quick_benchmark(b, shape)
        except Exception as e:
            print(f"  {b}: ERROR {e}")

    # rocprofv3 trace
    for b in backends:
        try:
            run_rocprofv3(b, shape)
        except Exception as e:
            print(f"  {b} rocprofv3 ERROR: {e}")

    if args.rocprof_compute:
        for b in backends:
            try:
                run_rocprof_compute(b, shape)
            except Exception as e:
                print(f"  {b} rocprof-compute ERROR: {e}")


if __name__ == "__main__":
    main()
