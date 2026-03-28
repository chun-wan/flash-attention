"""
Performance benchmark for FA4 ROCm flash attention kernels.

Measures TFLOPS and compares against:
  1. PyTorch SDPA (torch.nn.functional.scaled_dot_product_attention)
  2. ROCm FlashAttention-2 CK backend (if installed)
  3. Aiter Triton backend (if installed)

Standard benchmark shapes cover common LLM serving workloads:
  - Prefill: batch=2, seqlen=2048/4096/8192, heads=32, hdim=128
  - Decode: batch=64, seqlen_q=1, seqlen_k=2048/4096/8192

Usage:
    python bench_fa4_rocm.py                   # full benchmark
    python bench_fa4_rocm.py --prefill-only    # prefill shapes only
    python bench_fa4_rocm.py --decode-only     # decode shapes only
    python bench_fa4_rocm.py --quick           # minimal shapes for quick check
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Callable

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

NUM_WARMUP = 20
NUM_ITERS = 100


@dataclass
class BenchShape:
    batch: int
    seqlen_q: int
    seqlen_k: int
    num_heads_q: int
    num_heads_k: int
    head_dim: int
    causal: bool
    dtype: torch.dtype = torch.bfloat16

    @property
    def flops(self) -> int:
        """Forward-pass FLOPs: 4 * B * Sq * Sk * H * D (Q@K^T + P@V, both fwd)."""
        effective_causal_factor = 0.5 if self.causal else 1.0
        return int(4 * self.batch * self.seqlen_q * self.seqlen_k
                   * self.num_heads_q * self.head_dim * effective_causal_factor)

    @property
    def name(self) -> str:
        gqa = f"gqa{self.num_heads_q // self.num_heads_k}" if self.num_heads_q != self.num_heads_k else "mha"
        c = "causal" if self.causal else ""
        dt = "bf16" if self.dtype == torch.bfloat16 else "fp16"
        return f"b{self.batch}_sq{self.seqlen_q}_sk{self.seqlen_k}_h{self.num_heads_q}_d{self.head_dim}_{gqa}_{c}_{dt}"

    @property
    def short_name(self) -> str:
        return f"b{self.batch} sq{self.seqlen_q} sk{self.seqlen_k} h{self.num_heads_q} d{self.head_dim}"


@dataclass
class BenchResult:
    shape: BenchShape
    backend: str
    latency_us: float
    tflops: float
    memory_mb: float = 0.0
    error: str = ""


# Standard prefill shapes
PREFILL_SHAPES = [
    BenchShape(2, 2048, 2048, 32, 32, 128, True),
    BenchShape(2, 4096, 4096, 32, 32, 128, True),
    BenchShape(1, 8192, 8192, 32, 32, 128, True),
    BenchShape(4, 1024, 1024, 32, 32, 128, True),
    # GQA
    BenchShape(2, 2048, 2048, 32, 8, 128, True),
    # Non-causal
    BenchShape(2, 2048, 2048, 32, 32, 128, False),
    # Different head dims
    BenchShape(2, 2048, 2048, 32, 32, 64, True),
]

# Standard decode shapes
DECODE_SHAPES = [
    BenchShape(64, 1, 2048, 32, 32, 128, False),
    BenchShape(32, 1, 4096, 32, 32, 128, False),
    BenchShape(16, 1, 8192, 32, 32, 128, False),
    BenchShape(128, 1, 1024, 32, 32, 128, False),
    # GQA decode
    BenchShape(64, 1, 2048, 32, 8, 128, False),
]

# Quick check shapes
QUICK_SHAPES = [
    BenchShape(2, 2048, 2048, 32, 32, 128, True),
    BenchShape(64, 1, 2048, 32, 32, 128, False),
]


def benchmark_fn(fn, q, k, v, causal, softmax_scale, warmup=NUM_WARMUP, iters=NUM_ITERS):
    """Benchmark a function and return latency in microseconds."""
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        _ = fn(q, k, v, causal=causal, softmax_scale=softmax_scale)
    torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(iters):
        _ = fn(q, k, v, causal=causal, softmax_scale=softmax_scale)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters * 1e6  # microseconds


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    return torch.cuda.memory_allocated() / 1024 / 1024


def torch_sdpa_wrapper(q, k, v, causal=False, softmax_scale=None):
    """Wrap torch SDPA to match our API."""
    head_dim = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    batch, sq, hq, hd = q.shape
    hk = k.shape[2]

    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    if hq != hk:
        k_t = k_t.repeat_interleave(hq // hk, dim=1)
        v_t = v_t.repeat_interleave(hq // hk, dim=1)

    with torch.no_grad():
        out = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, is_causal=causal, scale=softmax_scale
        )
    return out.transpose(1, 2)


def load_backends() -> Dict[str, Callable]:
    """Discover and load available attention backends."""
    backends = {}

    # Always available: PyTorch SDPA
    backends["torch_sdpa"] = torch_sdpa_wrapper

    # FA4 ROCm (our kernel)
    try:
        from flash_attn_rocm import flash_attn_func as fa4_func
        backends["fa4_rocm"] = lambda q, k, v, causal=False, softmax_scale=None: \
            fa4_func(q, k, v, causal=causal, softmax_scale=softmax_scale)
        print("[OK] FA4 ROCm loaded")
    except ImportError as e:
        print(f"[--] FA4 ROCm not available: {e}")

    # ROCm Flash Attention 2 (CK backend)
    try:
        from flash_attn import flash_attn_func as fa2_func
        backends["flash_attn_v2_ck"] = lambda q, k, v, causal=False, softmax_scale=None: \
            fa2_func(q, k, v, causal=causal, softmax_scale=softmax_scale)
        print("[OK] FlashAttention-2 CK loaded")
    except ImportError:
        print("[--] FlashAttention-2 CK not available")

    # Aiter Triton backend
    try:
        from aiter import flash_attn_func as aiter_func
        backends["aiter_triton"] = lambda q, k, v, causal=False, softmax_scale=None: \
            aiter_func(q, k, v, causal=causal, softmax_scale=softmax_scale)
        print("[OK] Aiter Triton loaded")
    except ImportError:
        print("[--] Aiter Triton not available")

    return backends


def run_benchmark(shapes: List[BenchShape], backends: Dict[str, Callable]) -> List[BenchResult]:
    """Run benchmarks on all shapes with all backends."""
    results = []

    for shape in shapes:
        print(f"\n--- {shape.short_name} (causal={shape.causal}) ---")

        torch.manual_seed(42)
        q = torch.randn(shape.batch, shape.seqlen_q, shape.num_heads_q, shape.head_dim,
                         dtype=shape.dtype, device="cuda")
        k = torch.randn(shape.batch, shape.seqlen_k, shape.num_heads_k, shape.head_dim,
                         dtype=shape.dtype, device="cuda")
        v = torch.randn(shape.batch, shape.seqlen_k, shape.num_heads_k, shape.head_dim,
                         dtype=shape.dtype, device="cuda")
        softmax_scale = 1.0 / math.sqrt(shape.head_dim)

        for name, fn in backends.items():
            try:
                torch.cuda.reset_peak_memory_stats()
                mem_before = get_memory_mb()

                latency_us = benchmark_fn(fn, q, k, v, shape.causal, softmax_scale)
                tflops = shape.flops / (latency_us * 1e-6) / 1e12
                mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 - mem_before

                result = BenchResult(
                    shape=shape, backend=name,
                    latency_us=latency_us, tflops=tflops,
                    memory_mb=mem_peak,
                )
                results.append(result)
                print(f"  {name:25s}: {latency_us:10.1f} us  {tflops:8.2f} TFLOPS  {mem_peak:8.1f} MB")

            except Exception as e:
                result = BenchResult(
                    shape=shape, backend=name,
                    latency_us=float("inf"), tflops=0.0, error=str(e),
                )
                results.append(result)
                print(f"  {name:25s}: ERROR - {str(e)[:80]}")

        del q, k, v
        torch.cuda.empty_cache()

    return results


def print_summary(results: List[BenchResult]):
    """Print a summary table with speedup ratios."""
    print("\n" + "=" * 100)
    print("PERFORMANCE SUMMARY")
    print("=" * 100)

    # Group by shape
    shapes = list(dict.fromkeys(r.shape.short_name for r in results))
    backends = list(dict.fromkeys(r.backend for r in results))

    # Header
    header = f"{'Shape':40s}"
    for b in backends:
        header += f"  {b:>15s}"
    print(header)
    print("-" * len(header))

    for shape_name in shapes:
        shape_results = {r.backend: r for r in results if r.shape.short_name == shape_name}
        row = f"{shape_name:40s}"
        for b in backends:
            if b in shape_results and shape_results[b].tflops > 0:
                row += f"  {shape_results[b].tflops:12.2f} TF"
            else:
                row += f"  {'N/A':>15s}"
        print(row)

    # Speedup table (if fa4_rocm and a baseline exist)
    if "fa4_rocm" in backends and "torch_sdpa" in backends:
        print(f"\n{'Speedup vs torch_sdpa':40s}")
        print("-" * 60)
        for shape_name in shapes:
            shape_results = {r.backend: r for r in results if r.shape.short_name == shape_name}
            if "fa4_rocm" in shape_results and "torch_sdpa" in shape_results:
                fa4_tflops = shape_results["fa4_rocm"].tflops
                sdpa_tflops = shape_results["torch_sdpa"].tflops
                if sdpa_tflops > 0 and fa4_tflops > 0:
                    speedup = fa4_tflops / sdpa_tflops
                    print(f"  {shape_name:38s}  {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="FA4 ROCm Performance Benchmark")
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP)
    parser.add_argument("--iters", type=int, default=NUM_ITERS)
    args = parser.parse_args()

    global NUM_WARMUP, NUM_ITERS
    NUM_WARMUP = args.warmup
    NUM_ITERS = args.iters

    if not torch.cuda.is_available():
        print("No GPU available. Exiting.")
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    print(f"ROCm: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Warmup: {NUM_WARMUP}, Iterations: {NUM_ITERS}")

    backends = load_backends()
    print(f"\nActive backends: {list(backends.keys())}")

    if args.quick:
        shapes = QUICK_SHAPES
    elif args.prefill_only:
        shapes = PREFILL_SHAPES
    elif args.decode_only:
        shapes = DECODE_SHAPES
    else:
        shapes = PREFILL_SHAPES + DECODE_SHAPES

    results = run_benchmark(shapes, backends)
    print_summary(results)


if __name__ == "__main__":
    main()
