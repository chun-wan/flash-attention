# FA4 ROCm -- FlashAttention-4 for AMD GPUs

Port of FlashAttention-4 to AMD ROCm (MI300X / gfx942), with three implementation tiers:

1. **C++/HIP kernel** (`kernels/flash_fwd_gfx942.hip`) -- tiled attention forward pass using MFMA intrinsics
2. **FlyDSL kernels** (`flydsl_kernels/`) -- prefill, decode, and combine kernels using FlyDSL's layout algebra
3. **AVO optimization** -- evolutionary assembly-level optimization via the AFTT AVO agent

## Quick Start

```bash
# Install (JIT compilation on first use)
cd fa4_rocm
pip install -e .

# Python API
from fa4_rocm import flash_attn_func
out = flash_attn_func(q, k, v, causal=True)
```

## Requirements

- AMD GPU: MI300X (gfx942) or MI350 (gfx950)
- ROCm 6.x or 7.x
- PyTorch 2.2+ with ROCm support
- For FlyDSL kernels: FlyDSL installed (`pip install -e /FlyDSL`)

## Testing

```bash
# Correctness tests
pytest tests/test_correctness.py -v

# Performance benchmark
python benchmarks/bench_fa4_rocm.py

# Quick benchmark (2 shapes only)
python benchmarks/bench_fa4_rocm.py --quick
```

## Architecture

### C++/HIP Kernel

The core kernel in `kernels/flash_fwd_gfx942.hip` implements:

- Tiled FlashAttention loop with online softmax
- BLOCK_M=64, BLOCK_N=64, HEAD_DIM=128 (configurable at compile time)
- 4 warps (256 threads) per workgroup
- LDS (shared memory) tiling for Q, K, V tiles
- Supports BF16/FP16, causal/non-causal, MHA/GQA/MQA

### FlyDSL Kernels

- `flash_fwd_flydsl.py` -- Full prefill kernel using FlyDSL layout algebra
- `flash_decode_flydsl.py` -- Decode (single-query) with FlashDecoding split-KV
- `flash_combine_flydsl.py` -- Reduce kernel for combining split results
- `autotune_configs.py` -- Tile size search space and heuristic selection

### AVO Integration

The AVO agent (`HipblasLtAnalyzeTool/tools/`) can evolve the compiled assembly:

```bash
# Extract assembly from compiled kernel
hipcc -save-temps --offload-arch=gfx942 kernels/flash_fwd_gfx942.hip

# Import into AVO workspace
python HipblasLtAnalyzeTool/tools/run_avo.py init \
    --workspace /tmp/fa4_avo

# Run evolutionary optimization
python HipblasLtAnalyzeTool/tools/run_avo.py evolve \
    --kernel flash_fwd_bf16_hdim128 \
    --max-iterations 100
```

## Supported Features

| Feature | Status |
|---------|--------|
| BF16 forward | Done |
| FP16 forward | Done |
| Causal mask | Done |
| MHA | Done |
| GQA/MQA | Done |
| Head dim 64/96/128 | Done |
| Variable length | Pad-based |
| FlashDecoding (split-KV) | Done (FlyDSL) |
| Backward pass | Planned |
| FP8 | Planned |
| Paged KV cache | Planned |
