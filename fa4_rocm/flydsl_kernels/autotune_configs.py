"""
Autotune configurations for FA4 ROCm FlyDSL kernels.

Defines tile size search space and autotuning harness using FlyDSL's
autotune module for finding optimal kernel configurations on the target GPU.

Tile sizes affect:
  - BLOCK_M: query rows per workgroup (determines parallelism over seqlen_q)
  - BLOCK_N: key columns per iteration (determines inner loop granularity)
  - NUM_WARPS: waves per workgroup (occupancy vs register pressure tradeoff)
  - KV_BLOCK_SIZE: for decode, tokens per iteration

Constraints (gfx942):
  - LDS limit: 64KB per CU
  - BLOCK_M * HEAD_DIM * 2 + BLOCK_N * HEAD_DIM * 2 * 2 <= 65536
  - For HEAD_DIM=128: BLOCK_M*256 + BLOCK_N*512 <= 65536
  - Register pressure: ~256 VGPRs per wave, 4 waves ideal occupancy
"""

from dataclasses import dataclass, field
from typing import List, Optional

HEAD_DIM_DEFAULT = 128
LDS_LIMIT_BYTES = 65536
WARP_SIZE = 64


@dataclass
class FlashFwdConfig:
    """Configuration for the flash attention forward (prefill) kernel."""
    block_m: int = 64
    block_n: int = 64
    num_warps: int = 4
    head_dim: int = HEAD_DIM_DEFAULT

    @property
    def num_threads(self) -> int:
        return self.num_warps * WARP_SIZE

    @property
    def lds_bytes(self) -> int:
        elem_bytes = 2  # fp16/bf16
        lds_q = self.block_m * self.head_dim * elem_bytes
        lds_k = self.block_n * self.head_dim * elem_bytes
        lds_v = self.block_n * self.head_dim * elem_bytes
        return lds_q + lds_k + lds_v

    def is_valid(self) -> bool:
        if self.lds_bytes > LDS_LIMIT_BYTES:
            return False
        if self.block_m % 16 != 0 or self.block_n % 16 != 0:
            return False
        if self.head_dim % 16 != 0:
            return False
        if self.num_warps < 1 or self.num_warps > 16:
            return False
        # Total elements in Q/O tile must be evenly distributable across threads
        if (self.block_m * self.head_dim) % self.num_threads != 0:
            return False
        if (self.block_n * self.head_dim) % self.num_threads != 0:
            return False
        return True


@dataclass
class FlashDecodeConfig:
    """Configuration for the flash attention decode (single-query) kernel."""
    kv_block_size: int = 64
    num_warps: int = 4
    num_partitions: int = 8
    head_dim: int = HEAD_DIM_DEFAULT

    @property
    def num_threads(self) -> int:
        return self.num_warps * WARP_SIZE

    @property
    def lds_bytes(self) -> int:
        elem_bytes = 2
        lds_q = self.head_dim * elem_bytes
        lds_k = self.kv_block_size * self.head_dim * elem_bytes
        lds_v = self.kv_block_size * self.head_dim * elem_bytes
        lds_scores = self.kv_block_size * 4
        return lds_q + lds_k + lds_v + lds_scores

    def is_valid(self) -> bool:
        if self.lds_bytes > LDS_LIMIT_BYTES:
            return False
        if self.kv_block_size % 16 != 0:
            return False
        if self.num_partitions < 1:
            return False
        return True


# Prefill configs to search (ordered by expected performance)
PREFILL_CONFIGS: List[FlashFwdConfig] = [
    # Conservative: fits easily in LDS, good occupancy
    FlashFwdConfig(block_m=64,  block_n=64,  num_warps=4, head_dim=128),
    # Larger M: better Q reuse
    FlashFwdConfig(block_m=128, block_n=64,  num_warps=4, head_dim=128),
    # Larger N: fewer K/V load iterations
    FlashFwdConfig(block_m=64,  block_n=128, num_warps=4, head_dim=128),
    # More warps: higher occupancy
    FlashFwdConfig(block_m=64,  block_n=64,  num_warps=8, head_dim=128),
    FlashFwdConfig(block_m=128, block_n=64,  num_warps=8, head_dim=128),
    # Smaller tiles for short sequences
    FlashFwdConfig(block_m=32,  block_n=64,  num_warps=4, head_dim=128),
    FlashFwdConfig(block_m=32,  block_n=32,  num_warps=4, head_dim=128),
    # For head_dim=64
    FlashFwdConfig(block_m=128, block_n=128, num_warps=4, head_dim=64),
    FlashFwdConfig(block_m=128, block_n=64,  num_warps=4, head_dim=64),
    FlashFwdConfig(block_m=64,  block_n=64,  num_warps=4, head_dim=64),
    # For head_dim=96
    FlashFwdConfig(block_m=128, block_n=64,  num_warps=4, head_dim=96),
    FlashFwdConfig(block_m=64,  block_n=64,  num_warps=4, head_dim=96),
]

# Decode configs to search
DECODE_CONFIGS: List[FlashDecodeConfig] = [
    FlashDecodeConfig(kv_block_size=64,  num_warps=4, num_partitions=8),
    FlashDecodeConfig(kv_block_size=128, num_warps=4, num_partitions=8),
    FlashDecodeConfig(kv_block_size=64,  num_warps=4, num_partitions=16),
    FlashDecodeConfig(kv_block_size=64,  num_warps=8, num_partitions=8),
    FlashDecodeConfig(kv_block_size=256, num_warps=4, num_partitions=4),
    FlashDecodeConfig(kv_block_size=64,  num_warps=4, num_partitions=32),
]

# Filter to valid configs
PREFILL_CONFIGS = [c for c in PREFILL_CONFIGS if c.is_valid()]
DECODE_CONFIGS = [c for c in DECODE_CONFIGS if c.is_valid()]


def get_best_prefill_config(
    seqlen_q: int,
    seqlen_k: int,
    head_dim: int,
    num_heads: int,
) -> FlashFwdConfig:
    """
    Select the best prefill config based on problem size heuristics.
    For full autotuning, use FlyDSL's @autotune decorator at compile time.
    """
    candidates = [c for c in PREFILL_CONFIGS if c.head_dim == head_dim]
    if not candidates:
        candidates = [c for c in PREFILL_CONFIGS if c.head_dim == HEAD_DIM_DEFAULT]

    # Heuristic: for short sequences, use smaller tiles; for long, larger
    if seqlen_q <= 128:
        candidates.sort(key=lambda c: c.block_m)
    elif seqlen_q >= 4096:
        candidates.sort(key=lambda c: -c.block_m)
    else:
        candidates.sort(key=lambda c: -(c.block_m * c.block_n))

    return candidates[0]


def get_best_decode_config(
    seqlen_k: int,
    num_heads: int,
) -> FlashDecodeConfig:
    """
    Select the best decode config based on KV length heuristics.
    """
    if seqlen_k > 16384:
        return FlashDecodeConfig(kv_block_size=64, num_warps=4, num_partitions=32)
    elif seqlen_k > 4096:
        return FlashDecodeConfig(kv_block_size=64, num_warps=4, num_partitions=16)
    elif seqlen_k > 1024:
        return FlashDecodeConfig(kv_block_size=64, num_warps=4, num_partitions=8)
    else:
        return FlashDecodeConfig(kv_block_size=64, num_warps=4, num_partitions=4)


def print_config_info():
    """Print all valid configurations and their LDS usage."""
    print("=== Prefill Configs ===")
    for c in PREFILL_CONFIGS:
        print(f"  BLOCK_M={c.block_m:3d}  BLOCK_N={c.block_n:3d}  "
              f"NUM_WARPS={c.num_warps}  HEAD_DIM={c.head_dim:3d}  "
              f"LDS={c.lds_bytes:5d}B  threads={c.num_threads:3d}")

    print("\n=== Decode Configs ===")
    for c in DECODE_CONFIGS:
        print(f"  KV_BLOCK={c.kv_block_size:3d}  NUM_WARPS={c.num_warps}  "
              f"PARTITIONS={c.num_partitions:2d}  "
              f"LDS={c.lds_bytes:5d}B  threads={c.num_threads:3d}")


if __name__ == "__main__":
    print_config_info()
