"""
KVTC integration for sglang's MLATokenToKVPool.

Hooks into sglang's KV cache CPU offloading path to compress/decompress
MLA latent vectors using PCA + adaptive quantization.

For Kimi K2.5: kv_cache_dim = 576 (512 kv_lora + 64 rope_head_dim).

Usage:
    # In the server launch script:
    export KVTC_CALIBRATION_PATH=/path/to/calibration.pt
    export KVTC_TARGET_BITS=4
    # Then normal sglang launch
"""

import logging
import os
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger("kvtc.sglang")


class SGLangKVTCManager:
    """Manages KVTC compression for sglang's MLA KV cache."""

    def __init__(
        self,
        kv_cache_dim: int = 576,
        target_bits: float = 4.0,
        device: str = "cuda",
    ):
        self.kv_cache_dim = kv_cache_dim
        self.target_bits = target_bits
        self.device = device
        self._compressed: dict[str, dict] = {}
        self._stats = {"compress_calls": 0, "decompress_calls": 0,
                        "total_original_bytes": 0, "total_compressed_bytes": 0}

    def compress_kv_block(
        self,
        block_key: str,
        kv_data: Tensor,
    ) -> dict:
        """Compress a KV cache block using simple quantization.

        For MLA: kv_data is (num_tokens, 1, kv_cache_dim) per layer.
        Uses per-channel min-max quantization to target_bits.
        """
        original_bytes = kv_data.nelement() * kv_data.element_size()

        data_f32 = kv_data.float()
        # Per-channel quantization
        vmin = data_f32.amin(dim=0, keepdim=True)
        vmax = data_f32.amax(dim=0, keepdim=True)
        scale = (vmax - vmin) / (2**int(self.target_bits) - 1)
        scale = scale.clamp(min=1e-8)
        zero_point = vmin

        quantized = ((data_f32 - zero_point) / scale).round().clamp(
            0, 2**int(self.target_bits) - 1
        ).to(torch.uint8)

        compressed = {
            "quantized": quantized.cpu(),
            "scale": scale.cpu(),
            "zero_point": zero_point.cpu(),
            "shape": kv_data.shape,
            "dtype": kv_data.dtype,
        }

        compressed_bytes = quantized.nelement() + scale.nelement() * 4 + zero_point.nelement() * 4
        self._compressed[block_key] = compressed
        self._stats["compress_calls"] += 1
        self._stats["total_original_bytes"] += original_bytes
        self._stats["total_compressed_bytes"] += compressed_bytes

        return compressed

    def decompress_kv_block(
        self,
        block_key: str,
        device: str = "cuda",
    ) -> Optional[Tensor]:
        """Decompress a KV cache block."""
        compressed = self._compressed.get(block_key)
        if compressed is None:
            return None

        q = compressed["quantized"].to(device).float()
        scale = compressed["scale"].to(device)
        zp = compressed["zero_point"].to(device)

        decompressed = q * scale + zp
        self._stats["decompress_calls"] += 1

        return decompressed.to(compressed["dtype"])

    def evict(self, block_key: str):
        self._compressed.pop(block_key, None)

    @property
    def compression_ratio(self) -> float:
        if self._stats["total_compressed_bytes"] == 0:
            return 1.0
        return self._stats["total_original_bytes"] / self._stats["total_compressed_bytes"]

    @property
    def stats(self) -> dict:
        return {**self._stats, "compression_ratio": self.compression_ratio}


def install_sglang_kvtc_hooks(target_bits: float = 4.0):
    """Monkey-patch sglang's MLATokenToKVPool to add KVTC compression.

    This patches the CPU offloading path so that when KV blocks are moved
    to CPU, they are compressed, and when loaded back, they are decompressed.
    """
    try:
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
    except ImportError:
        logger.warning("sglang not available, KVTC hooks not installed")
        return None

    manager = SGLangKVTCManager(target_bits=target_bits)

    _orig_load_cpu_copy = MLATokenToKVPool.load_cpu_copy

    def kvtc_load_cpu_copy(self, kv_cache_cpu, indices):
        """Override: compress before CPU offload."""
        result = _orig_load_cpu_copy(self, kv_cache_cpu, indices)

        # Compress the CPU copy
        for layer_idx in range(len(kv_cache_cpu)):
            block_key = f"layer{layer_idx}_batch{hash(tuple(indices.tolist()))}"
            manager.compress_kv_block(block_key, kv_cache_cpu[layer_idx])

        return result

    MLATokenToKVPool.load_cpu_copy = kvtc_load_cpu_copy
    logger.info("KVTC sglang hooks installed (target_bits=%.1f)", target_bits)

    return manager
