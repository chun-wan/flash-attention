"""FA4 ROCm -- FlashAttention-4 for AMD GPUs"""

from .flash_attn_rocm import flash_attn_func, flash_attn_varlen_func

__all__ = ["flash_attn_func", "flash_attn_varlen_func"]
__version__ = "0.1.0"
