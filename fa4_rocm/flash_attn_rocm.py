"""
FA4 ROCm -- FlashAttention-4 for AMD GPUs (gfx942 / MI300X)

Python interface matching the FA4 CuTeDSL API:
    flash_attn_func(q, k, v, causal=False, softmax_scale=None)

Uses JIT-compiled HIP kernels via torch.utils.cpp_extension.
"""

import os
import math
from typing import Optional, Tuple
from pathlib import Path

import torch
from torch import Tensor

_KERNELS_DIR = Path(__file__).parent / "kernels"
_ext = None


def _get_ext():
    """JIT compile and load the HIP extension."""
    global _ext
    if _ext is not None:
        return _ext

    from torch.utils.cpp_extension import load

    sources = [
        str(_KERNELS_DIR / "flash_attn_combined.hip"),
    ]

    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags = [
        "-O3",
        "--offload-arch=gfx942",
        "-std=c++17",
        "-DHEAD_DIM=128",
        "-DBLOCK_M=64",
        "-DBLOCK_N=64",
        "-mcode-object-version=5",
    ]

    _ext = load(
        name="fa4_rocm_ops",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=bool(os.environ.get("FA4_ROCM_VERBOSE", "")),
    )
    return _ext


def flash_attn_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    backend: str = "ck",
) -> Tensor | Tuple[Tensor, Tensor]:
    """
    Scaled dot-product attention using FlashAttention-4 algorithm on AMD ROCm.

    Computes: softmax(Q @ K^T * scale) @ V

    Args:
        q: Query tensor, shape (batch, seqlen_q, num_heads_q, head_dim), fp16/bf16.
        k: Key tensor, shape (batch, seqlen_k, num_heads_k, head_dim), fp16/bf16.
        v: Value tensor, shape (batch, seqlen_k, num_heads_k, head_dim), fp16/bf16.
        causal: If True, apply causal attention mask (lower-triangular).
        softmax_scale: Scaling factor for QK^T. Defaults to 1/sqrt(head_dim).
        return_lse: If True, also return the log-sum-exp per row.
        backend: "ck" (CK/aiter, fastest), "triton", or "hip" (MFMA C++ kernel).

    Returns:
        out: Attention output, shape (batch, seqlen_q, num_heads_q, head_dim).
        lse: (optional) Log-sum-exp, shape (batch, num_heads_q, seqlen_q).

    Supports GQA/MQA: num_heads_q must be divisible by num_heads_k.
    """
    assert q.dim() == 4, f"Expected 4D q, got {q.dim()}D"
    assert k.dim() == 4, f"Expected 4D k, got {k.dim()}D"
    assert v.dim() == 4, f"Expected 4D v, got {v.dim()}D"
    assert q.dtype in (torch.float16, torch.bfloat16), f"Unsupported dtype {q.dtype}"

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    head_dim = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if backend == "ck":
        from .ck_kernels.flash_attn_ck import ck_flash_attn_func
        return ck_flash_attn_func(
            q, k, v, causal=causal, softmax_scale=softmax_scale,
            return_lse=return_lse, mode="aiter",
        )

    if backend == "triton":
        from .triton_kernels.flash_fwd_triton import flash_attn_triton_func
        return flash_attn_triton_func(q, k, v, causal=causal, softmax_scale=softmax_scale)

    if backend == "hip":
        ext = _get_ext()
        results = ext.flash_attn_fwd(q, k, v, softmax_scale, causal, return_lse)
        if return_lse:
            return results[0], results[1]
        return results[0]

    raise ValueError(f"Unknown backend: {backend!r}. Use 'ck', 'triton', or 'hip'.")


def flash_attn_varlen_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> Tensor:
    """
    Variable-length flash attention (packed sequences).

    Currently dispatches by padding to max_seqlen and calling the fixed-length kernel.
    A dedicated varlen kernel is planned for a future iteration.

    Args:
        q: (total_q, num_heads_q, head_dim)
        k: (total_k, num_heads_k, head_dim)
        v: (total_k, num_heads_k, head_dim)
        cu_seqlens_q: (batch+1,) cumulative sequence lengths for Q
        cu_seqlens_k: (batch+1,) cumulative sequence lengths for K
        max_seqlen_q: maximum Q sequence length
        max_seqlen_k: maximum K sequence length
    """
    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads_q = q.shape[1]
    num_heads_k = k.shape[1]
    head_dim = q.shape[2]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # Pad into (batch, max_seqlen, heads, hdim) tensors
    q_padded = torch.zeros(batch_size, max_seqlen_q, num_heads_q, head_dim,
                           dtype=q.dtype, device=q.device)
    k_padded = torch.zeros(batch_size, max_seqlen_k, num_heads_k, head_dim,
                           dtype=k.dtype, device=k.device)
    v_padded = torch.zeros(batch_size, max_seqlen_k, num_heads_k, head_dim,
                           dtype=v.dtype, device=v.device)

    for i in range(batch_size):
        sq = cu_seqlens_q[i+1] - cu_seqlens_q[i]
        sk = cu_seqlens_k[i+1] - cu_seqlens_k[i]
        q_padded[i, :sq] = q[cu_seqlens_q[i]:cu_seqlens_q[i+1]]
        k_padded[i, :sk] = k[cu_seqlens_k[i]:cu_seqlens_k[i+1]]
        v_padded[i, :sk] = v[cu_seqlens_k[i]:cu_seqlens_k[i+1]]

    out = flash_attn_func(q_padded, k_padded, v_padded, causal=causal,
                          softmax_scale=softmax_scale)

    # Unpad results
    total_q = q.shape[0]
    result = torch.empty(total_q, num_heads_q, head_dim, dtype=q.dtype, device=q.device)
    for i in range(batch_size):
        sq = cu_seqlens_q[i+1] - cu_seqlens_q[i]
        result[cu_seqlens_q[i]:cu_seqlens_q[i+1]] = out[i, :sq]

    return result
