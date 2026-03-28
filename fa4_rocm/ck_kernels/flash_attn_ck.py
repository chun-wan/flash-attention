"""
CK (Composable Kernel) backend for flash attention.

Two modes:
  1. "aiter" -- uses aiter's pre-built CK FMHA (fastest, includes v3 ASM path)
  2. "ck_ext" -- uses our standalone CK FMHA torch extension (for custom tiles)
"""

import math
import os
from typing import Optional, Tuple

import torch
from torch import Tensor

_aiter_available = None
_ck_ext = None


def _check_aiter():
    global _aiter_available
    if _aiter_available is not None:
        return _aiter_available
    try:
        from aiter import flash_attn_func as _  # noqa: F401
        _aiter_available = True
    except ImportError:
        _aiter_available = False
    return _aiter_available


def _get_ck_ext():
    """Load the standalone CK FMHA torch extension (JIT compiled)."""
    global _ck_ext
    if _ck_ext is not None:
        return _ck_ext
    try:
        import ck_fmha_ext
        _ck_ext = ck_fmha_ext
    except ImportError:
        raise ImportError(
            "CK FMHA extension not built. Run: "
            "cd ck_kernels && python build_ck_fmha.py"
        )
    return _ck_ext


def ck_flash_attn_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    mode: str = "aiter",
) -> Tensor | Tuple[Tensor, Tensor]:
    """
    CK-backed flash attention forward.

    Args:
        q: (batch, seqlen_q, nheads_q, hdim)
        k: (batch, seqlen_k, nheads_k, hdim)
        v: (batch, seqlen_k, nheads_k, hdim)
        causal: apply causal mask
        softmax_scale: defaults to 1/sqrt(hdim)
        return_lse: return log-sum-exp
        mode: "aiter" (pre-built, fast) or "ck_ext" (standalone build)
    """
    head_dim = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if mode == "aiter":
        return _aiter_flash_attn(q, k, v, causal, softmax_scale, return_lse)
    elif mode == "ck_ext":
        return _ck_ext_flash_attn(q, k, v, causal, softmax_scale, return_lse)
    else:
        raise ValueError(f"Unknown CK mode: {mode}")


def _aiter_flash_attn(q, k, v, causal, softmax_scale, return_lse):
    """Use aiter's flash_attn_func which dispatches v3 ASM -> CK tile."""
    if not _check_aiter():
        raise ImportError("aiter not installed. Use mode='ck_ext' instead.")

    from aiter import flash_attn_func as aiter_fa

    result = aiter_fa(
        q, k, v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        return_lse=return_lse,
        return_attn_probs=False,
    )

    if return_lse:
        if isinstance(result, tuple):
            return result[0], result[1]
        return result, None
    if isinstance(result, tuple):
        return result[0]
    return result


def _ck_ext_flash_attn(q, k, v, causal, softmax_scale, return_lse):
    """Use our standalone CK FMHA torch extension."""
    ext = _get_ck_ext()
    o, lse = ext.flash_attn_fwd(q, k, v, softmax_scale, causal, return_lse)
    if return_lse:
        return o, lse
    return o


def ck_flash_attn_varlen_func(
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
    """CK-backed variable-length flash attention (via aiter)."""
    if not _check_aiter():
        raise ImportError("aiter not installed; varlen requires aiter")

    from aiter import flash_attn_varlen_func as aiter_varlen

    head_dim = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    out, _, _, _ = aiter_varlen(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    return out
