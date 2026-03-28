"""
FA4 ROCm Triton Forward Attention Kernel for gfx942.

FlashAttention-2 with online softmax, causal masking, GQA.
Config: BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=1 (gfx942 optimal).
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fa4_fwd_kernel(
    Q, K, V, O,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_os, stride_oh, stride_od,
    seqlen_q, seqlen_k,
    num_heads_q, num_heads_k,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634

    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // num_heads_q
    off_hq = off_bh % num_heads_q
    off_hk = off_hq // (num_heads_q // num_heads_k)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Q pointers: [BLOCK_M, HEAD_DIM]
    q_ptrs = Q + off_b * stride_qb + off_hq * stride_qh + \
             (offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd)
    # K pointers: [HEAD_DIM, BLOCK_N] (transposed for Q @ K^T)
    k_ptrs = K + off_b * stride_kb + off_hk * stride_kh + \
             (offs_d[:, None] * stride_kd + offs_n[None, :] * stride_ks)
    # V pointers: [BLOCK_N, HEAD_DIM]
    v_ptrs = V + off_b * stride_vb + off_hk * stride_vh + \
             (offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)
    # O pointers: [BLOCK_M, HEAD_DIM]
    o_ptrs = O + off_b * stride_ob + off_hq * stride_oh + \
             (offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)

    # Load Q once
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + float("-inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Determine KV block range
    if CAUSAL:
        hi = tl.minimum(seqlen_k, (start_m + 1) * BLOCK_M)
    else:
        hi = seqlen_k
    num_kv_blocks = tl.cdiv(hi, BLOCK_N)

    # Main loop over KV blocks
    for block_n_idx in range(0, num_kv_blocks):
        start_n = block_n_idx * BLOCK_N
        kv_offs = start_n + offs_n

        # Load K^T: [HEAD_DIM, BLOCK_N]
        k = tl.load(k_ptrs + start_n * stride_ks)

        # S = Q @ K^T: [BLOCK_M, BLOCK_N]
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        # Causal mask
        if CAUSAL:
            causal_mask = offs_m[:, None] >= kv_offs[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Boundary mask
        boundary_mask = kv_offs[None, :] < seqlen_k
        qk = tl.where(boundary_mask, qk, float("-inf"))

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        q_shifted = tl.where(m_ij[:, None] == float("-inf"),
                             float("-inf"), qk - m_ij[:, None])
        p = tl.math.exp2(q_shifted * RCP_LN2)
        l_ij = tl.sum(p, 1)

        # Rescale accumulator
        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        alpha = tl.math.exp2(m_diff * RCP_LN2)
        acc = acc * alpha[:, None]

        # Load V: [BLOCK_N, HEAD_DIM]
        v = tl.load(v_ptrs + start_n * stride_vs)

        # Accumulate P @ V
        acc += tl.dot(p.to(v.type.element_ty), v)

        # Update softmax state
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Normalize and store
    acc = acc / l_i[:, None]
    out = acc.to(q.type.element_ty)
    tl.store(o_ptrs, out, mask=offs_m[:, None] < seqlen_q)


def flash_attn_triton_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Flash Attention forward using Triton kernel.

    Args:
        q: [batch, seqlen_q, num_heads_q, head_dim]
        k: [batch, seqlen_k, num_heads_k, head_dim]
        v: [batch, seqlen_k, num_heads_k, head_dim]
    """
    assert q.dim() == 4
    batch, seqlen_q_val, num_heads_q, head_dim = q.shape
    seqlen_k_val = k.shape[1]
    num_heads_k = k.shape[2]
    assert num_heads_q % num_heads_k == 0

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)

    BLOCK_M = 128
    BLOCK_N = 64

    grid = (triton.cdiv(seqlen_q_val, BLOCK_M), batch * num_heads_q)

    _fa4_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        seqlen_q_val, seqlen_k_val,
        num_heads_q, num_heads_k,
        softmax_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=head_dim,
        CAUSAL=causal,
        num_warps=4,
        num_stages=1,
    )

    return o
