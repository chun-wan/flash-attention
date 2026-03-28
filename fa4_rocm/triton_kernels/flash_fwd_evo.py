"""
Parameterized FA4 Triton kernel for AVO evolution.
All optimization knobs exposed as constexpr parameters.
"""
import math
from typing import Optional
import torch
import triton
import triton.language as tl


@triton.jit
def _evo_fwd_inner_unmasked(
    acc, l_i, m_i, q,
    K_ptrs, V_ptrs, stride_kn, stride_vn,
    sm_scale_log2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    lo, hi,
):
    offs_n = tl.arange(0, BLOCK_N)
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_ptrs + start_n * stride_kn)
        if PRE_LOAD_V:
            v = tl.load(V_ptrs + start_n * stride_vn)
        qk = tl.dot(q, k) * sm_scale_log2
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = tl.load(V_ptrs + start_n * stride_vn)
        acc += tl.dot(p.to(v.type.element_ty), v)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _evo_fwd_inner_masked(
    acc, l_i, m_i, q,
    K_ptrs, V_ptrs, stride_kn, stride_vn,
    seqlen_k, offs_m,
    sm_scale_log2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    lo, hi,
):
    offs_n = tl.arange(0, BLOCK_N)
    for start_n in range(lo, hi, BLOCK_N):
        kv_offs = start_n + offs_n
        k = tl.load(K_ptrs + start_n * stride_kn)
        if PRE_LOAD_V:
            v = tl.load(V_ptrs + start_n * stride_vn)
        qk = tl.dot(q, k) * sm_scale_log2
        mask = (offs_m[:, None] >= kv_offs[None, :]) & (kv_offs[None, :] < seqlen_k)
        qk = tl.where(mask, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        q_shifted = tl.where(m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None])
        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)
        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        alpha = tl.math.exp2(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = tl.load(V_ptrs + start_n * stride_vn)
        acc += tl.dot(p.to(v.type.element_ty), v)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _evo_fwd_kernel(
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
    PRE_LOAD_V: tl.constexpr,
    PRE_SCALE_Q: tl.constexpr,
):
    sm_scale_log2 = sm_scale * 1.4426950408889634

    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // num_heads_q
    off_hq = off_bh % num_heads_q
    off_hk = off_hq // (num_heads_q // num_heads_k)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + off_b * stride_qb + off_hq * stride_qh + \
             (offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd)
    k_ptrs = K + off_b * stride_kb + off_hk * stride_kh + \
             (offs_d[:, None] * stride_kd + offs_n[None, :] * stride_ks)
    v_ptrs = V + off_b * stride_vb + off_hk * stride_vh + \
             (offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)
    o_ptrs = O + off_b * stride_ob + off_hq * stride_oh + \
             (offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)

    if PRE_SCALE_Q:
        q = (q * sm_scale_log2).to(q.type.element_ty)
        sm_scale_log2_inner = 1.0
    else:
        sm_scale_log2_inner = sm_scale_log2

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + float("-inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    if CAUSAL:
        unmasked_end = start_m * BLOCK_M
        unmasked_blocks = (unmasked_end // BLOCK_N) * BLOCK_N
        if unmasked_blocks > 0:
            acc, l_i, m_i = _evo_fwd_inner_unmasked(
                acc, l_i, m_i, q, k_ptrs, v_ptrs, stride_ks, stride_vs,
                sm_scale_log2_inner, BLOCK_M, BLOCK_N, HEAD_DIM, PRE_LOAD_V,
                0, unmasked_blocks)
        causal_hi = tl.minimum(seqlen_k, (start_m + 1) * BLOCK_M)
        masked_hi = ((causal_hi + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
        acc, l_i, m_i = _evo_fwd_inner_masked(
            acc, l_i, m_i, q, k_ptrs, v_ptrs, stride_ks, stride_vs,
            seqlen_k, offs_m, sm_scale_log2_inner,
            BLOCK_M, BLOCK_N, HEAD_DIM, PRE_LOAD_V,
            unmasked_blocks, masked_hi)
    else:
        total_hi = tl.cdiv(seqlen_k, BLOCK_N) * BLOCK_N
        acc, l_i, m_i = _evo_fwd_inner_unmasked(
            acc, l_i, m_i, q, k_ptrs, v_ptrs, stride_ks, stride_vs,
            sm_scale_log2_inner, BLOCK_M, BLOCK_N, HEAD_DIM, PRE_LOAD_V,
            0, total_hi)

    acc = acc / l_i[:, None]
    tl.store(o_ptrs, acc.to(q.type.element_ty), mask=offs_m[:, None] < seqlen_q)


def evo_flash_attn(q, k, v, causal=False, softmax_scale=None,
                   BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=1,
                   waves_per_eu=2, PRE_LOAD_V=False, PRE_SCALE_Q=False):
    """Parameterized FA4 kernel for AVO evolution."""
    assert q.dim() == 4
    batch, sq, hq, hd = q.shape
    sk = k.shape[1]
    hk = k.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(hd)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o = torch.empty_like(q)
    grid = (triton.cdiv(sq, BLOCK_M), batch * hq)
    _evo_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        sq, sk, hq, hk, softmax_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=hd,
        CAUSAL=causal, PRE_LOAD_V=PRE_LOAD_V, PRE_SCALE_Q=PRE_SCALE_Q,
        num_warps=num_warps, num_stages=num_stages, waves_per_eu=waves_per_eu,
    )
    return o
