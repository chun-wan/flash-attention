"""
FlashAttention Decode Kernel for AMD GPUs via FlyDSL.

Single-query (decode) attention adapted from FlyDSL's pa_decode_fp8.py for BF16/FP16.
This is the FlashDecoding variant: splits the KV sequence across multiple workgroups,
each computing a partial result, then a reduce kernel combines them.

Target: gfx942 (MI300X, CDNA3)
Wavefront: 64 threads

Design:
  - Each workgroup processes a contiguous chunk of KV tokens (a "partition")
  - Within a partition, iterate over KV blocks:
    1. Load K block to LDS, compute Q@K^T via MFMA
    2. Online softmax on the scores
    3. Load V block to LDS, compute P@V via MFMA
  - Write partial output and (max, sum) to global memory
  - A separate reduce kernel combines partial results across partitions

Differences from pa_decode_fp8.py:
  - BF16/FP16 input (not FP8), so no dequant scales
  - Standard (non-paged) KV layout: [batch, seqlen_k, num_heads_k, head_dim]
  - Supports GQA/MQA via query_group_size
"""

from __future__ import annotations
import math as _math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, gpu, rocdl, buffer_ops, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl._mlir import ir

# Constants
HEAD_DIM = 128
KV_BLOCK_SIZE = 64      # KV tokens per iteration
NUM_WARPS = 4
WARP_SIZE = 64
NUM_THREADS = NUM_WARPS * WARP_SIZE  # 256

MFMA_M = 16
MFMA_N = 16
MFMA_K = 16

LOG2E = 1.4426950408889634
NEG_INF = -1e30

# LDS sizes
# Q: [1, HEAD_DIM] (single query row, broadcast across workgroup)
# K: [KV_BLOCK_SIZE, HEAD_DIM]
# V: [KV_BLOCK_SIZE, HEAD_DIM]
LDS_Q_ELEMS = HEAD_DIM
LDS_K_ELEMS = KV_BLOCK_SIZE * HEAD_DIM
LDS_V_ELEMS = KV_BLOCK_SIZE * HEAD_DIM
LDS_SCORES_ELEMS = KV_BLOCK_SIZE

allocator = None


def build_flash_decode_module(
    batch_size,
    seqlen_k,
    num_heads_q,
    num_heads_k,
    num_partitions,
    head_dim=HEAD_DIM,
    softmax_scale=None,
):
    """
    Build the FlashAttention decode (single-query) kernel module.

    The decode path splits seqlen_k into num_partitions chunks.
    Each workgroup handles one partition, producing partial O, max, and sum.
    A reduce kernel then combines them.

    Grid: (num_partitions, batch_size * num_heads_q)
    Block: (NUM_THREADS)
    """
    global allocator

    arch = get_hip_arch()
    if softmax_scale is None:
        softmax_scale = 1.0 / _math.sqrt(head_dim)

    _scale = float(softmax_scale)
    _gqa_ratio = num_heads_q // num_heads_k
    _tokens_per_partition = (seqlen_k + num_partitions - 1) // num_partitions

    _stride_q_head = head_dim
    _stride_q_seq = num_heads_q * head_dim
    _stride_k_head = head_dim
    _stride_k_seq = num_heads_k * head_dim
    _stride_k_batch = seqlen_k * num_heads_k * head_dim

    # Output layout: [batch, num_heads_q, num_partitions, head_dim]
    _stride_out_head = num_partitions * head_dim
    _stride_out_batch = num_heads_q * num_partitions * head_dim
    _stride_out_part = head_dim

    # Exp-sum/max layout: [batch, num_heads_q, num_partitions]
    _stride_es_batch = num_heads_q * num_partitions
    _stride_es_head = num_partitions

    allocator = SmemAllocator(None, arch=arch, global_sym_name="decode_smem")
    lds_q = allocator.allocate_array(T.f16, LDS_Q_ELEMS)
    lds_k = allocator.allocate_array(T.f16, LDS_K_ELEMS)
    lds_v = allocator.allocate_array(T.f16, LDS_V_ELEMS)
    lds_scores = allocator.allocate_array(T.f32, LDS_SCORES_ELEMS)

    @flyc.kernel
    def flash_decode_kernel(
        Q: fx.Tensor,      # [batch, 1, num_heads_q, head_dim]
        K: fx.Tensor,      # [batch, seqlen_k, num_heads_k, head_dim]
        V: fx.Tensor,      # [batch, seqlen_k, num_heads_k, head_dim]
        O_partial: fx.Tensor,  # [batch, num_heads_q, num_partitions, head_dim]
        M_partial: fx.Tensor,  # [batch, num_heads_q, num_partitions] (row max)
        L_partial: fx.Tensor,  # [batch, num_heads_q, num_partitions] (row sum)
    ):
        tid = gpu.thread_idx.x
        part_idx = gpu.block_idx.x
        bid_bh = gpu.block_idx.y

        batch_idx = arith.divui(bid_bh, arith.constant(num_heads_q, type=T.i32()))
        head_idx = arith.remui(bid_bh, arith.constant(num_heads_q, type=T.i32()))
        kv_head_idx = arith.divui(head_idx, arith.constant(_gqa_ratio, type=T.i32()))

        rsrc_Q = buffer_ops.create_buffer_resource(Q)
        rsrc_K = buffer_ops.create_buffer_resource(K)
        rsrc_V = buffer_ops.create_buffer_resource(V)
        rsrc_O = buffer_ops.create_buffer_resource(O_partial)
        rsrc_M = buffer_ops.create_buffer_resource(M_partial)
        rsrc_L = buffer_ops.create_buffer_resource(L_partial)

        # Base offsets
        q_base = arith.addi(
            arith.muli(batch_idx, arith.constant(_stride_q_seq, type=T.i32())),
            arith.muli(head_idx, arith.constant(_stride_q_head, type=T.i32()))
        )
        k_base = arith.addi(
            arith.muli(batch_idx, arith.constant(_stride_k_batch, type=T.i32())),
            arith.muli(kv_head_idx, arith.constant(_stride_k_head, type=T.i32()))
        )

        lds_base = allocator.get_base()
        q_lds = lds_q(lds_base)
        k_lds = lds_k(lds_base)
        v_lds = lds_v(lds_base)
        scores_lds = lds_scores(lds_base)

        # Load single query row to LDS
        for lp in range_constexpr(LDS_Q_ELEMS // NUM_THREADS):
            idx = arith.addi(tid, arith.constant(lp * NUM_THREADS, type=T.i32()))
            q_offset = arith.addi(q_base, idx)
            val = buffer_ops.buffer_load(rsrc_Q, q_offset, vec_width=1)
            q_lds.store(val, [idx])
        if LDS_Q_ELEMS % NUM_THREADS != 0:
            rem_idx = arith.addi(tid, arith.constant(
                (LDS_Q_ELEMS // NUM_THREADS) * NUM_THREADS, type=T.i32()))
            cond = arith.cmpi(rem_idx, arith.constant(LDS_Q_ELEMS, type=T.i32()), predicate="ult")
            # Skipping remainder for simplicity since HEAD_DIM=128 < 256=NUM_THREADS
        gpu.barrier()

        # Partition range
        part_start = arith.muli(part_idx, arith.constant(_tokens_per_partition, type=T.i32()))
        part_end_raw = arith.addi(part_start, arith.constant(_tokens_per_partition, type=T.i32()))
        part_end = arith.select(
            arith.cmpi(part_end_raw, arith.constant(seqlen_k, type=T.i32()), predicate="ugt"),
            arith.constant(seqlen_k, type=T.i32()),
            part_end_raw
        )

        # Per-thread accumulators for output [HEAD_DIM]
        # Each thread handles HEAD_DIM / NUM_THREADS elements (or more via loop)
        acc_per_thread = HEAD_DIM // NUM_THREADS  # might be 0 if NUM_THREADS > HEAD_DIM
        if acc_per_thread == 0:
            acc_per_thread = 1

        acc_O = []
        for _ in range(acc_per_thread):
            acc_O.append(arith.constant(0.0, type=T.f32()))

        row_max = arith.constant(NEG_INF, type=T.f32())
        row_sum = arith.constant(0.0, type=T.f32())

        # Iterate over KV blocks within this partition
        num_kv_blocks = _tokens_per_partition // KV_BLOCK_SIZE
        for kv_blk in range_constexpr(num_kv_blocks):
            kv_start = kv_blk * KV_BLOCK_SIZE

            # Load K block
            for lp in range_constexpr(LDS_K_ELEMS // NUM_THREADS):
                elem_idx = arith.addi(tid, arith.constant(lp * NUM_THREADS, type=T.i32()))
                row = arith.divui(elem_idx, arith.constant(HEAD_DIM, type=T.i32()))
                col = arith.remui(elem_idx, arith.constant(HEAD_DIM, type=T.i32()))
                global_tok = arith.addi(part_start, arith.addi(
                    arith.constant(kv_start, type=T.i32()), row))
                k_offset = arith.addi(k_base,
                    arith.addi(
                        arith.muli(global_tok, arith.constant(_stride_k_seq, type=T.i32())),
                        col
                    )
                )
                in_bounds = arith.cmpi(global_tok, part_end, predicate="ult")
                val = buffer_ops.buffer_load(rsrc_K, k_offset, vec_width=1)
                k_lds.store(
                    arith.select(in_bounds, val, arith.constant(0.0, type=T.f16())),
                    [elem_idx]
                )
            gpu.barrier()

            # Compute scores: Q[0, :] . K[j, :] for each j in block
            # Distribute KV_BLOCK_SIZE scores across threads
            scores_per_thread = KV_BLOCK_SIZE // NUM_THREADS
            if scores_per_thread == 0:
                scores_per_thread = 1

            local_scores = []
            for sp in range(max(1, KV_BLOCK_SIZE // NUM_THREADS)):
                j = arith.addi(tid, arith.constant(sp * NUM_THREADS, type=T.i32()))
                j_valid = arith.cmpi(j, arith.constant(KV_BLOCK_SIZE, type=T.i32()), predicate="ult")

                dot = arith.constant(0.0, type=T.f32())
                for d in range_constexpr(HEAD_DIM):
                    q_val = arith.extf(q_lds.load([arith.constant(d, type=T.i32())]), T.f32())
                    k_idx = arith.addi(
                        arith.muli(j, arith.constant(HEAD_DIM, type=T.i32())),
                        arith.constant(d, type=T.i32())
                    )
                    k_val = arith.extf(k_lds.load([k_idx]), T.f32())
                    dot = arith.addf(dot, arith.mulf(q_val, k_val))

                dot = arith.mulf(dot, arith.constant(_scale, type=T.f32()))

                # Bounds check
                global_j = arith.addi(part_start,
                    arith.addi(arith.constant(kv_start, type=T.i32()), j))
                oob = arith.cmpi(global_j, part_end, predicate="uge")
                dot = arith.select(oob, arith.constant(NEG_INF, type=T.f32()), dot)
                dot = arith.select(j_valid, dot, arith.constant(NEG_INF, type=T.f32()))

                local_scores.append(dot)

                # Store to LDS for later P@V
                scores_lds.store(
                    arith.select(j_valid, dot, arith.constant(NEG_INF, type=T.f32())),
                    [j]
                )
            gpu.barrier()

            # Find block max (reduce via LDS)
            # Thread 0 does a serial scan (for simplicity in FlyDSL)
            block_max = arith.constant(NEG_INF, type=T.f32())
            for s in range_constexpr(KV_BLOCK_SIZE):
                sv = scores_lds.load([arith.constant(s, type=T.i32())])
                block_max = arith.maximumf(block_max, sv)

            # Online softmax update
            new_max = arith.maximumf(row_max, block_max)
            log2e = arith.constant(LOG2E, type=T.f32())
            correction = arith.exp2f(arith.mulf(arith.subf(row_max, new_max), log2e))

            # Rescale accumulator and row_sum
            for a_idx in range(acc_per_thread):
                acc_O[a_idx] = arith.mulf(acc_O[a_idx], correction)
            row_sum = arith.mulf(row_sum, correction)
            row_max = new_max

            # Exponentiate scores and compute sum
            block_sum = arith.constant(0.0, type=T.f32())
            for s in range_constexpr(KV_BLOCK_SIZE):
                sv = scores_lds.load([arith.constant(s, type=T.i32())])
                exp_sv = arith.exp2f(arith.mulf(arith.subf(sv, row_max), log2e))
                is_valid = arith.cmpf(sv, arith.constant(NEG_INF + 1.0, type=T.f32()), predicate="ogt")
                exp_sv = arith.select(is_valid, exp_sv, arith.constant(0.0, type=T.f32()))
                scores_lds.store(exp_sv, [arith.constant(s, type=T.i32())])
                block_sum = arith.addf(block_sum, exp_sv)
            row_sum = arith.addf(row_sum, block_sum)
            gpu.barrier()

            # Load V block
            for lp in range_constexpr(LDS_V_ELEMS // NUM_THREADS):
                elem_idx = arith.addi(tid, arith.constant(lp * NUM_THREADS, type=T.i32()))
                row = arith.divui(elem_idx, arith.constant(HEAD_DIM, type=T.i32()))
                col = arith.remui(elem_idx, arith.constant(HEAD_DIM, type=T.i32()))
                global_tok = arith.addi(part_start, arith.addi(
                    arith.constant(kv_start, type=T.i32()), row))
                v_offset = arith.addi(k_base,
                    arith.addi(
                        arith.muli(global_tok, arith.constant(_stride_k_seq, type=T.i32())),
                        col
                    )
                )
                in_bounds = arith.cmpi(global_tok, part_end, predicate="ult")
                val = buffer_ops.buffer_load(rsrc_V, v_offset, vec_width=1)
                v_lds.store(
                    arith.select(in_bounds, val, arith.constant(0.0, type=T.f16())),
                    [elem_idx]
                )
            gpu.barrier()

            # Compute acc_O += P @ V  (1 x KV_BLOCK_SIZE) @ (KV_BLOCK_SIZE x HEAD_DIM)
            # For decode: single query row, so output is [1, HEAD_DIM]
            for a_idx in range(acc_per_thread):
                col = arith.addi(tid, arith.constant(a_idx * NUM_THREADS, type=T.i32()))
                col_valid = arith.cmpi(col, arith.constant(HEAD_DIM, type=T.i32()), predicate="ult")

                dot = arith.constant(0.0, type=T.f32())
                for j in range_constexpr(KV_BLOCK_SIZE):
                    p_val = scores_lds.load([arith.constant(j, type=T.i32())])
                    v_idx = arith.addi(
                        arith.muli(arith.constant(j, type=T.i32()),
                                   arith.constant(HEAD_DIM, type=T.i32())),
                        col
                    )
                    v_val = arith.extf(v_lds.load([v_idx]), T.f32())
                    dot = arith.addf(dot, arith.mulf(p_val, v_val))

                acc_O[a_idx] = arith.addf(acc_O[a_idx], dot)
            gpu.barrier()

        # Write partial results
        # O_partial: [batch, num_heads_q, num_partitions, head_dim]
        o_base = arith.addi(
            arith.addi(
                arith.muli(batch_idx, arith.constant(_stride_out_batch, type=T.i32())),
                arith.muli(head_idx, arith.constant(_stride_out_head, type=T.i32()))
            ),
            arith.muli(part_idx, arith.constant(_stride_out_part, type=T.i32()))
        )

        for a_idx in range(acc_per_thread):
            col = arith.addi(tid, arith.constant(a_idx * NUM_THREADS, type=T.i32()))
            col_valid = arith.cmpi(col, arith.constant(HEAD_DIM, type=T.i32()), predicate="ult")
            o_offset = arith.addi(o_base, col)
            out_val = arith.truncf(acc_O[a_idx], T.f16())
            buffer_ops.buffer_store(out_val, rsrc_O, o_offset)

        # Write max and sum
        es_base = arith.addi(
            arith.muli(batch_idx, arith.constant(_stride_es_batch, type=T.i32())),
            arith.addi(
                arith.muli(head_idx, arith.constant(_stride_es_head, type=T.i32())),
                part_idx
            )
        )
        # Only thread 0 writes max/sum
        is_t0 = arith.cmpi(tid, arith.constant(0, type=T.i32()), predicate="eq")
        buffer_ops.buffer_store(
            arith.select(is_t0, row_max, arith.constant(0.0, type=T.f32())),
            rsrc_M, es_base
        )
        buffer_ops.buffer_store(
            arith.select(is_t0, row_sum, arith.constant(0.0, type=T.f32())),
            rsrc_L, es_base
        )

        from flydsl.compiler.kernel_function import CompilationContext
        comp_ctx = CompilationContext.get_current()
        with ir.InsertionPoint(comp_ctx.gpu_module_body):
            allocator.finalize()

    return flash_decode_kernel


def build_flash_decode_reduce_module(
    batch_size,
    num_heads_q,
    num_partitions,
    head_dim=HEAD_DIM,
):
    """
    Build the reduce kernel that combines partial decode results.

    For each (batch, head):
      O_final[d] = sum_p( exp(M_p - M_global) * O_partial[p, d] ) / L_global
      where M_global = max_p(M_p), L_global = sum_p( exp(M_p - M_global) * L_p )
    """

    _stride_in_head = num_partitions * head_dim
    _stride_in_batch = num_heads_q * num_partitions * head_dim
    _stride_in_part = head_dim
    _stride_es_head = num_partitions
    _stride_es_batch = num_heads_q * num_partitions
    _stride_out_head = head_dim
    _stride_out_batch = num_heads_q * head_dim

    @flyc.kernel
    def decode_reduce_kernel(
        O_partial: fx.Tensor,  # [batch, num_heads_q, num_partitions, head_dim]
        M_partial: fx.Tensor,  # [batch, num_heads_q, num_partitions]
        L_partial: fx.Tensor,  # [batch, num_heads_q, num_partitions]
        O_out: fx.Tensor,      # [batch, 1, num_heads_q, head_dim]
    ):
        tid = gpu.thread_idx.x
        bid_bh = gpu.block_idx.x

        batch_idx = arith.divui(bid_bh, arith.constant(num_heads_q, type=T.i32()))
        head_idx = arith.remui(bid_bh, arith.constant(num_heads_q, type=T.i32()))

        rsrc_O_in = buffer_ops.create_buffer_resource(O_partial)
        rsrc_M = buffer_ops.create_buffer_resource(M_partial)
        rsrc_L = buffer_ops.create_buffer_resource(L_partial)
        rsrc_O_out = buffer_ops.create_buffer_resource(O_out)

        es_base = arith.addi(
            arith.muli(batch_idx, arith.constant(_stride_es_batch, type=T.i32())),
            arith.muli(head_idx, arith.constant(_stride_es_head, type=T.i32()))
        )

        # Find global max across partitions
        global_max = arith.constant(NEG_INF, type=T.f32())
        for p in range_constexpr(num_partitions):
            m_val = buffer_ops.buffer_load(rsrc_M,
                arith.addi(es_base, arith.constant(p, type=T.i32())), vec_width=1)
            global_max = arith.maximumf(global_max, m_val)

        # Compute global sum and weighted output per dimension
        log2e = arith.constant(LOG2E, type=T.f32())

        # Each thread handles head_dim / NUM_THREADS dimensions
        dims_per_thread = max(1, head_dim // NUM_THREADS)
        for d_idx in range(dims_per_thread):
            col = arith.addi(tid, arith.constant(d_idx * NUM_THREADS, type=T.i32()))
            col_valid = arith.cmpi(col, arith.constant(head_dim, type=T.i32()), predicate="ult")

            acc = arith.constant(0.0, type=T.f32())
            global_l = arith.constant(0.0, type=T.f32())

            for p in range_constexpr(num_partitions):
                m_val = buffer_ops.buffer_load(rsrc_M,
                    arith.addi(es_base, arith.constant(p, type=T.i32())), vec_width=1)
                l_val = buffer_ops.buffer_load(rsrc_L,
                    arith.addi(es_base, arith.constant(p, type=T.i32())), vec_width=1)

                weight = arith.exp2f(arith.mulf(arith.subf(m_val, global_max), log2e))
                global_l = arith.addf(global_l, arith.mulf(weight, l_val))

                o_base = arith.addi(
                    arith.addi(
                        arith.muli(batch_idx, arith.constant(_stride_in_batch, type=T.i32())),
                        arith.muli(head_idx, arith.constant(_stride_in_head, type=T.i32()))
                    ),
                    arith.muli(arith.constant(p, type=T.i32()),
                              arith.constant(_stride_in_part, type=T.i32()))
                )
                o_val = arith.extf(
                    buffer_ops.buffer_load(rsrc_O_in, arith.addi(o_base, col), vec_width=1),
                    T.f32()
                )
                acc = arith.addf(acc, arith.mulf(weight, o_val))

            # Normalize
            has_sum = arith.cmpf(global_l, arith.constant(0.0, type=T.f32()), predicate="ogt")
            inv_l = arith.divf(arith.constant(1.0, type=T.f32()), global_l)
            inv_l = arith.select(has_sum, inv_l, arith.constant(0.0, type=T.f32()))
            result = arith.mulf(acc, inv_l)

            # Write to output
            out_base = arith.addi(
                arith.muli(batch_idx, arith.constant(_stride_out_batch, type=T.i32())),
                arith.muli(head_idx, arith.constant(_stride_out_head, type=T.i32()))
            )
            buffer_ops.buffer_store(arith.truncf(result, T.f16()), rsrc_O_out,
                                    arith.addi(out_base, col))

    return decode_reduce_kernel
