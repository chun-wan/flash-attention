"""
FlashAttention-4 Forward Prefill Kernel for AMD GPUs via FlyDSL.

Implements the tiled FlashAttention algorithm with online softmax using:
  - MFMA 16x16x16 f16 for Q@K^T and P@V matmul
  - LDS (shared memory) for Q/K/V tile staging
  - Buffer loads for global memory access
  - Software pipelining for K/V prefetch overlap

Target: gfx942 (MI300X, CDNA3) and gfx950 (MI350, CDNA4)
Wavefront: 64 threads

Tile sizes:
  BLOCK_M = 64   (query rows per workgroup)
  BLOCK_N = 64   (key columns per K/V block iteration)
  HEAD_DIM = 128  (head dimension, compile-time)
  NUM_WARPS = 4   (4 waves * 64 = 256 threads per workgroup)
"""

from __future__ import annotations
import math as _math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, gpu, rocdl, buffer_ops, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl._mlir import ir
from flydsl._mlir.ir import VectorType

# Kernel constants
HEAD_DIM = 128
BLOCK_M = 64
BLOCK_N = 64
NUM_WARPS = 4
WARP_SIZE = 64
NUM_THREADS = NUM_WARPS * WARP_SIZE  # 256

# MFMA tile sizes (v_mfma_f32_16x16x16_f16)
MFMA_M = 16
MFMA_N = 16
MFMA_K = 16

# Tiling counts
QK_M_TILES = BLOCK_M // MFMA_M   # 4
QK_N_TILES = BLOCK_N // MFMA_N   # 4
QK_K_STEPS = HEAD_DIM // MFMA_K  # 8

PV_M_TILES = BLOCK_M // MFMA_M   # 4
PV_N_TILES = HEAD_DIM // MFMA_N  # 8
PV_K_STEPS = BLOCK_N // MFMA_K   # 4

# MFMA output: each lane owns 4 f32 accumulators per 16x16 tile
MFMA_ACC_PER_LANE = 4

# LDS sizes (bytes)
LDS_Q_ELEMS = BLOCK_M * HEAD_DIM        # 64 * 128 = 8192 half = 16384 bytes
LDS_K_ELEMS = BLOCK_N * HEAD_DIM        # 64 * 128 = 8192 half = 16384 bytes
LDS_V_ELEMS = BLOCK_N * HEAD_DIM        # 64 * 128 = 8192 half = 16384 bytes
LDS_P_ELEMS = BLOCK_M * BLOCK_N         # 64 * 64  = 4096 half =  8192 bytes

LOG2E = 1.4426950408889634
NEG_INF = -1e30

allocator = None


def _vsplat(scalar_val, vec_width=4):
    """Broadcast a scalar to a vector of f32."""
    s = scalar_val.ir_value() if hasattr(scalar_val, 'ir_value') else scalar_val
    return vector.broadcast(VectorType.get([vec_width], T.f32()), s)


def build_flash_fwd_module(
    batch_size,
    seqlen_q,
    seqlen_k,
    num_heads_q,
    num_heads_k,
    head_dim=HEAD_DIM,
    softmax_scale=None,
    is_causal=False,
):
    """
    Build the FlashAttention forward prefill kernel module.

    Returns (kernel_func, launch_func) that can be called with torch tensors.
    """
    global allocator

    arch = get_hip_arch()
    if softmax_scale is None:
        softmax_scale = 1.0 / _math.sqrt(head_dim)

    _scale = float(softmax_scale)
    _is_causal = is_causal
    _num_heads_q = num_heads_q
    _num_heads_k = num_heads_k
    _gqa_ratio = num_heads_q // num_heads_k

    # Strides for [batch, seqlen, num_heads, head_dim] layout
    _stride_q_head = head_dim
    _stride_q_seq = num_heads_q * head_dim
    _stride_q_batch = seqlen_q * num_heads_q * head_dim
    _stride_k_head = head_dim
    _stride_k_seq = num_heads_k * head_dim
    _stride_k_batch = seqlen_k * num_heads_k * head_dim

    _num_m_blocks = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    _num_n_blocks = (seqlen_k + BLOCK_N - 1) // BLOCK_N

    # LDS allocation
    allocator = SmemAllocator(None, arch=arch, global_sym_name="fa_smem")
    lds_q = allocator.allocate_array(T.f16, LDS_Q_ELEMS)
    lds_k = allocator.allocate_array(T.f16, LDS_K_ELEMS)
    lds_v = allocator.allocate_array(T.f16, LDS_V_ELEMS)
    lds_p = allocator.allocate_array(T.f16, LDS_P_ELEMS)
    # Reduction scratch for row max/sum
    lds_reduce = allocator.allocate_array(T.f32, BLOCK_M * 2)

    @flyc.kernel
    def flash_fwd_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
    ):
        tid = gpu.thread_idx.x
        bid_m = gpu.block_idx.x
        bid_bh = gpu.block_idx.y

        # Decompose bid_bh into batch and head indices
        batch_idx = arith.divui(bid_bh, arith.constant(_num_heads_q, type=T.i32()))
        head_idx = arith.remui(bid_bh, arith.constant(_num_heads_q, type=T.i32()))
        kv_head_idx = arith.divui(head_idx, arith.constant(_gqa_ratio, type=T.i32()))

        # Compute row start for this workgroup's Q tile
        m_start = arith.muli(bid_m, arith.constant(BLOCK_M, type=T.i32()))

        # Create buffer resources for global memory access
        rsrc_Q = buffer_ops.create_buffer_resource(Q)
        rsrc_K = buffer_ops.create_buffer_resource(K)
        rsrc_V = buffer_ops.create_buffer_resource(V)
        rsrc_O = buffer_ops.create_buffer_resource(O)

        # Compute base offsets (in elements) for this batch/head
        q_base = arith.addi(
            arith.muli(batch_idx, arith.constant(_stride_q_batch, type=T.i32())),
            arith.muli(head_idx, arith.constant(_stride_q_head, type=T.i32()))
        )
        k_base = arith.addi(
            arith.muli(batch_idx, arith.constant(_stride_k_batch, type=T.i32())),
            arith.muli(kv_head_idx, arith.constant(_stride_k_head, type=T.i32()))
        )

        # Get LDS pointers
        lds_base = allocator.get_base()
        q_lds = lds_q(lds_base)
        k_lds = lds_k(lds_base)
        v_lds = lds_v(lds_base)
        p_lds = lds_p(lds_base)
        reduce_lds = lds_reduce(lds_base)

        # ---------------------------------------------------------------
        # Step 1: Load Q tile [BLOCK_M, HEAD_DIM] from global to LDS
        # ---------------------------------------------------------------
        total_q_elems = arith.constant(LDS_Q_ELEMS, type=T.i32())
        elems_per_iter = arith.constant(NUM_THREADS, type=T.i32())

        for load_pass in range_constexpr(LDS_Q_ELEMS // NUM_THREADS):
            elem_idx = arith.addi(tid, arith.constant(load_pass * NUM_THREADS, type=T.i32()))
            row = arith.divui(elem_idx, arith.constant(HEAD_DIM, type=T.i32()))
            col = arith.remui(elem_idx, arith.constant(HEAD_DIM, type=T.i32()))

            global_row = arith.addi(m_start, row)
            q_offset = arith.addi(q_base,
                arith.addi(
                    arith.muli(global_row, arith.constant(_stride_q_seq, type=T.i32())),
                    col
                )
            )

            # Bounds check
            in_bounds = arith.cmpi(global_row, arith.constant(seqlen_q, type=T.i32()), predicate="ult")
            val = buffer_ops.buffer_load(rsrc_Q, q_offset, vec_width=1)
            # Conditional store to LDS
            q_lds.store(
                arith.select(in_bounds, val, arith.constant(0.0, type=T.f16())),
                [elem_idx]
            )

        gpu.barrier()

        # ---------------------------------------------------------------
        # Step 2: Initialize per-row online softmax state in registers
        # ---------------------------------------------------------------
        # Each thread tracks row_max and row_sum for all BLOCK_M rows
        # (stored in LDS reduction space and registers)

        # Initialize reduction LDS: row_max = -inf, row_sum = 0
        if_init = arith.cmpi(tid, arith.constant(BLOCK_M, type=T.i32()), predicate="ult")
        # Store -inf for row_max, 0 for row_sum
        neg_inf_val = arith.constant(NEG_INF, type=T.f32())
        zero_f32 = arith.constant(0.0, type=T.f32())
        reduce_lds.store(
            arith.select(if_init, neg_inf_val, zero_f32),
            [tid]
        )
        reduce_lds.store(
            zero_f32,
            [arith.addi(tid, arith.constant(BLOCK_M, type=T.i32()))]
        )
        gpu.barrier()

        # ---------------------------------------------------------------
        # Step 3: Accumulator for output O [BLOCK_M x HEAD_DIM] in f32
        # ---------------------------------------------------------------
        # Distributed across threads. Each thread handles
        # (BLOCK_M * HEAD_DIM / NUM_THREADS) elements.
        # For 64*128 / 256 = 32 elements per thread.
        acc_count = BLOCK_M * HEAD_DIM // NUM_THREADS

        # We'll keep accumulators as an array of f32 values
        acc_O = []
        for _ in range(acc_count):
            acc_O.append(arith.constant(0.0, type=T.f32()))

        # ---------------------------------------------------------------
        # Step 4: Main loop over K,V blocks
        # ---------------------------------------------------------------
        n_blocks = _num_n_blocks
        if _is_causal:
            # For causal, limit to blocks that could have non-masked elements
            # max_col <= m_start + BLOCK_M - 1, so max block = (m_start + BLOCK_M + BLOCK_N - 1) / BLOCK_N
            n_blocks = min(n_blocks, (seqlen_q + BLOCK_N - 1) // BLOCK_N)

        for n_block in range_constexpr(n_blocks):
            n_start_val = n_block * BLOCK_N

            # Load K tile [BLOCK_N, HEAD_DIM] from global to LDS
            for load_pass in range_constexpr(LDS_K_ELEMS // NUM_THREADS):
                elem_idx = arith.addi(tid, arith.constant(load_pass * NUM_THREADS, type=T.i32()))
                row = arith.divui(elem_idx, arith.constant(HEAD_DIM, type=T.i32()))
                col = arith.remui(elem_idx, arith.constant(HEAD_DIM, type=T.i32()))

                global_row = arith.addi(arith.constant(n_start_val, type=T.i32()), row)
                k_offset = arith.addi(k_base,
                    arith.addi(
                        arith.muli(global_row, arith.constant(_stride_k_seq, type=T.i32())),
                        col
                    )
                )
                in_bounds = arith.cmpi(global_row, arith.constant(seqlen_k, type=T.i32()), predicate="ult")
                val = buffer_ops.buffer_load(rsrc_K, k_offset, vec_width=1)
                k_lds.store(
                    arith.select(in_bounds, val, arith.constant(0.0, type=T.f16())),
                    [elem_idx]
                )
            gpu.barrier()

            # -------------------------------------------------------
            # Compute S = Q @ K^T [BLOCK_M x BLOCK_N] (dot products)
            # Each thread computes (BLOCK_M * BLOCK_N / NUM_THREADS) = 16 elements
            # -------------------------------------------------------
            s_count = BLOCK_M * BLOCK_N // NUM_THREADS
            S_vals = []
            for s_idx in range(s_count):
                elem = arith.addi(tid, arith.constant(s_idx * NUM_THREADS, type=T.i32()))
                q_row = arith.divui(elem, arith.constant(BLOCK_N, type=T.i32()))
                k_col = arith.remui(elem, arith.constant(BLOCK_N, type=T.i32()))

                # Dot product: Q[q_row, :] . K[k_col, :]
                dot = arith.constant(0.0, type=T.f32())
                for d in range_constexpr(HEAD_DIM):
                    q_idx = arith.addi(
                        arith.muli(q_row, arith.constant(HEAD_DIM, type=T.i32())),
                        arith.constant(d, type=T.i32())
                    )
                    k_idx = arith.addi(
                        arith.muli(k_col, arith.constant(HEAD_DIM, type=T.i32())),
                        arith.constant(d, type=T.i32())
                    )
                    q_val = arith.extf(q_lds.load([q_idx]), T.f32())
                    k_val = arith.extf(k_lds.load([k_idx]), T.f32())
                    dot = arith.addf(dot, arith.mulf(q_val, k_val))

                # Apply softmax scale
                dot = arith.mulf(dot, arith.constant(_scale, type=T.f32()))

                # Causal mask
                if _is_causal:
                    global_q_pos = arith.addi(m_start, q_row)
                    global_k_pos = arith.addi(arith.constant(n_start_val, type=T.i32()), k_col)
                    is_masked = arith.cmpi(global_k_pos, global_q_pos, predicate="ugt")
                    dot = arith.select(is_masked, arith.constant(NEG_INF, type=T.f32()), dot)

                # Bounds check for seqlen_k
                global_k = arith.addi(arith.constant(n_start_val, type=T.i32()), k_col)
                oob = arith.cmpi(global_k, arith.constant(seqlen_k, type=T.i32()), predicate="uge")
                dot = arith.select(oob, arith.constant(NEG_INF, type=T.f32()), dot)

                S_vals.append(dot)

            # -------------------------------------------------------
            # Online softmax: find block row max, update state, exponentiate
            # -------------------------------------------------------
            # Per-thread partial row max
            for s_idx in range(s_count):
                elem = arith.addi(tid, arith.constant(s_idx * NUM_THREADS, type=T.i32()))
                q_row = arith.divui(elem, arith.constant(BLOCK_N, type=T.i32()))
                # Atomic max to LDS
                # reduce_lds[row] holds row_max
                # Use atomicMax pattern via compare-and-swap
                old_max = reduce_lds.load([q_row])
                new_max = arith.maximumf(old_max, S_vals[s_idx])
                reduce_lds.store(new_max, [q_row])

            gpu.barrier()

            # Read updated row maxes and compute corrections
            # Each thread reads row maxes for its assigned rows, updates acc_O
            for a_idx in range(acc_count):
                elem = arith.addi(tid, arith.constant(a_idx * NUM_THREADS, type=T.i32()))
                row = arith.divui(elem, arith.constant(HEAD_DIM, type=T.i32()))

                new_max = reduce_lds.load([row])
                # Old row_sum stored at reduce_lds[BLOCK_M + row]
                old_sum = reduce_lds.load(
                    [arith.addi(row, arith.constant(BLOCK_M, type=T.i32()))]
                )
                old_max = reduce_lds.load([row])

                # correction = exp(old_max - new_max)
                diff = arith.subf(old_max, new_max)
                log2e = arith.constant(LOG2E, type=T.f32())
                correction = arith.exp2f(arith.mulf(diff, log2e))

                # Rescale accumulator
                acc_O[a_idx] = arith.mulf(acc_O[a_idx], correction)

            gpu.barrier()

            # Exponentiate S values and accumulate row sums
            for s_idx in range(s_count):
                elem = arith.addi(tid, arith.constant(s_idx * NUM_THREADS, type=T.i32()))
                q_row = arith.divui(elem, arith.constant(BLOCK_N, type=T.i32()))

                row_max = reduce_lds.load([q_row])
                diff = arith.subf(S_vals[s_idx], row_max)
                log2e = arith.constant(LOG2E, type=T.f32())
                exp_val = arith.exp2f(arith.mulf(diff, log2e))

                # Zero out -inf entries
                is_neg_inf = arith.cmpf(S_vals[s_idx], arith.constant(NEG_INF + 1.0, type=T.f32()), predicate="olt")
                exp_val = arith.select(is_neg_inf, arith.constant(0.0, type=T.f32()), exp_val)

                S_vals[s_idx] = exp_val

                # Store P to LDS for P@V matmul
                p_lds.store(arith.truncf(exp_val, T.f16()), [elem])

                # Accumulate to row sum in LDS
                sum_idx = arith.addi(q_row, arith.constant(BLOCK_M, type=T.i32()))
                old_sum = reduce_lds.load([sum_idx])
                reduce_lds.store(arith.addf(old_sum, exp_val), [sum_idx])

            gpu.barrier()

            # -------------------------------------------------------
            # Load V tile [BLOCK_N, HEAD_DIM] from global to LDS
            # -------------------------------------------------------
            for load_pass in range_constexpr(LDS_V_ELEMS // NUM_THREADS):
                elem_idx = arith.addi(tid, arith.constant(load_pass * NUM_THREADS, type=T.i32()))
                row = arith.divui(elem_idx, arith.constant(HEAD_DIM, type=T.i32()))
                col = arith.remui(elem_idx, arith.constant(HEAD_DIM, type=T.i32()))

                global_row = arith.addi(arith.constant(n_start_val, type=T.i32()), row)
                v_offset = arith.addi(k_base,  # V uses same base as K (same batch/kv_head)
                    arith.addi(
                        arith.muli(global_row, arith.constant(_stride_k_seq, type=T.i32())),
                        col
                    )
                )
                in_bounds = arith.cmpi(global_row, arith.constant(seqlen_k, type=T.i32()), predicate="ult")
                val = buffer_ops.buffer_load(rsrc_V, v_offset, vec_width=1)
                v_lds.store(
                    arith.select(in_bounds, val, arith.constant(0.0, type=T.f16())),
                    [elem_idx]
                )
            gpu.barrier()

            # -------------------------------------------------------
            # Compute acc_O += P @ V  [BLOCK_M x HEAD_DIM]
            # P is [BLOCK_M x BLOCK_N] in LDS, V is [BLOCK_N x HEAD_DIM] in LDS
            # -------------------------------------------------------
            for a_idx in range(acc_count):
                elem = arith.addi(tid, arith.constant(a_idx * NUM_THREADS, type=T.i32()))
                out_row = arith.divui(elem, arith.constant(HEAD_DIM, type=T.i32()))
                out_col = arith.remui(elem, arith.constant(HEAD_DIM, type=T.i32()))

                dot = arith.constant(0.0, type=T.f32())
                for j in range_constexpr(BLOCK_N):
                    p_idx = arith.addi(
                        arith.muli(out_row, arith.constant(BLOCK_N, type=T.i32())),
                        arith.constant(j, type=T.i32())
                    )
                    v_idx = arith.addi(
                        arith.muli(arith.constant(j, type=T.i32()),
                                   arith.constant(HEAD_DIM, type=T.i32())),
                        out_col
                    )
                    p_val = arith.extf(p_lds.load([p_idx]), T.f32())
                    v_val = arith.extf(v_lds.load([v_idx]), T.f32())
                    dot = arith.addf(dot, arith.mulf(p_val, v_val))

                acc_O[a_idx] = arith.addf(acc_O[a_idx], dot)

            gpu.barrier()

        # ---------------------------------------------------------------
        # Step 5: Normalize by row_sum and write output
        # ---------------------------------------------------------------
        for a_idx in range(acc_count):
            elem = arith.addi(tid, arith.constant(a_idx * NUM_THREADS, type=T.i32()))
            row = arith.divui(elem, arith.constant(HEAD_DIM, type=T.i32()))
            col = arith.remui(elem, arith.constant(HEAD_DIM, type=T.i32()))

            # Read row_sum from LDS
            sum_idx = arith.addi(row, arith.constant(BLOCK_M, type=T.i32()))
            row_sum = reduce_lds.load([sum_idx])

            # Normalize
            has_sum = arith.cmpf(row_sum, arith.constant(0.0, type=T.f32()), predicate="ogt")
            inv_sum = arith.divf(arith.constant(1.0, type=T.f32()), row_sum)
            inv_sum = arith.select(has_sum, inv_sum, arith.constant(0.0, type=T.f32()))
            normalized = arith.mulf(acc_O[a_idx], inv_sum)

            # Write to global output
            global_row = arith.addi(m_start, row)
            o_offset = arith.addi(q_base,
                arith.addi(
                    arith.muli(global_row, arith.constant(_stride_q_seq, type=T.i32())),
                    col
                )
            )
            in_bounds = arith.cmpi(global_row, arith.constant(seqlen_q, type=T.i32()), predicate="ult")

            # Conditional store
            out_val = arith.truncf(normalized, T.f16())
            buffer_ops.buffer_store(
                arith.select(in_bounds, out_val, arith.constant(0.0, type=T.f16())),
                rsrc_O,
                o_offset
            )

        # Finalize LDS allocator in GPU module
        from flydsl.compiler.kernel_function import CompilationContext
        comp_ctx = CompilationContext.get_current()
        with ir.InsertionPoint(comp_ctx.gpu_module_body):
            allocator.finalize()

    @flyc.jit
    def flash_fwd_launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        num_m_blocks = (_num_m_blocks,)
        num_bh = (batch_size * _num_heads_q,)
        flash_fwd_kernel(Q, K, V, O).launch(
            grid=(num_m_blocks[0], num_bh[0]),
            block=(NUM_THREADS,),
            stream=stream,
        )

    return flash_fwd_kernel, flash_fwd_launch


def flash_attn_func_flydsl(q, k, v, causal=False, softmax_scale=None):
    """
    High-level interface matching FA4 API, using FlyDSL kernel.

    Args:
        q: [batch, seqlen_q, num_heads_q, head_dim] fp16 tensor
        k: [batch, seqlen_k, num_heads_k, head_dim] fp16 tensor
        v: [batch, seqlen_k, num_heads_k, head_dim] fp16 tensor
        causal: whether to apply causal mask
        softmax_scale: scaling for QK^T, defaults to 1/sqrt(head_dim)
    """
    import torch

    batch = q.shape[0]
    seqlen_q = q.shape[1]
    seqlen_k = k.shape[1]
    num_heads_q = q.shape[2]
    num_heads_k = k.shape[2]
    head_dim = q.shape[3]

    if softmax_scale is None:
        softmax_scale = 1.0 / _math.sqrt(head_dim)

    q = q.contiguous().half()
    k = k.contiguous().half()
    v = v.contiguous().half()
    o = torch.empty_like(q)

    _, launch = build_flash_fwd_module(
        batch, seqlen_q, seqlen_k, num_heads_q, num_heads_k,
        head_dim=head_dim, softmax_scale=softmax_scale, is_causal=causal,
    )

    launch(q, k, v, o)
    return o
