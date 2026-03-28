"""
FlashAttention Combine Kernel for AMD GPUs via FlyDSL.

Combines partial results from split-KV (FlashDecoding) into final output.
This is a standalone reduce kernel that works with both the prefill and decode
split paths.

Input:
  O_partial: [batch, num_heads, num_splits, head_dim]  -- partial outputs
  LSE_partial: [batch, num_heads, num_splits]           -- log-sum-exp per split

Output:
  O: [batch, seqlen_q, num_heads, head_dim]            -- combined output

Algorithm:
  For each (batch, head, query_row):
    global_lse = logsumexp(LSE_partial[split_0..split_N])
    O[row, d] = sum_s( exp(LSE_s - global_lse) * O_partial[s, d] )
"""

from __future__ import annotations
import math as _math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, buffer_ops, range_constexpr
from flydsl.expr.typing import T

LOG2E = 1.4426950408889634
NEG_INF = -1e30
NUM_THREADS = 256


def build_combine_module(
    batch_size,
    num_heads,
    num_splits,
    head_dim=128,
):
    """
    Build the combine kernel module.

    Grid: (batch_size * num_heads,)
    Block: (NUM_THREADS,)
    """

    _stride_partial_head = num_splits * head_dim
    _stride_partial_batch = num_heads * num_splits * head_dim
    _stride_partial_split = head_dim
    _stride_lse_head = num_splits
    _stride_lse_batch = num_heads * num_splits
    _stride_out_head = head_dim
    _stride_out_batch = num_heads * head_dim

    @flyc.kernel
    def combine_kernel(
        O_partial: fx.Tensor,
        LSE_partial: fx.Tensor,
        O_out: fx.Tensor,
    ):
        tid = gpu.thread_idx.x
        bid = gpu.block_idx.x

        batch_idx = arith.divui(bid, arith.constant(num_heads, type=T.i32()))
        head_idx = arith.remui(bid, arith.constant(num_heads, type=T.i32()))

        rsrc_partial = buffer_ops.create_buffer_resource(O_partial)
        rsrc_lse = buffer_ops.create_buffer_resource(LSE_partial)
        rsrc_out = buffer_ops.create_buffer_resource(O_out)

        lse_base = arith.addi(
            arith.muli(batch_idx, arith.constant(_stride_lse_batch, type=T.i32())),
            arith.muli(head_idx, arith.constant(_stride_lse_head, type=T.i32()))
        )

        # Step 1: Find global max LSE
        global_max = arith.constant(NEG_INF, type=T.f32())
        for s in range_constexpr(num_splits):
            lse_val = buffer_ops.buffer_load(
                rsrc_lse,
                arith.addi(lse_base, arith.constant(s, type=T.i32())),
                vec_width=1
            )
            global_max = arith.maximumf(global_max, lse_val)

        # Step 2: Compute global normalizer (sum of exp(lse - max))
        global_sum = arith.constant(0.0, type=T.f32())
        log2e = arith.constant(LOG2E, type=T.f32())
        for s in range_constexpr(num_splits):
            lse_val = buffer_ops.buffer_load(
                rsrc_lse,
                arith.addi(lse_base, arith.constant(s, type=T.i32())),
                vec_width=1
            )
            weight = arith.exp2f(arith.mulf(arith.subf(lse_val, global_max), log2e))
            global_sum = arith.addf(global_sum, weight)

        # Step 3: For each dimension, compute weighted sum
        dims_per_thread = max(1, head_dim // NUM_THREADS)
        for d_idx in range(dims_per_thread):
            col = arith.addi(tid, arith.constant(d_idx * NUM_THREADS, type=T.i32()))
            col_valid = arith.cmpi(col, arith.constant(head_dim, type=T.i32()), predicate="ult")

            acc = arith.constant(0.0, type=T.f32())
            for s in range_constexpr(num_splits):
                lse_val = buffer_ops.buffer_load(
                    rsrc_lse,
                    arith.addi(lse_base, arith.constant(s, type=T.i32())),
                    vec_width=1
                )
                weight = arith.exp2f(arith.mulf(arith.subf(lse_val, global_max), log2e))

                partial_base = arith.addi(
                    arith.addi(
                        arith.muli(batch_idx, arith.constant(_stride_partial_batch, type=T.i32())),
                        arith.muli(head_idx, arith.constant(_stride_partial_head, type=T.i32()))
                    ),
                    arith.muli(arith.constant(s, type=T.i32()),
                              arith.constant(_stride_partial_split, type=T.i32()))
                )
                o_val = arith.extf(
                    buffer_ops.buffer_load(rsrc_partial, arith.addi(partial_base, col), vec_width=1),
                    T.f32()
                )
                acc = arith.addf(acc, arith.mulf(weight, o_val))

            # Normalize
            has_sum = arith.cmpf(global_sum, arith.constant(0.0, type=T.f32()), predicate="ogt")
            inv_sum = arith.divf(arith.constant(1.0, type=T.f32()), global_sum)
            inv_sum = arith.select(has_sum, inv_sum, arith.constant(0.0, type=T.f32()))
            result = arith.mulf(acc, inv_sum)

            out_base = arith.addi(
                arith.muli(batch_idx, arith.constant(_stride_out_batch, type=T.i32())),
                arith.muli(head_idx, arith.constant(_stride_out_head, type=T.i32()))
            )
            buffer_ops.buffer_store(arith.truncf(result, T.f16()), rsrc_out,
                                    arith.addi(out_base, col))

    return combine_kernel
