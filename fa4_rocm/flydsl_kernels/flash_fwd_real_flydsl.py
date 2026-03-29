"""
Real FlyDSL Flash Attention Forward (Correct, Non-Causal).

Each thread computes one O[row, d] element via scalar attention.
buffer_load i32 reads 2 packed bf16; we always read pairs and use the right half.

Usage: cd /opt/FlyDSL && python /workspace/fa4_rocm/flydsl_kernels/flash_fwd_real_flydsl.py
"""

import os, sys, math, time, json
sys.path.insert(0, "/opt/FlyDSL")
sys.path.insert(0, "/opt/FlyDSL/kernels")

import flydsl
from flydsl.dialects.ext import flir, arith, gpu, buffer_ops, vector, rocdl
from flydsl.dialects.ext import math as flydsl_math
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator
from _mlir import ir
from flydsl.lang.ir.types import T, memref
from kernels.kernels_common import stream_ptr_to_async_token

HD = 128
BLOCK_SIZE = 64
DYN = ir.ShapedType.get_dynamic_size()


def compile_flash_attn(seqlen, num_heads, batch, softmax_scale):
    arch = get_hip_arch()
    _sq = seqlen
    _nh = num_heads
    _batch = batch
    _scale = float(softmax_scale)
    # Total output elements (even dims only: HD/2 per row)
    _out_per_row = HD // 2
    _total = batch * num_heads * seqlen * _out_per_row
    n_blocks = (_total + BLOCK_SIZE - 1) // BLOCK_SIZE

    module_name = "flash_fwd_bf16"

    class _FA(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            pass

        @flir.kernel
        def flash_kernel(
            self: flir.T.i64,
            arg_o: lambda: memref(DYN, T.bf16),
            arg_q: lambda: memref(DYN, T.bf16),
            arg_k: lambda: memref(DYN, T.bf16),
            arg_v: lambda: memref(DYN, T.bf16),
        ):
            tid_idx = gpu.thread_id("x")
            bid_x = gpu.block_id("x")

            c_bs = arith.constant(BLOCK_SIZE, index=True)
            global_idx = bid_x * c_bs + tid_idx

            c_hd = arith.constant(HD, index=True)
            c_hd2 = arith.constant(HD // 2, index=True)
            c_sq = arith.constant(_sq, index=True)
            c_nh = arith.constant(_nh, index=True)
            c_2 = arith.constant(2, index=True)

            # Each thread handles one dword of output = 2 bf16 output elements
            # global_idx -> (batch, head, row, d_pair)
            d_pair = global_idx % c_hd2
            rem1 = global_idx // c_hd2
            row_idx = rem1 % c_sq
            rem2 = rem1 // c_sq
            head_idx = rem2 % c_nh
            batch_idx = rem2 // c_nh

            q_rsrc = buffer_ops.create_buffer_resource(arg_q, max_size=True)
            k_rsrc = buffer_ops.create_buffer_resource(arg_k, max_size=True)
            v_rsrc = buffer_ops.create_buffer_resource(arg_v, max_size=True)
            o_rsrc = buffer_ops.create_buffer_resource(arg_o, max_size=True)

            bh_base = (batch_idx * c_nh + head_idx) * c_sq * c_hd

            zero_f = arith.constant(0.0, type=T.f32)
            scale_f = arith.constant(_scale, type=T.f32)
            log2e_f = arith.constant(1.4426950408889634, type=T.f32)
            neg_inf = arith.constant(float("-inf"), type=T.f32)
            one_f = arith.constant(1.0, type=T.f32)

            # Online softmax accumulators
            row_max = neg_inf
            row_sum = zero_f
            acc_lo = zero_f  # accumulator for even dim
            acc_hi = zero_f  # accumulator for odd dim

            for k_idx in range_constexpr(_sq):
                c_k = arith.constant(k_idx, index=True)

                # dot = Q[row,:] . K[k,:] * scale
                dot = zero_f
                for dp in range_constexpr(HD // 2):
                    c_dp = arith.constant(dp, index=True)

                    q_dw = arith.index_cast(T.i32,
                        (bh_base + row_idx * c_hd) // c_2 + c_dp)
                    k_dw = arith.index_cast(T.i32,
                        (bh_base + c_k * c_hd) // c_2 + c_dp)

                    q_i32 = buffer_ops.buffer_load(q_rsrc, q_dw, vec_width=1, dtype=T.i32)
                    k_i32 = buffer_ops.buffer_load(k_rsrc, k_dw, vec_width=1, dtype=T.i32)

                    q_v2 = vector.bitcast(T.vec(2, T.bf16),
                        vector.broadcast(T.vec(1, T.i32), q_i32))
                    k_v2 = vector.bitcast(T.vec(2, T.bf16),
                        vector.broadcast(T.vec(1, T.i32), k_i32))

                    q0 = arith.extf(T.f32, vector.extract(q_v2, static_position=[0]))
                    q1 = arith.extf(T.f32, vector.extract(q_v2, static_position=[1]))
                    k0 = arith.extf(T.f32, vector.extract(k_v2, static_position=[0]))
                    k1 = arith.extf(T.f32, vector.extract(k_v2, static_position=[1]))

                    dot = dot + q0 * k0 + q1 * k1

                dot = dot * scale_f

                # Online softmax
                old_max = row_max
                new_max = arith.maximum(old_max, dot)
                rescale = flydsl_math.exp2(arith.unwrap((old_max - new_max) * log2e_f))
                p = flydsl_math.exp2(arith.unwrap((dot - new_max) * log2e_f))

                row_sum = row_sum * rescale + p
                row_max = new_max
                acc_lo = acc_lo * rescale
                acc_hi = acc_hi * rescale

                # V[k, 2*d_pair : 2*d_pair+2]
                v_dw = arith.index_cast(T.i32,
                    (bh_base + c_k * c_hd) // c_2 + d_pair)
                v_i32 = buffer_ops.buffer_load(v_rsrc, v_dw, vec_width=1, dtype=T.i32)
                v_v2 = vector.bitcast(T.vec(2, T.bf16),
                    vector.broadcast(T.vec(1, T.i32), v_i32))
                v0 = arith.extf(T.f32, vector.extract(v_v2, static_position=[0]))
                v1 = arith.extf(T.f32, vector.extract(v_v2, static_position=[1]))

                acc_lo = acc_lo + p * v0
                acc_hi = acc_hi + p * v1

            # Normalize
            inv_sum = one_f / row_sum
            res_lo = arith.trunc_f(T.bf16, acc_lo * inv_sum)
            res_hi = arith.trunc_f(T.bf16, acc_hi * inv_sum)

            # Pack 2 bf16 into i32 and store
            out_v2 = vector.from_elements(T.vec(2, T.bf16), [res_lo, res_hi])
            out_i32_v = vector.bitcast(T.vec(1, T.i32), out_v2)
            out_i32 = vector.extract(out_i32_v, static_position=[0])

            o_dw = arith.index_cast(T.i32,
                (bh_base + row_idx * c_hd) // c_2 + d_pair)
            buffer_ops.buffer_store(out_i32, o_rsrc, o_dw)

        @flir.jit
        def launch(
            self: flir.T.i64,
            arg_o: lambda: memref(DYN, T.bf16),
            arg_q: lambda: memref(DYN, T.bf16),
            arg_k: lambda: memref(DYN, T.bf16),
            arg_v: lambda: memref(DYN, T.bf16),
            sp: flir.T.i64,
        ):
            st = stream_ptr_to_async_token(sp)
            g_x = arith.constant(n_blocks, index=True)
            c1 = arith.constant(1, index=True)
            b_x = arith.constant(BLOCK_SIZE, index=True)
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "flash_kernel"],
                grid_size=(g_x, c1, c1),
                block_size=(b_x, c1, c1),
                kernel_operands=[arg_o, arg_q, arg_k, arg_v],
                async_dependencies=[st],
            )

    m = _FA()
    return flydsl.compile(m)


def build_flash_attn_module(seqlen, num_heads, batch, softmax_scale):
    """Public API for FlyDSL library."""
    return compile_flash_attn(seqlen, num_heads, batch, softmax_scale)


def test_flash_attn():
    import torch

    batch, seqlen, nh, hd = 1, 16, 1, HD
    scale = 1.0 / math.sqrt(hd)

    print("=" * 60)
    print("FlyDSL Flash Attention -- Correctness Test")
    print("=" * 60)
    print(f"Shape: b={batch} s={seqlen} h={nh} d={hd}")

    exe = compile_flash_attn(seqlen, nh, batch, scale)
    print("Compilation: SUCCESS")

    torch.manual_seed(42)
    q = torch.randn(batch, seqlen, nh, hd, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, seqlen, nh, hd, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, seqlen, nh, hd, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros_like(q)

    stream = torch.cuda.current_stream()
    exe.launch(o.view(-1), q.view(-1), k.view(-1), v.view(-1), stream.cuda_stream)
    torch.cuda.synchronize()
    print("Launch: SUCCESS")

    with torch.no_grad():
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float(), k.transpose(1, 2).float(),
            v.transpose(1, 2).float(), is_causal=False, scale=scale
        ).transpose(1, 2).to(torch.bfloat16)

    err = (o.float() - ref.float()).abs().max().item()
    mean_err = (o.float() - ref.float()).abs().mean().item()
    has_nan = torch.isnan(o).any().item()
    correct = err < 0.1 and not has_nan
    print(f"Max error: {err:.6f}")
    print(f"Mean error: {mean_err:.6f}")
    print(f"NaN: {has_nan}")
    print(f"Correct: {correct}")

    if not correct:
        print("Output[:8]:", o[0, 0, 0, :8].tolist())
        print("Ref[:8]:  ", ref[0, 0, 0, :8].tolist())

    tf, lat = 0.0, 0.0
    if correct:
        torch.cuda.synchronize()
        for _ in range(5):
            exe.launch(o.view(-1), q.view(-1), k.view(-1), v.view(-1), stream.cuda_stream)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N = 20
        for _ in range(N):
            exe.launch(o.view(-1), q.view(-1), k.view(-1), v.view(-1), stream.cuda_stream)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        lat = elapsed / N * 1e6
        flops = 4 * batch * seqlen * seqlen * nh * hd
        tf = flops / (elapsed / N) / 1e12
        print(f"Latency: {lat:.1f} us | TFLOPS: {tf:.4f}")

    result = {
        "backend": "flydsl_flir", "compiled": True, "launched": True,
        "correct": correct, "max_error": err, "mean_error": mean_err,
        "tflops": round(tf, 4), "latency_us": round(lat, 1),
        "batch": batch, "seqlen": seqlen, "nheads": nh, "hdim": hd,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    os.chdir("/opt/FlyDSL")
    test_flash_attn()
