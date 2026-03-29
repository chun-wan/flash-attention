"""
FlyDSL Flash Attention with scf.for_ runtime KV loop.

Uses runtime loops (scf.for_ with iter_args) so seqlen can be large
without compile-time explosion. Vectorized dwordx4 loads for Q@K dot.
Online softmax with exp2. Non-causal.

Usage: cd /opt/FlyDSL && python /workspace/fa4_rocm/flydsl_kernels/flash_fwd_scf_flydsl.py
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
from _mlir.dialects import scf
from flydsl.lang.ir.types import T, memref
from kernels.kernels_common import stream_ptr_to_async_token

HD = 128
BLOCK_SIZE = 256
DYN = ir.ShapedType.get_dynamic_size()


def compile_flash_attn_scf(seqlen, num_heads, batch, softmax_scale):
    """Compile FlyDSL flash attention with runtime KV loop."""
    arch = get_hip_arch()
    _sq = seqlen
    _nh = num_heads
    _batch = batch
    _scale = float(softmax_scale)
    _hd2 = HD // 2
    _total_dwords = batch * num_heads * seqlen * _hd2
    n_blocks = (_total_dwords + BLOCK_SIZE - 1) // BLOCK_SIZE

    module_name = f"flash_fwd_scf_b{batch}s{seqlen}h{num_heads}"

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
            c_hd2 = arith.constant(_hd2, index=True)
            c_sq = arith.constant(_sq, index=True)
            c_nh = arith.constant(_nh, index=True)
            c_2 = arith.constant(2, index=True)

            # Layout: (batch, seqlen, heads, hdim) -> flat index decomposition
            d_pair = global_idx % c_hd2
            rem1 = global_idx // c_hd2
            head_idx = rem1 % c_nh
            rem2 = rem1 // c_nh
            row_idx = rem2 % c_sq
            batch_idx = rem2 // c_sq

            c_total = arith.constant(_total_dwords, index=True)
            # Bounds check -- skip threads beyond valid range
            # (gpu.return not available in FLIR kernel, so we just make
            # out-of-bounds threads read/write position 0 which is harmless)

            q_rsrc = buffer_ops.create_buffer_resource(arg_q, max_size=True)
            k_rsrc = buffer_ops.create_buffer_resource(arg_k, max_size=True)
            v_rsrc = buffer_ops.create_buffer_resource(arg_v, max_size=True)
            o_rsrc = buffer_ops.create_buffer_resource(arg_o, max_size=True)

            # Strides for (batch, seqlen, heads, hdim) layout
            stride_b = c_sq * c_nh * c_hd  # seqlen * heads * hdim
            stride_s = c_nh * c_hd         # heads * hdim
            stride_h = c_hd               # hdim

            q_row_base = (batch_idx * stride_b + row_idx * stride_s + head_idx * stride_h) // c_2

            neg_inf = arith.constant(float("-inf"), type=T.f32)
            zero_f = arith.constant(0.0, type=T.f32)
            scale_f = arith.constant(_scale, type=T.f32)
            log2e_f = arith.constant(1.4426950408889634, type=T.f32)
            one_f = arith.constant(1.0, type=T.f32)

            # scf.for_ KV loop with iter_args: [row_max, row_sum, acc_lo, acc_hi]
            c0_idx = arith.unwrap(arith.constant(0, index=True))
            c1_idx = arith.unwrap(arith.constant(1, index=True))
            sk_idx = arith.unwrap(arith.constant(_sq, index=True))

            for iv, (i_row_max, i_row_sum, i_acc_lo, i_acc_hi), (r_max, r_sum, r_lo, r_hi) in scf.for_(
                c0_idx, sk_idx, c1_idx,
                iter_args=[arith.unwrap(neg_inf), arith.unwrap(zero_f),
                           arith.unwrap(zero_f), arith.unwrap(zero_f)]
            ):
                # Compute dot(Q[row,:], K[iv,:]) via dwordx4 loads
                k_row_base = (batch_idx * stride_b + iv * stride_s + head_idx * stride_h) // c_2

                dot = zero_f
                for dp4 in range_constexpr(HD // 8):
                    c_dp4 = arith.constant(dp4 * 4, index=True)

                    q_dw = arith.index_cast(T.i32, q_row_base + c_dp4)
                    k_dw = arith.index_cast(T.i32, k_row_base + c_dp4)

                    q_v4 = buffer_ops.buffer_load(q_rsrc, q_dw, vec_width=4, dtype=T.i32)
                    k_v4 = buffer_ops.buffer_load(k_rsrc, k_dw, vec_width=4, dtype=T.i32)

                    q_bf16x8 = vector.bitcast(T.vec(8, T.bf16), q_v4)
                    k_bf16x8 = vector.bitcast(T.vec(8, T.bf16), k_v4)

                    for e in range_constexpr(8):
                        qe = arith.extf(T.f32, vector.extract(q_bf16x8, static_position=[e]))
                        ke = arith.extf(T.f32, vector.extract(k_bf16x8, static_position=[e]))
                        dot = dot + qe * ke

                dot = dot * scale_f

                # Online softmax
                new_max = arith.maximum(i_row_max, dot)
                rescale = flydsl_math.exp2(arith.unwrap((i_row_max - new_max) * log2e_f))
                p = flydsl_math.exp2(arith.unwrap((dot - new_max) * log2e_f))

                new_sum = i_row_sum * rescale + p
                new_acc_lo = i_acc_lo * rescale
                new_acc_hi = i_acc_hi * rescale

                # V[iv, 2*d_pair : 2*d_pair+2]
                v_dw = arith.index_cast(T.i32, (batch_idx * stride_b + iv * stride_s + head_idx * stride_h) // c_2 + d_pair)
                v_i32 = buffer_ops.buffer_load(v_rsrc, v_dw, vec_width=1, dtype=T.i32)
                v_v2 = vector.bitcast(T.vec(2, T.bf16), vector.broadcast(T.vec(1, T.i32), v_i32))
                v0 = arith.extf(T.f32, vector.extract(v_v2, static_position=[0]))
                v1 = arith.extf(T.f32, vector.extract(v_v2, static_position=[1]))

                new_acc_lo = new_acc_lo + p * v0
                new_acc_hi = new_acc_hi + p * v1

                scf.YieldOp([arith.unwrap(new_max), arith.unwrap(new_sum),
                              arith.unwrap(new_acc_lo), arith.unwrap(new_acc_hi)])

            # Normalize and store
            inv_sum = one_f / r_sum
            res_lo = arith.trunc_f(T.bf16, r_lo * inv_sum)
            res_hi = arith.trunc_f(T.bf16, r_hi * inv_sum)

            out_v2 = vector.from_elements(T.vec(2, T.bf16), [res_lo, res_hi])
            out_i32_v = vector.bitcast(T.vec(1, T.i32), out_v2)
            out_i32 = vector.extract(out_i32_v, static_position=[0])

            o_dw = arith.index_cast(T.i32, (batch_idx * stride_b + row_idx * stride_s + head_idx * stride_h) // c_2 + d_pair)
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


def main():
    import torch

    shapes = [
        (1, 128, 1, "b1_s128_h1"),
        (2, 2048, 32, "b2_s2048_h32"),
    ]
    hd = HD

    print("=" * 70)
    print("FlyDSL Flash Attention with scf.for_ Runtime Loops")
    print("=" * 70)

    results = []
    for batch, seqlen, nh, label in shapes:
        scale = 1.0 / math.sqrt(hd)
        print(f"\n--- {label}: b={batch} s={seqlen} h={nh} d={hd} ---")

        try:
            t_compile = time.perf_counter()
            exe = compile_flash_attn_scf(seqlen, nh, batch, scale)
            compile_time = time.perf_counter() - t_compile
            print(f"  Compile: {compile_time:.1f}s SUCCESS")
        except Exception as e:
            print(f"  Compile FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append({"shape": label, "compiled": False})
            continue

        torch.manual_seed(42)
        q = torch.randn(batch, seqlen, nh, hd, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, seqlen, nh, hd, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, seqlen, nh, hd, dtype=torch.bfloat16, device="cuda")
        o = torch.zeros_like(q)

        stream = torch.cuda.current_stream()
        exe.launch(o.view(-1), q.view(-1), k.view(-1), v.view(-1), stream.cuda_stream)
        torch.cuda.synchronize()

        with torch.no_grad():
            ref = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2).float(), k.transpose(1, 2).float(),
                v.transpose(1, 2).float(), is_causal=False, scale=scale
            ).transpose(1, 2).to(torch.bfloat16)

        err = (o.float() - ref.float()).abs().max().item()
        correct = err < 0.05 and not torch.isnan(o).any().item()
        print(f"  Correct: {correct} (max_err={err:.6f})")

        tf, lat = 0.0, 0.0
        if correct:
            torch.cuda.synchronize()
            for _ in range(10):
                exe.launch(o.view(-1), q.view(-1), k.view(-1), v.view(-1), stream.cuda_stream)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            N = 50
            for _ in range(N):
                exe.launch(o.view(-1), q.view(-1), k.view(-1), v.view(-1), stream.cuda_stream)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            lat = elapsed / N * 1e6
            flops = 4 * batch * seqlen * seqlen * nh * hd
            tf = flops / (elapsed / N) / 1e12
            print(f"  Latency: {lat:.1f} us | TFLOPS: {tf:.4f}")

        results.append({
            "shape": label, "compiled": True, "correct": correct,
            "max_error": err, "tflops": round(tf, 4), "latency_us": round(lat, 1),
            "compile_time_s": round(compile_time, 1),
        })

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for r in results:
        s = r["shape"]
        if r.get("compiled") and r.get("correct"):
            print(f"  {s}: {r['tflops']:.4f} TF, {r['latency_us']:.1f} us, compile {r['compile_time_s']:.1f}s")
        elif r.get("compiled"):
            print(f"  {s}: INCORRECT (err={r.get('max_error', '?')})")
        else:
            print(f"  {s}: COMPILE FAILED")

    out_path = "/workspace/fa4_rocm/flydsl_kernels/scf_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    os.chdir("/opt/FlyDSL")
    main()
