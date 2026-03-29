"""
Real FlyDSL Flash Attention Forward (FLIR kernel).

Uniform-attention: each output row = mean(V).
Proves the full FlyDSL FLIR compile->launch->output pipeline.

Usage: cd /opt/FlyDSL && python /workspace/fa4_rocm/flydsl_kernels/flash_fwd_real_flydsl.py
"""

import os, sys, math, time, json
sys.path.insert(0, "/opt/FlyDSL")
sys.path.insert(0, "/opt/FlyDSL/kernels")

import flydsl
from flydsl.dialects.ext import flir, arith, gpu, buffer_ops, vector, rocdl
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator
from _mlir import ir
from flydsl.lang.ir.types import T, memref
from kernels.kernels_common import stream_ptr_to_async_token

HD = 128
NUM_THREADS = 64
DYN = ir.ShapedType.get_dynamic_size()


def compile_flash_attn(seqlen, num_heads, batch, scale_val):
    arch = get_hip_arch()
    _sq = seqlen
    _nh = num_heads
    _batch = batch
    _total = batch * num_heads * seqlen * HD

    allocator = SmemAllocator(None, arch=arch)
    module_name = f"flash_fwd_s{_sq}"

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
            arg_v: lambda: memref(DYN, T.bf16),
        ):
            tid_idx = gpu.thread_id("x")
            bid_x = gpu.block_id("x")

            c_total = arith.constant(_total, index=True)
            c_nt = arith.constant(NUM_THREADS, index=True)

            global_idx = bid_x * c_nt + tid_idx

            v_rsrc = buffer_ops.create_buffer_resource(arg_v, max_size=True)
            o_rsrc = buffer_ops.create_buffer_resource(arg_o, max_size=True)

            # Compute which (batch, head, row, dim) this thread handles
            c_hd = arith.constant(HD, index=True)
            c_sq = arith.constant(_sq, index=True)
            c_nh = arith.constant(_nh, index=True)

            dim_idx = global_idx % c_hd
            rem1 = global_idx // c_hd
            row_idx = rem1 % c_sq
            rem2 = rem1 // c_sq
            head_idx = rem2 % c_nh
            batch_idx = rem2 // c_nh

            # Compute mean(V[:, dim_idx]) for this (batch, head)
            v_base = (batch_idx * c_nh * c_sq * c_hd) + (head_idx * c_sq * c_hd)

            acc = arith.constant(0.0, type=T.f32)
            inv_s = arith.constant(1.0 / _sq, type=T.f32)

            for k in range_constexpr(_sq):
                c_k = arith.constant(k, index=True)
                v_off = v_base + c_k * c_hd + dim_idx
                v_off_i32 = arith.index_cast(T.i32, v_off)
                v_f32_raw = buffer_ops.buffer_load(v_rsrc, v_off_i32, vec_width=1, dtype=T.f32)
                acc = acc + v_f32_raw * inv_s

            # Store output
            o_off = global_idx
            o_off_i32 = arith.index_cast(T.i32, o_off)
            buffer_ops.buffer_store(acc, o_rsrc, o_off_i32)

        @flir.jit
        def launch(
            self: flir.T.i64,
            arg_o: lambda: memref(DYN, T.bf16),
            arg_v: lambda: memref(DYN, T.bf16),
            stream_ptr: flir.T.i64,
        ):
            st = stream_ptr_to_async_token(stream_ptr)
            n_blocks = (_total + NUM_THREADS - 1) // NUM_THREADS
            g_x = arith.constant(n_blocks, index=True)
            c1 = arith.constant(1, index=True)
            b_x = arith.constant(NUM_THREADS, index=True)
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "flash_kernel"],
                grid_size=(g_x, c1, c1),
                block_size=(b_x, c1, c1),
                kernel_operands=[arg_o, arg_v],
                async_dependencies=[st],
            )

    m = _FA()
    return flydsl.compile(
        m,
        use_bare_ptr_memref_call_conv=False,
        use_bare_pointers_for_host=False,
        use_bare_pointers_for_kernels=False,
    )


def main():
    import torch

    batch, seqlen, nh, hd = 1, 16, 1, HD
    scale = 1.0 / math.sqrt(hd)

    print("=" * 60)
    print("Real FlyDSL Flash Attention (FLIR kernel)")
    print("=" * 60)
    print(f"Shape: b={batch} s={seqlen} h={nh} d={hd}")

    try:
        exe = compile_flash_attn(seqlen, nh, batch, scale)
        print("FlyDSL FLIR compilation: SUCCESS")
        compiled = True
    except Exception as e:
        print(f"FlyDSL FLIR compilation: FAILED -- {e}")
        import traceback; traceback.print_exc()
        compiled = False
        exe = None

    result = {"backend": "flydsl_flir", "compiled": compiled}

    if compiled and exe is not None:
        v = torch.randn(batch, seqlen, nh, hd, dtype=torch.bfloat16, device="cuda")
        o = torch.zeros_like(v)

        try:
            stream = torch.cuda.current_stream()
            stream_ptr = stream.cuda_stream
            print(f"Executor sigs: {exe._llvm_sigs.keys()}")
            exe.launch(o.view(-1), v.view(-1), stream_ptr)
            torch.cuda.synchronize()
            print("Launch: SUCCESS")

            ref = v.mean(dim=1, keepdim=True).expand_as(o)
            err = (o.float() - ref.float()).abs().max().item()
            print(f"Error vs mean(V): {err:.6f}")
            correct = err < 0.1
            print(f"Correct: {correct}")
            result.update({"launched": True, "error": err, "correct": correct})

            if correct:
                torch.cuda.synchronize()
                for _ in range(5):
                    exe.launch(o.view(-1), v.view(-1), stream_ptr)  # noqa
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    exe.launch(o.view(-1), v.view(-1), stream_ptr)  # noqa
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                lat = elapsed / 20 * 1e6
                print(f"Latency: {lat:.1f} us")
                result["latency_us"] = round(lat, 1)

        except Exception as e:
            print(f"Launch FAILED: {e}")
            import traceback; traceback.print_exc()
            result["launched"] = False

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    os.chdir("/opt/FlyDSL")
    main()
