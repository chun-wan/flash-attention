"""Test MFMA bf16 with correct vector<4xi16> operand type."""
import sys, os
sys.path.insert(0, "/opt/FlyDSL")
sys.path.insert(0, "/opt/FlyDSL/kernels")
os.chdir("/opt/FlyDSL")

import flydsl
from flydsl.dialects.ext import flir, arith, gpu, buffer_ops, vector, rocdl
from flydsl.dialects.ext.python_control_flow import range_constexpr
from _mlir import ir
from flydsl.lang.ir.types import T, memref
from kernels.kernels_common import stream_ptr_to_async_token

DYN = ir.ShapedType.get_dynamic_size()

class _T(flir.MlirModule):
    GPU_MODULE_NAME = "test_mfma_bf16"
    GPU_MODULE_TARGETS = [
        '#rocdl.target<chip = "gfx942", abi = "500", features = "+sramecc,+xnack">'
    ]
    def init_gpu_module(self):
        pass

    @flir.kernel
    def kern(self: flir.T.i64,
             arg_a: lambda: memref(DYN, T.bf16),
             arg_b: lambda: memref(DYN, T.bf16),
             arg_c: lambda: memref(DYN, T.f32)):
        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=True)

        tid = gpu.thread_id("x")
        c0 = arith.constant(0, type=T.i32)

        # Load 2 dwords = 4 bf16 for MFMA A operand
        a_i32_0 = buffer_ops.buffer_load(a_rsrc, c0, vec_width=1, dtype=T.i32)
        a_i32_1 = buffer_ops.buffer_load(a_rsrc, arith.constant(1, type=T.i32), vec_width=1, dtype=T.i32)

        # Pack to vector<2xi32> -> bitcast to vector<4xi16> (NOT i64!)
        a_v2i32 = vector.from_elements(T.vec(2, T.i32), [a_i32_0, a_i32_1])
        a_v4i16 = vector.bitcast(T.vec(4, T.i16), a_v2i32)

        b_i32_0 = buffer_ops.buffer_load(b_rsrc, c0, vec_width=1, dtype=T.i32)
        b_i32_1 = buffer_ops.buffer_load(b_rsrc, arith.constant(1, type=T.i32), vec_width=1, dtype=T.i32)
        b_v2i32 = vector.from_elements(T.vec(2, T.i32), [b_i32_0, b_i32_1])
        b_v4i16 = vector.bitcast(T.vec(4, T.i16), b_v2i32)

        acc = arith.constant_vector(0.0, T.f32x4)

        # MFMA bf16 16x16x16 with vector<4xi16> operands
        result = rocdl.mfma_f32_16x16x16bf16_1k(
            T.f32x4, [a_v4i16, b_v4i16, acc, 0, 0, 0]
        )

        # Store result[0] to verify
        elem0 = vector.extract(result, static_position=[0])
        tid_i32 = arith.index_cast(T.i32, tid)
        buffer_ops.buffer_store(elem0, c_rsrc, tid_i32)

    @flir.jit
    def launch(self: flir.T.i64,
               arg_a: lambda: memref(DYN, T.bf16),
               arg_b: lambda: memref(DYN, T.bf16),
               arg_c: lambda: memref(DYN, T.f32),
               sp: flir.T.i64):
        st = stream_ptr_to_async_token(sp)
        c1 = arith.constant(1, index=True)
        c64 = arith.constant(64, index=True)
        flir.gpu_ext.LaunchFuncOp(
            ["test_mfma_bf16", "kern"],
            grid_size=(c1, c1, c1),
            block_size=(c64, c1, c1),
            kernel_operands=[arg_a, arg_b, arg_c],
            async_dependencies=[st],
        )

m = _T()
exe = flydsl.compile(m)
print("MFMA bf16 COMPILE: SUCCESS")

import torch
a = torch.ones(256, dtype=torch.bfloat16, device="cuda")
b = torch.ones(256, dtype=torch.bfloat16, device="cuda")
c = torch.zeros(64, dtype=torch.float32, device="cuda")
exe.launch(a, b, c, torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()
print(f"MFMA bf16 LAUNCH: SUCCESS")
print(f"c[0] = {c[0].item()} (expected 16.0 for 1*1 with K=16)")
print(f"c[:4] = {c[:4].tolist()}")
