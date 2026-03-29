"""Test MFMA IR generation to debug type mismatch."""
import sys, os
sys.path.insert(0, "/opt/FlyDSL")
sys.path.insert(0, "/opt/FlyDSL/kernels")
os.chdir("/opt/FlyDSL")

import flydsl
from flydsl.dialects.ext import flir, arith, gpu, buffer_ops, vector, rocdl
from _mlir import ir
from flydsl.lang.ir.types import T, memref
from kernels.kernels_common import stream_ptr_to_async_token

DYN = ir.ShapedType.get_dynamic_size()

class _T(flir.MlirModule):
    GPU_MODULE_NAME = "test_mfma3"
    GPU_MODULE_TARGETS = [
        '#rocdl.target<chip = "gfx942", abi = "500", features = "+sramecc,+xnack">'
    ]
    def init_gpu_module(self):
        pass

    @flir.kernel
    def kern(self: flir.T.i64, arg_c: lambda: memref(DYN, T.f32)):
        # Use f16 MFMA (better LLVM lowering support than bf16_1k)
        one_f16 = arith.constant(1.0, type=T.f16)
        a_vec = vector.from_elements(T.vec(4, T.f16), [one_f16] * 4)
        b_vec = vector.from_elements(T.vec(4, T.f16), [one_f16] * 4)

        # Pack to i64
        a_i64v = vector.bitcast(T.vec(1, T.i64), a_vec)
        a_i64 = vector.extract(a_i64v, static_position=[0])
        b_i64v = vector.bitcast(T.vec(1, T.i64), b_vec)
        b_i64 = vector.extract(b_i64v, static_position=[0])

        acc = arith.constant_vector(0.0, T.f32x4)
        result = rocdl.mfma_f32_16x16x16f16(
            T.f32x4, [a_i64, b_i64, acc, 0, 0, 0]
        )

        # Just verify compilation succeeds -- don't store

    @flir.jit
    def launch(self: flir.T.i64, arg_c: lambda: memref(DYN, T.f32), sp: flir.T.i64):
        st = stream_ptr_to_async_token(sp)
        c1 = arith.constant(1, index=True)
        c64 = arith.constant(64, index=True)
        flir.gpu_ext.LaunchFuncOp(
            ["test_mfma3", "kern"],
            grid_size=(c1, c1, c1),
            block_size=(c64, c1, c1),
            kernel_operands=[arg_c],
            async_dependencies=[st],
        )

m = _T()

# Print MFMA IR
asm = m.module.operation.get_asm()
for line in asm.split("\n"):
    if "mfma" in line.lower() or "rocdl" in line.lower():
        print(line.strip())

# Try to compile
try:
    exe = flydsl.compile(m)
    print("MFMA COMPILE: SUCCESS")

    import torch
    c = torch.zeros(64 * 4, dtype=torch.float32, device="cuda")
    exe.launch(c, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    print("MFMA LAUNCH: SUCCESS")
except Exception as e:
    print(f"MFMA COMPILE/LAUNCH FAILED: {e}")
