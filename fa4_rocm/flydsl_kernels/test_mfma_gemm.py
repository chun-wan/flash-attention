"""
Test MFMA 16x16 tile matmul: A[16,16] @ B[16,16] -> C[16,16]
using rocdl.mfma_f32_16x16x16bf16_1k with 64 cooperative threads.

Goal: verify thread-to-element mapping and correctness against torch matmul.
"""
import os, sys, math, torch
sys.path.insert(0, "/opt/FlyDSL")
sys.path.insert(0, "/opt/FlyDSL/kernels")
os.chdir("/opt/FlyDSL")

import flydsl
from flydsl.dialects.ext import flir, arith, gpu, buffer_ops, vector, rocdl
from flydsl.dialects.ext.python_control_flow import range_constexpr
from _mlir import ir
from _mlir.dialects import scf
from flydsl.lang.ir.types import T, memref
from kernels.kernels_common import stream_ptr_to_async_token

DYN = ir.ShapedType.get_dynamic_size()

# MFMA 16x16x16 bf16: 64 threads, each has 4 output elements
# Thread t: output C[t%16, (t//16)*4 + 0..3]
# A operand: thread t reads A[t%16, (t//16)*4 : (t//16)*4+4] (4 bf16 = vector<4xi16>)
# B operand: thread t reads B[(t//16)*4 : (t//16)*4+4, t%16] -- column-major for B

class _GEMM(flir.MlirModule):
    GPU_MODULE_NAME = "test_mfma_gemm"
    GPU_MODULE_TARGETS = [
        '#rocdl.target<chip = "gfx942", abi = "500", features = "+sramecc,+xnack">'
    ]
    def init_gpu_module(self):
        pass

    @flir.kernel
    def gemm_kernel(
        self: flir.T.i64,
        arg_a: lambda: memref(DYN, T.bf16),  # A[16, K] row-major, flat
        arg_b: lambda: memref(DYN, T.bf16),  # B[K, 16] row-major, flat
        arg_c: lambda: memref(DYN, T.f32),   # C[16, 16] row-major, flat
    ):
        tid = gpu.thread_id("x")
        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=True)

        tid_i32 = arith.index_cast(T.i32, tid)
        c16 = arith.constant(16, type=T.i32)

        lane = tid_i32 % arith.constant(64, type=T.i32)
        m_idx = lane % c16             # row 0-15
        n_group = lane / c16           # 0-3 (which group of 4 output cols)

        acc = arith.constant_vector(0.0, T.f32x4)

        # K=16: single MFMA step
        # A operand: 4 consecutive bf16 from A[m_idx, n_group*4 : n_group*4+4]
        # In dword terms: A is [16, 16] bf16 = [16, 8] dwords
        # A[m_idx, n_group*4..n_group*4+3] = 2 consecutive dwords starting at
        #   dword_off = m_idx * 8 + n_group * 2
        a_dw_off = arith.unwrap(m_idx * arith.constant(8, type=T.i32) + n_group * arith.constant(2, type=T.i32))
        a_dw0 = buffer_ops.buffer_load(a_rsrc, a_dw_off, vec_width=1, dtype=T.i32)
        a_dw1 = buffer_ops.buffer_load(a_rsrc, arith.unwrap(a_dw_off + arith.constant(1, type=T.i32)), vec_width=1, dtype=T.i32)
        a_v2i32 = vector.from_elements(T.vec(2, T.i32), [a_dw0, a_dw1])
        a_v4i16 = vector.bitcast(T.vec(4, T.i16), a_v2i32)

        # B operand: B is [16, 16], we need B[(n_group*4):(n_group*4+4), m_idx]
        # But MFMA B is column-oriented: thread t reads B[k, n] where k = (t//16)*4..+4
        # Actually for mfma C = A @ B where A is row, B is row:
        # Thread reads B[n_group*4 + j, m_idx] for j=0..3 (4 elements from 4 different rows)
        # Since B is row-major [16, 16], B[row, col] = flat[row*16 + col]
        # So B[n_group*4+j, m_idx] = flat[(n_group*4+j)*16 + m_idx]
        # In dwords: each element is 1 bf16 = 0.5 dword, need to load pairs
        # Load B[n_group*4, m_idx] and B[n_group*4+1, m_idx] as one dword
        # dword_off = ((n_group*4)*16 + m_idx) / 2 if m_idx is even
        # This is complex. Simpler: load as individual bf16 via scalar loads

        # Actually, for MFMA the B operand layout is:
        # B operand: 4 bf16 from column n_group*4+0..3 at row m_idx
        # Wait - the MFMA documentation says for C = A * B:
        # srcA[lane] reads A[lane%M, (lane/M)*4 : (lane/M)*4+4]  (4 consecutive in K dim)
        # srcB[lane] reads B[(lane/N)*4 : (lane/N)*4+4, lane%N]  (4 from K dim, column lane%N)
        # For 16x16x16: M=N=16, K=16
        # srcA[lane] = A[lane%16, (lane/16)*4 : +4]
        # srcB[lane] = B[(lane/16)*4 : +4, lane%16]  -- 4 elements from K dim at column lane%16
        # D[lane] = C[lane%16, (lane/16)*4 : +4]

        # So B operand: thread t reads B[k_off+0..3, t%16] where k_off = (t/16)*4
        # B is row-major [16, 16] bf16: B[k, n] = flat[k*16 + n]
        # 4 elements: B[k_off+j, m_idx] at flat[(k_off+j)*16 + m_idx] for j=0..3
        # Each is 1 bf16. Pack 4 bf16 into vector<4xi16>

        b_n = m_idx  # column in B = thread's row index (lane % 16)
        b_k_off = n_group * arith.constant(4, type=T.i32)

        # Load 4 bf16 from B column
        # B is [K=16, N=16] row-major. MFMA srcB needs B[(t/16)*4:+4, t%16].
        # Pass B^T[16,16] instead, so B^T[n, k] = B[k, n].
        # Then srcB reads from B^T the same way srcA reads from A:
        # B^T[b_n, b_k_off*4 : b_k_off*4+4] = 4 consecutive bf16 in row b_n
        # This is 2 consecutive dwords at dword_off = b_n * 8 + b_k_off * 2
        # (same formula as A loading)
        bt_dw_off = arith.unwrap(b_n * arith.constant(8, type=T.i32) + n_group * arith.constant(2, type=T.i32))
        b_dw0 = buffer_ops.buffer_load(b_rsrc, bt_dw_off, vec_width=1, dtype=T.i32)
        b_dw1 = buffer_ops.buffer_load(b_rsrc, arith.unwrap(bt_dw_off + arith.constant(1, type=T.i32)), vec_width=1, dtype=T.i32)
        b_v2i32 = vector.from_elements(T.vec(2, T.i32), [b_dw0, b_dw1])
        b_v4i16 = vector.bitcast(T.vec(4, T.i16), b_v2i32)

        # MFMA
        result = rocdl.mfma_f32_16x16x16bf16_1k(
            T.f32x4, [a_v4i16, b_v4i16, acc, 0, 0, 0]
        )

        # Store result: MFMA output at lane t is C[(t/16)*4+j, t%16] (transposed from expected)
        for j in range_constexpr(4):
            cj = arith.constant(j, type=T.i32)
            row_out = n_group * arith.constant(4, type=T.i32) + cj
            col_out = m_idx
            c_flat = arith.unwrap(row_out * c16 + col_out)
            elem = vector.extract(result, static_position=[j])
            buffer_ops.buffer_store(elem, c_rsrc, c_flat)

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
            ["test_mfma_gemm", "gemm_kernel"],
            grid_size=(c1, c1, c1),
            block_size=(c64, c1, c1),
            kernel_operands=[arg_a, arg_b, arg_c],
            async_dependencies=[st],
        )

m = _GEMM()
exe = flydsl.compile(m)
print("MFMA GEMM compile: OK")

# Now B is loaded as B^T (transposed), so C = A @ B^T
# Pass B^T to the kernel
torch.manual_seed(42)

# Test 1: A=I, B^T=I -> C = I@I = I
print("\n--- Test 1: A=I, B^T=I -> C=I ---")
A = torch.eye(16, dtype=torch.bfloat16, device="cuda")
Bt = torch.eye(16, dtype=torch.bfloat16, device="cuda")  # B^T
C = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
exe.launch(A.view(-1), Bt.view(-1), C.view(-1), torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()
print(f"C diag: {C.diag()[:4].tolist()}")
print(f"C[0,:4]: {C[0,:4].tolist()}")

# Test 2: A=ones, B^T=ones -> C=16
print("\n--- Test 2: A=1, B^T=1 -> C=16 ---")
A = torch.ones(16, 16, dtype=torch.bfloat16, device="cuda")
Bt = torch.ones(16, 16, dtype=torch.bfloat16, device="cuda")
C = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
exe.launch(A.view(-1), Bt.view(-1), C.view(-1), torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()
print(f"C[0,:4]: {C[0,:4].tolist()} (expect 16)")

# Test 3: Random - C = A @ B^T
print("\n--- Test 3: Random C = A @ B^T ---")
torch.manual_seed(42)
A = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
B = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
Bt = B.T.contiguous()
C = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
exe.launch(A.view(-1), Bt.view(-1), C.view(-1), torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()
ref = A.float() @ B.float().T
err = (C - ref).abs().max().item()
print(f"C vs A@B^T: max_err = {err:.6f}")
print(f"C[0,:4]:     {C[0,:4].tolist()}")
print(f"A@B^T[0,:4]: {ref[0,:4].tolist()}")
print(f"CORRECT: {err < 0.1}")
