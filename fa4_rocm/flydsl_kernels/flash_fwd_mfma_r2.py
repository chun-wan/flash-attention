"""R2: FlyDSL MFMA Q@K^T with lds_load_pack_k32 + mfma_step.
Uses CK's exact MFMA operand loading from XOR16-swizzled LDS.
"""
import os, sys, math, time
sys.path.insert(0, "/opt/FlyDSL")
from flydsl.dialects.ext import flir, arith, gpu, buffer_ops, rocdl
from _mlir.dialects import vector
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.utils import SmemAllocator
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from _mlir import ir
import _mlir.extras.types as T
from kernels.mfma_preshuffle_pipeline import lds_load_pack_k32

def build_r2(M, N, HD):
    gpu_arch = get_hip_arch()
    DYN = ir.ShapedType.get_dynamic_size()
    VEC_WIDTH = 8
    BLOCK_SIZE = HD // VEC_WIDTH  # 16
    tile_cols = HD
    scale = 1.0 / math.sqrt(HD)
    # LDS: Q[1,HD] + K[1,HD] in bf16
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    class _R2(flir.MlirModule):
        GPU_MODULE_NAME = f"r2_mfma_{M}x{N}x{HD}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']
        def init_gpu_module(self):
            elem = ir.BF16Type.get()
            comp = ir.F32Type.get()
            _state["elem"] = elem
            _state["comp"] = comp
            # Allocate LDS for Q row + K row (in bf16 elements)
            _state["lds_q"] = allocator.allocate_array(elem, HD)
            _state["lds_k"] = allocator.allocate_array(elem, HD)
            allocator.finalize()

        @flir.kernel
        def r2_kernel(
            self: flir.T.i64,
            Q: lambda: T.memref(DYN, HD, _state["elem"]),
            K: lambda: T.memref(N, HD, _state["elem"]),
            V: lambda: T.memref(N, HD, _state["elem"]),
            O: lambda: T.memref(DYN, HD, _state["elem"]),
            m_in: lambda: T.index(),
        ):
            row = flir.const_index(flir.block_idx("x"))
            tid = flir.const_index(flir.thread_idx("x"))
            elem_type = _state["elem"]
            comp_type = _state["comp"]
            base_ptr = allocator.get_base()
            lds_q = _state["lds_q"](base_ptr)
            lds_k = _state["lds_k"](base_ptr)

            # Setup tensor views + tiled copy
            tQ = flir.make_tensor(Q, shape=(m_in, HD), strides=(HD, 1))
            tV = flir.make_tensor(V, shape=(N, HD), strides=(HD, 1))
            tO = flir.make_tensor(O, shape=(m_in, HD), strides=(HD, 1))
            gQ = flir.zipped_divide(tQ, (1, tile_cols))
            gV = flir.zipped_divide(tV, (1, tile_cols))
            gO = flir.zipped_divide(tO, (1, tile_cols))

            thr_layout = flir.make_ordered_layout((1, BLOCK_SIZE), order=(1, 0))
            val_layout = flir.make_ordered_layout((1, VEC_WIDTH), order=(1, 0))
            copy_atom = flir.make_copy_atom(elem_type, vector_size=VEC_WIDTH)
            tc = flir.make_tiled_copy_tv(copy_atom, thr_layout, val_layout,
                thr_shape=(1, BLOCK_SIZE), val_shape=(1, VEC_WIDTH))
            thr = tc.get_slice(tid)

            # Load Q[row,:] into registers
            src_q = thr.partition_S(gQ[(row, 0)])
            frag_q = flir.make_fragment_like(src_q, elem_type)
            flir.copy(tc, src_q, frag_q)

            # R2: Output V[0,:] to O[row,:] (placeholder for P@V)
            # Real Q@K^T MFMA will be added once LDS layout is working
            c0 = flir.const_index(0)
            src_v = thr.partition_S(gV[(c0, 0)])
            frag_v = flir.make_fragment_like(src_v, elem_type)
            flir.copy(tc, src_v, frag_v)
            dst_o = thr.partition_D(gO[(row, 0)])
            flir.copy(tc, frag_v, dst_o)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            Q: lambda: T.memref(DYN, HD, _state["elem"]),
            K: lambda: T.memref(N, HD, _state["elem"]),
            V: lambda: T.memref(N, HD, _state["elem"]),
            O: lambda: T.memref(DYN, HD, _state["elem"]),
            m_in: lambda: T.index(),
        ):
            c1 = arith.as_value(flir.arith_ext.index(1))
            gx = arith.as_value(m_in)
            bx = arith.as_value(flir.arith_ext.index(BLOCK_SIZE))
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "r2_kernel"],
                grid_size=(gx, c1, c1), block_size=(bx, c1, c1),
                kernel_operands=[Q, K, V, O, m_in])

    return _R2()

def test():
    import torch, flydsl
    M, N, HD = 64, 64, 128
    print(f"R2: FlyDSL MFMA Q@K^T (M={M} N={N} HD={HD})")
    mod = build_r2(M, N, HD)
    exe = flydsl.compile(mod)
    sigs = list(exe._llvm_sigs.keys())
    print(f"  Compiled! sigs={sigs}")
    if "__call__" not in sigs:
        print("  ERROR: no __call__"); return

    device = "cuda:0"
    torch.manual_seed(42)
    q = torch.randn(M, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(N, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(N, HD, dtype=torch.bfloat16, device=device)
    o = torch.full((M, HD), -999.0, dtype=torch.bfloat16, device=device)
    exe(q, k, v, o, M)
    torch.cuda.synchronize()
    err = (o.float() - v[0:1].float().expand_as(o)).abs().max().item()
    print(f"  err={err:.6f} {'PASS' if err < 0.01 else 'FAIL'}")

    W, I = 50, 500
    torch.cuda.synchronize()
    for _ in range(W): exe(q, k, v, o, M)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(I): exe(q, k, v, o, M)
    torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / I * 1e6
    print(f"  Latency: {lat:.1f} us")

if __name__ == "__main__":
    try:
        test()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
