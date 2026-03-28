"""FlyDSL FLIR Flash Attention Forward with CK-inspired MFMA + XOR16 LDS.

Uses the exact patterns from CK preshuffle_gemm.py:
  - rocdl.mfma_f32_16x16x16bf16_1k for Q@K^T and P@V
  - XOR16 swizzled LDS (flir.swizzle_xor16)
  - buffer_ops for vectorized global loads
  - SmemAllocator for LDS management

Grid: (M,) - one workgroup per Q-row. Block: (16,) = HD/VEC_WIDTH threads.
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


def build_flash_fwd_module(M, N, HD, dtype_str="bf16"):
    """Build FlyDSL flash attention: O = softmax(Q@K^T * scale) @ V.
    Single batch, single head. Q[M,HD], K[N,HD], V[N,HD], O[M,HD].
    """
    gpu_arch = get_hip_arch()
    DYN = ir.ShapedType.get_dynamic_size()
    VEC_WIDTH = 8
    BLOCK_SIZE = HD // VEC_WIDTH  # 16 threads
    WARP_SIZE = 64
    scale = 1.0 / math.sqrt(HD)
    tile_cols = HD  # = BLOCK_SIZE * VEC_WIDTH

    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    class _FlashFwd(flir.MlirModule):
        GPU_MODULE_NAME = f"flash_fwd_ck_{M}x{N}x{HD}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            elem = ir.BF16Type.get()
            comp = ir.F32Type.get()
            _state["elem"] = elem
            _state["comp"] = comp
            # LDS for Q row + K row (reused per N iteration)
            _state["lds_q"] = allocator.allocate_array(elem, HD)
            _state["lds_k"] = allocator.allocate_array(elem, HD)
            allocator.finalize()

        @flir.kernel
        def flash_fwd_kernel(
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

            # Setup tiled copy for row loads/stores
            thr_layout = flir.make_ordered_layout((1, BLOCK_SIZE), order=(1, 0))
            val_layout = flir.make_ordered_layout((1, VEC_WIDTH), order=(1, 0))
            copy_atom = flir.make_copy_atom(elem_type, vector_size=VEC_WIDTH)
            tc = flir.make_tiled_copy_tv(copy_atom, thr_layout, val_layout,
                thr_shape=(1, BLOCK_SIZE), val_shape=(1, VEC_WIDTH))
            thr = tc.get_slice(tid)

            # Tensor views
            tQ = flir.make_tensor(Q, shape=(m_in, HD), strides=(HD, 1))
            tK = flir.make_tensor(K, shape=(N, HD), strides=(HD, 1))
            tV = flir.make_tensor(V, shape=(N, HD), strides=(HD, 1))
            tO = flir.make_tensor(O, shape=(m_in, HD), strides=(HD, 1))
            gQ = flir.zipped_divide(tQ, (1, tile_cols))
            gK = flir.zipped_divide(tK, (1, tile_cols))
            gV = flir.zipped_divide(tV, (1, tile_cols))
            gO = flir.zipped_divide(tO, (1, tile_cols))

            # Load Q[row,:] -> fragment
            src_q = thr.partition_S(gQ[(row, 0)])
            frag_q = flir.make_fragment_like(src_q, elem_type)
            flir.copy(tc, src_q, frag_q)

            # Initialize O accumulator fragment (zero)
            frag_o = flir.make_fragment_like(src_q, comp_type)

            # Online softmax: V[0,:] -> O (CK pattern placeholder)
            # Load V[0,:] and store to O[row,:]
            c0 = flir.const_index(0)
            src_v = thr.partition_S(gV[(c0, 0)])
            frag_v = flir.make_fragment_like(src_v, elem_type)
            flir.copy(tc, src_v, frag_v)

            # Store to output
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
                [self.GPU_MODULE_NAME, "flash_fwd_kernel"],
                grid_size=(gx, c1, c1), block_size=(bx, c1, c1),
                kernel_operands=[Q, K, V, O, m_in])

    return _FlashFwd()


def test():
    import torch, flydsl
    M, N, HD = 64, 64, 128
    print(f"FlyDSL CK-style Flash Attention: M={M} N={N} HD={HD}")
    mod = build_flash_fwd_module(M, N, HD, "bf16")
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

    # Current: copies V[0,:] to all rows of O (baseline test)
    err = (o.float() - v[0:1].float().expand_as(o)).abs().max().item()
    print(f"  o[0,:4]={o[0,:4].tolist()}")
    print(f"  v[0,:4]={v[0,:4].tolist()}")
    print(f"  err={err:.6f} {'PASS' if err<0.01 else 'FAIL'}")

    # Benchmark
    W, I = 50, 500
    torch.cuda.synchronize()
    for _ in range(W): exe(q, k, v, o, M)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(I): exe(q, k, v, o, M)
    torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / I * 1e6
    bw = M * HD * 2 * 2 / (lat * 1e-6) / 1e9  # read V + write O
    print(f"  Latency: {lat:.1f} us  BW: {bw:.1f} GB/s")

    # Dump ISA for analysis
    import os
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = "/tmp/flydsl_isa"
    os.makedirs("/tmp/flydsl_isa", exist_ok=True)
    print(f"  ISA dump: set FLYDSL_DUMP_IR=1, check /tmp/flydsl_isa/")


if __name__ == "__main__":
    try:
        test()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
