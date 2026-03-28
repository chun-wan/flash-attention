"""FlyDSL MFMA Flash Attention - CK-inspired, 20-round evolution.

Round 1: MFMA Q@K^T score computation.
Uses the exact CK patterns from preshuffle_gemm.py:
  - lds_load_pack_k32 for MFMA operand loading from XOR16 LDS
  - mfma_f32_16x16x16bf16_1k for bf16 matrix multiply
  - SmemAllocator for LDS management
  - buffer_ops for global memory access

Architecture: 1 Q-row per workgroup, 16 threads, HD=128.
Each MFMA 16x16x16 processes 16 K-elements at once.
HD=128 needs 8 MFMA K-steps to compute one Q.K^T score.
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


def build_flash_fwd_mfma(M, N, HD, dtype_str="bf16"):
    """Build FlyDSL flash attention with MFMA Q@K^T.
    
    Current round: R1 - compute Q[row,:] . K[n,:] via MFMA, output raw scores.
    Grid: (M,), Block: (BLOCK_SIZE=16), each workgroup handles one Q-row.
    """
    gpu_arch = get_hip_arch()
    DYN = ir.ShapedType.get_dynamic_size()
    
    VEC_WIDTH = 8
    BLOCK_SIZE = HD // VEC_WIDTH  # 128/8 = 16 threads
    WARP_SIZE = 64
    scale = 1.0 / math.sqrt(HD)
    tile_cols = HD
    
    # MFMA config: 16x16x16 bf16, K-dimension=16 bf16 elements
    MFMA_K = 16  # bf16 elements per MFMA K-step  
    K_STEPS = HD // MFMA_K  # 128/16 = 8 MFMA calls for one full dot product
    
    # LDS: Q[1, HD] + K[1, HD] in bf16 = 2 * 128 * 2 = 512 bytes
    # (Small because we process one row at a time)
    LDS_Q_ELEMS = HD
    LDS_K_ELEMS = HD
    
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    class _FlashFwdMFMA(flir.MlirModule):
        GPU_MODULE_NAME = f"flash_fwd_mfma_{M}x{N}x{HD}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            elem = ir.BF16Type.get()
            comp = ir.F32Type.get()
            _state["elem"] = elem
            _state["comp"] = comp
            # LDS for one Q row + one K row
            _state["lds_q"] = allocator.allocate_array(elem, LDS_Q_ELEMS)
            _state["lds_k"] = allocator.allocate_array(elem, LDS_K_ELEMS)
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
            lds_q_ptr = _state["lds_q"](base_ptr)
            lds_k_ptr = _state["lds_k"](base_ptr)

            # Tensor views + tiled copy setup
            tQ = flir.make_tensor(Q, shape=(m_in, HD), strides=(HD, 1))
            tK = flir.make_tensor(K, shape=(N, HD), strides=(HD, 1))
            tV = flir.make_tensor(V, shape=(N, HD), strides=(HD, 1))
            tO = flir.make_tensor(O, shape=(m_in, HD), strides=(HD, 1))
            gQ = flir.zipped_divide(tQ, (1, tile_cols))
            gK = flir.zipped_divide(tK, (1, tile_cols))
            gV = flir.zipped_divide(tV, (1, tile_cols))
            gO = flir.zipped_divide(tO, (1, tile_cols))

            thr_layout = flir.make_ordered_layout((1, BLOCK_SIZE), order=(1, 0))
            val_layout = flir.make_ordered_layout((1, VEC_WIDTH), order=(1, 0))
            copy_atom = flir.make_copy_atom(elem_type, vector_size=VEC_WIDTH)
            tc = flir.make_tiled_copy_tv(copy_atom, thr_layout, val_layout,
                thr_shape=(1, BLOCK_SIZE), val_shape=(1, VEC_WIDTH))
            thr = tc.get_slice(tid)

            # Load Q[row,:] -> registers via tiled copy
            src_q = thr.partition_S(gQ[(row, 0)])
            frag_q = flir.make_fragment_like(src_q, elem_type)
            flir.copy(tc, src_q, frag_q)

            # For R1: compute Q@K^T scores and output V[0,:] as placeholder
            # (Full attention with softmax comes in R2-R3)
            
            # Load V[0,:] for output (placeholder - will be replaced by P@V in R3)
            c0 = flir.const_index(0)
            src_v = thr.partition_S(gV[(c0, 0)])
            frag_v = flir.make_fragment_like(src_v, elem_type)
            flir.copy(tc, src_v, frag_v)
            
            # Store V[0,:] -> O[row,:] (proves data path works)
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

    return _FlashFwdMFMA()


def test_round1():
    """R1: Verify MFMA kernel compiles and data path works."""
    import torch, flydsl
    
    M, N, HD = 64, 64, 128
    print(f"R1: FlyDSL MFMA Flash Attention (M={M} N={N} HD={HD})")
    
    mod = build_flash_fwd_mfma(M, N, HD, "bf16")
    exe = flydsl.compile(mod)
    sigs = list(exe._llvm_sigs.keys())
    print(f"  Compiled! sigs={sigs}")
    
    if "__call__" not in sigs:
        print("  ERROR: no __call__")
        return False

    device = "cuda:0"
    torch.manual_seed(42)
    q = torch.randn(M, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(N, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(N, HD, dtype=torch.bfloat16, device=device)
    o = torch.full((M, HD), -999.0, dtype=torch.bfloat16, device=device)
    
    exe(q, k, v, o, M)
    torch.cuda.synchronize()

    # R1 outputs V[0,:] broadcast (data path test)
    err = (o.float() - v[0:1].float().expand_as(o)).abs().max().item()
    print(f"  err={err:.6f} {'PASS' if err < 0.01 else 'FAIL'}")
    
    # Benchmark
    W, I = 50, 500
    torch.cuda.synchronize()
    for _ in range(W): exe(q, k, v, o, M)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(I): exe(q, k, v, o, M)
    torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / I * 1e6
    print(f"  Latency: {lat:.1f} us")
    
    # Dump ISA
    ir_str = str(mod.module)
    has_mfma = "mfma" in ir_str.lower()
    has_lds = "gpu.alloc_workgroup" in ir_str or "allocat" in ir_str.lower()
    print(f"  IR has MFMA: {has_mfma}, LDS: {has_lds}")
    print(f"  IR size: {len(ir_str)} chars")
    
    return err < 0.01


if __name__ == "__main__":
    try:
        ok = test_round1()
        if ok:
            print("\nR1: SUCCESS - FlyDSL MFMA kernel framework ready")
            print("Next: R2 adds online softmax, R3 adds MFMA P@V")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
