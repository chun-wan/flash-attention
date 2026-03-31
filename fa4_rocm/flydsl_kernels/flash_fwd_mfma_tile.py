"""
FlyDSL MFMA-Tiled Flash Attention.

Uses MFMA bf16 16x16x16 for Q@K^T and P@V.
64 threads (1 warp) per workgroup, each processes 16 Q-rows.
scf.for_ runtime KV loop. Online softmax.

For Q@K^T: MFMA with Q(row-major) and K(row-major) -> computes Q@K^T automatically.
Output mapping: lane t -> C[(t/16)*4+j, t%16].

Usage: cd /opt/FlyDSL && python /workspace/fa4_rocm/flydsl_kernels/flash_fwd_mfma_tile.py
"""
import os, sys, math, time, json, torch
sys.path.insert(0, "/opt/FlyDSL")
sys.path.insert(0, "/opt/FlyDSL/kernels")
os.chdir("/opt/FlyDSL")

import flydsl
from flydsl.utils import SmemAllocator, SmemPtr
from flydsl.dialects.ext import flir, arith, gpu, buffer_ops, vector, rocdl
from flydsl.dialects.ext import math as flydsl_math
from flydsl.dialects.ext.python_control_flow import range_constexpr
from _mlir import ir
from _mlir.dialects import scf
from flydsl.lang.ir.types import T, memref
from kernels.kernels_common import stream_ptr_to_async_token

HD = 128
BLOCK_M = 16
BLOCK_N = 16
WARP_SIZE = 64
MFMA_K = 16
K_STEPS = HD // MFMA_K  # 8
DYN = ir.ShapedType.get_dynamic_size()


def build_mfma_flash_attn(seqlen, num_heads, batch, softmax_scale):
    arch = "gfx942"
    _sq = seqlen
    _nh = num_heads
    _batch = batch
    _scale = float(softmax_scale)
    grid_m = (seqlen + BLOCK_M - 1) // BLOCK_M

    module_name = f"flash_mfma_tile_b{batch}s{seqlen}h{num_heads}"
    p_allocator = SmemAllocator(None, arch=arch)

    class _FA(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            # LDS layout:
            # [0 .. 255]: P tile 16x16 bf16 = 512 bytes = 256 bf16 elements
            # [256 .. 319]: softmax reduction max: 16 rows * 4 n_groups = 64 f32
            # [320 .. 383]: softmax reduction sum: 16 rows * 4 n_groups = 64 f32
            p_allocator.allocate_array(T.bf16, BLOCK_M * BLOCK_N + 256)  # extra for f32 reduction slots
            p_allocator.finalize()

        @flir.kernel
        def flash_kernel(
            self: flir.T.i64,
            arg_o: lambda: memref(DYN, T.f32),   # f32 output buffer (convert to bf16 on host)
            arg_q: lambda: memref(DYN, T.bf16),
            arg_k: lambda: memref(DYN, T.bf16),
            arg_v: lambda: memref(DYN, T.bf16),
        ):
            tid = gpu.thread_id("x")
            bid_m = gpu.block_id("x")
            bid_h = gpu.block_id("y")
            bid_b = gpu.block_id("z")

            tid_i32 = arith.index_cast(T.i32, tid)
            c16 = arith.constant(16, type=T.i32)
            c4 = arith.constant(4, type=T.i32)
            c2 = arith.constant(2, type=T.i32)
            c1_i32 = arith.constant(1, type=T.i32)

            lane = tid_i32 % arith.constant(WARP_SIZE, type=T.i32)
            m_idx = lane % c16          # row in tile (0-15)
            n_group = lane / c16        # output col group (0-3)

            m_block = arith.index_cast(T.i32, bid_m)
            head_i = arith.index_cast(T.i32, bid_h)
            batch_i = arith.index_cast(T.i32, bid_b)

            # (batch, seqlen, heads, hdim) strides in bf16 elements
            c_hd = arith.constant(HD, type=T.i32)
            c_sq = arith.constant(_sq, type=T.i32)
            c_nh = arith.constant(_nh, type=T.i32)
            stride_s = c_nh * c_hd
            stride_b = c_sq * stride_s

            m_start = m_block * arith.constant(BLOCK_M, type=T.i32)

            q_rsrc = buffer_ops.create_buffer_resource(arg_q, max_size=True)
            k_rsrc = buffer_ops.create_buffer_resource(arg_k, max_size=True)
            v_rsrc = buffer_ops.create_buffer_resource(arg_v, max_size=True)
            o_rsrc = buffer_ops.create_buffer_resource(arg_o, max_size=True)

            # Base offsets (in bf16 elements)
            q_base = batch_i * stride_b + (m_start + m_idx) * stride_s + head_i * c_hd
            k_head_base = batch_i * stride_b + head_i * c_hd  # K base for this (batch, head)

            zero_f = arith.constant(0.0, type=T.f32)
            neg_inf_f = arith.constant(float("-inf"), type=T.f32)
            scale_f = arith.constant(_scale, type=T.f32)
            log2e_f = arith.constant(1.4426950408889634, type=T.f32)

            # Output accumulators: 8 tiles of f32x4 (8 * 4 = 32 elements per thread,
            # but we need 16x128 = 2048 / 64 threads = 32 elements per thread for output)
            # Actually: for P@V, each MFMA step gives 4 output elements along HD dimension.
            # 8 MFMA steps * 4 = 32 f32 values per thread = covers HD=128 in some layout.
            # But the output mapping is transposed: element j at lane t goes to
            # O[(t/16)*4+j, t%16] for each V-tile step.
            # This means each thread contributes to 4 rows of a single column.
            # We need to accumulate across BLOCK_N KV blocks.

            # For P@V, we accumulate 8 MFMA tiles, each covering a different HD chunk.
            acc_pv = [arith.constant_vector(0.0, T.f32x4)] * K_STEPS

            # Loop state for softmax: each thread has 4 elements from the QK tile
            # Thread lane t has scores S[(t/16)*4+j, t%16] for the 16x16 QK tile
            # Online softmax: track max and sum per-element (4 values per thread)

            # scf.for_ KV loop
            c0 = arith.unwrap(arith.constant(0, index=True))
            c1 = arith.unwrap(arith.constant(1, index=True))
            n_blocks = (_sq + BLOCK_N - 1) // BLOCK_N
            sk = arith.unwrap(arith.constant(n_blocks, index=True))

            # iter_args: [max_vec(f32x4), sum_vec(f32x4), 8 acc_pv tiles(f32x4 each)]
            init_max = arith.constant_vector(float("-inf"), T.f32x4)
            init_sum = arith.constant_vector(0.0, T.f32x4)
            init_args = [arith.unwrap(init_max), arith.unwrap(init_sum)] + \
                         [arith.unwrap(a) for a in acc_pv]

            for iv, iter_vals, results in scf.for_(c0, sk, c1, iter_args=init_args):
                i_max = iter_vals[0]
                i_sum = iter_vals[1]
                i_accs = list(iter_vals[2:])

                n_start = arith.index_cast(T.i32, iv) * arith.constant(BLOCK_N, type=T.i32)
                k_row_base = k_head_base + n_start * stride_s

                # ======= Q @ K^T via 8 MFMA steps =======
                acc_qk = arith.constant_vector(0.0, T.f32x4)

                for ks in range_constexpr(K_STEPS):
                    k_off = ks * MFMA_K
                    # A operand: Q[m_idx, k_off + n_group*4 : +4]
                    q_dw = arith.unwrap((q_base + arith.constant(k_off, type=T.i32) + n_group * c4) / c2)
                    q0 = buffer_ops.buffer_load(q_rsrc, q_dw, vec_width=1, dtype=T.i32)
                    q1 = buffer_ops.buffer_load(q_rsrc, arith.unwrap(q_dw + c1_i32), vec_width=1, dtype=T.i32)
                    a_v4i16 = vector.bitcast(T.vec(4, T.i16), vector.from_elements(T.vec(2, T.i32), [q0, q1]))

                    # B operand: K[n_start + m_idx, k_off + n_group*4 : +4]
                    # (loaded same as A, MFMA transposes B internally)
                    k_elem = k_row_base + m_idx * stride_s + arith.constant(k_off, type=T.i32) + n_group * c4
                    k_dw = arith.unwrap(k_elem / c2)
                    k0 = buffer_ops.buffer_load(k_rsrc, k_dw, vec_width=1, dtype=T.i32)
                    k1 = buffer_ops.buffer_load(k_rsrc, arith.unwrap(k_dw + c1_i32), vec_width=1, dtype=T.i32)
                    b_v4i16 = vector.bitcast(T.vec(4, T.i16), vector.from_elements(T.vec(2, T.i32), [k0, k1]))

                    acc_qk = rocdl.mfma_f32_16x16x16bf16_1k(
                        T.f32x4, [a_v4i16, b_v4i16, acc_qk, 0, 0, 0]
                    )

                # (scaling done during LDS store below)

                # ======= Online Softmax via LDS: store scores, reload full row =======
                # Store all 16x16 scores to LDS as f32, then each thread reads its row
                log2e_c = arith.constant(1.4426950408889634, type=T.f32)

                lds_base = p_allocator.get_base()
                # Use f32 view for scores: 16x16 = 256 f32 starting at byte 0
                # (P tile bf16 will reuse same LDS space later)
                score_lds = SmemPtr(lds_base, 0, T.f32, shape=(256,)).get()

                # Store: lane t has S[(t/16)*4+j, t%16] -> LDS[row*16+col] as f32
                for j in range_constexpr(4):
                    s_row = n_group * c4 + arith.constant(j, type=T.i32)
                    s_col = m_idx
                    s_idx = arith.index_cast(T.index, s_row * c16 + s_col)
                    s_val = vector.extract(acc_qk, static_position=[j])
                    # Scale here
                    s_scaled = s_val * scale_f
                    vector.store(vector.broadcast(T.vec(1, T.f32), s_scaled), score_lds, [s_idx])

                gpu.barrier()

                # Each thread computes softmax for its 4 rows
                # Row r = n_group*4+j, read 16 elements: score_lds[r*16+0..15]
                new_max_elems = []
                p_elems_per_row = []
                sum_elems = []

                for j in range_constexpr(4):
                    s_row = n_group * c4 + arith.constant(j, type=T.i32)
                    # Find row max across 16 columns
                    row_max = arith.constant(float("-inf"), type=T.f32)
                    for c_col in range_constexpr(16):
                        idx = arith.index_cast(T.index, s_row * c16 + arith.constant(c_col, type=T.i32))
                        sv = vector.extract(vector.load_op(T.vec(1, T.f32), score_lds, [idx]), static_position=[0])
                        row_max = arith.maximum(row_max, sv)

                    # Online softmax: merge with running max
                    old_m_j = vector.extract(i_max, static_position=[j])
                    new_m_j = arith.maximum(old_m_j, row_max)
                    new_max_elems.append(new_m_j)

                    # Compute exp2 and sum for this row
                    row_sum = arith.constant(0.0, type=T.f32)
                    for c_col in range_constexpr(16):
                        idx = arith.index_cast(T.index, s_row * c16 + arith.constant(c_col, type=T.i32))
                        sv = vector.extract(vector.load_op(T.vec(1, T.f32), score_lds, [idx]), static_position=[0])
                        pv = flydsl_math.exp2(arith.unwrap((sv - new_m_j) * log2e_c))
                        row_sum = row_sum + pv

                    # Rescale factor for old accumulators
                    rescale_j = flydsl_math.exp2(arith.unwrap((old_m_j - new_m_j) * log2e_c))
                    old_sum_j = vector.extract(i_sum, static_position=[j])
                    new_sum_j = old_sum_j * rescale_j + row_sum
                    sum_elems.append(new_sum_j)

                    # Store the probability for this thread's column (m_idx) in this row
                    my_score_idx = arith.index_cast(T.index, s_row * c16 + m_idx)
                    my_sv = vector.extract(vector.load_op(T.vec(1, T.f32), score_lds, [my_score_idx]), static_position=[0])
                    my_p = flydsl_math.exp2(arith.unwrap((my_sv - new_m_j) * log2e_c))
                    p_elems_per_row.append(my_p)

                new_max = vector.from_elements(T.f32x4, new_max_elems)
                p_v = vector.from_elements(T.f32x4, p_elems_per_row)
                new_sum = vector.from_elements(T.f32x4, sum_elems)

                # Rescale old PV accumulators
                rescale_elems = []
                for j in range_constexpr(4):
                    old_m_j = vector.extract(i_max, static_position=[j])
                    new_m_j = vector.extract(new_max, static_position=[j])
                    re = flydsl_math.exp2(arith.unwrap((old_m_j - new_m_j) * log2e_c))
                    rescale_elems.append(re)
                rescale_v = vector.from_elements(T.f32x4, rescale_elems)

                # Rescale existing PV accumulators
                new_accs = []
                for acc_idx in range_constexpr(K_STEPS):
                    new_accs.append(i_accs[acc_idx] * rescale_v)

                # ======= P @ V via LDS transpose + MFMA =======
                # Store P to LDS row-major: LDS[row*16+col] = P[row,col]
                # Lane t has P[(t/16)*4+j, t%16] for j=0..3
                lds_base = p_allocator.get_base()
                p_lds = SmemPtr(lds_base, 0, T.bf16, shape=(BLOCK_M * BLOCK_N,)).get()

                for j in range_constexpr(4):
                    p_row = n_group * c4 + arith.constant(j, type=T.i32)
                    p_col = m_idx
                    lds_idx = arith.index_cast(T.index, p_row * c16 + p_col)
                    p_bf16 = arith.trunc_f(T.bf16, vector.extract(p_v, static_position=[j]))
                    vector.store(vector.broadcast(T.vec(1, T.bf16), p_bf16), p_lds, [lds_idx])

                gpu.barrier()

                # Reload P for MFMA srcA: P[m_idx, n_group*4:+4]
                # 4 bf16 from LDS[m_idx*16 + n_group*4 : +4]
                p_lds_off = arith.index_cast(T.index, m_idx * c16 + n_group * c4)
                # Load 4 bf16 as 2 i32 (since bf16 = 2 bytes, 4 bf16 = 8 bytes = 2 dwords)
                # Use vector.load of vec(4, bf16) then bitcast to vec(4, i16)
                p_loaded = vector.load_op(T.vec(4, T.bf16), p_lds, [p_lds_off])
                p_srcA = vector.bitcast(T.vec(4, T.i16), p_loaded)

                # P@V MFMA: C = P @ V^T_input (MFMA auto-transposes B)
                # To get P@V, pass V^T as srcB -> MFMA computes P @ (V^T)^T = P @ V
                # V is at [n_start+k, hd_dim], we need V^T[hd_dim, n_start+k]
                # V^T[d, k] = V[k, d]
                # For srcB at lane t: B^T[t%16, (t/16)*4:+4]
                # = V^T[t%16, (t/16)*4:+4] = V[(t/16)*4:+4, t%16]
                # So load V[(t/16)*4+j, t%16] = V[n_start + n_group*4+j, m_idx_hd]
                # where m_idx_hd is the HD dimension index

                for vs in range_constexpr(K_STEPS):
                    hd_off = arith.constant(vs * MFMA_K, type=T.i32)
                    # V srcB: each thread loads V[n_start+n_group*4:+4, hd_off+m_idx]
                    # These are 4 elements from 4 different KV rows at one HD column
                    # V[row, col] = flat[(batch_offset) + row * stride_s + col]
                    v_elems_i16 = []
                    for j in range_constexpr(4):
                        v_kv_row = n_start + n_group * c4 + arith.constant(j, type=T.i32)
                        v_hd_col = hd_off + m_idx
                        v_elem_off = k_head_base + v_kv_row * stride_s + v_hd_col
                        # Use byte-offset load: soffset_bytes = v_elem_off * 2
                        v_byte_off = arith.unwrap(v_elem_off * c2)
                        v_i32 = buffer_ops.buffer_load(
                            v_rsrc, arith.constant(0, type=T.i32),
                            vec_width=1, dtype=T.i32,
                            soffset_bytes=v_byte_off
                        )
                        # Low 16 bits is our bf16 (byte-aligned load)
                        v_i16 = arith.TruncIOp(T.i16, arith.unwrap(v_i32)).result
                        v_elems_i16.append(v_i16)

                    v_srcB = vector.from_elements(T.vec(4, T.i16), v_elems_i16)

                    new_accs[vs] = rocdl.mfma_f32_16x16x16bf16_1k(
                        T.f32x4, [p_srcA, v_srcB, new_accs[vs], 0, 0, 0]
                    )

                gpu.barrier()

                # (old scalar P@V code removed - replaced by MFMA P@V via LDS above)

                yield_args = [arith.unwrap(new_max), arith.unwrap(new_sum)] + \
                             [arith.unwrap(a) for a in new_accs]
                scf.YieldOp(yield_args)

            # ======= Normalize and Store =======
            final_sum = results[1]
            final_accs = list(results[2:])

            # Compute 1/sum for normalization
            inv_sum_elems = []
            for e in range_constexpr(4):
                s_elem = vector.extract(final_sum, static_position=[e])
                inv_s = arith.constant(1.0, type=T.f32) / s_elem
                inv_sum_elems.append(inv_s)
            inv_sum_v = vector.from_elements(T.f32x4, inv_sum_elems)

            # Store: MFMA output at lane t, tile vs, element j -> O[(t/16)*4+j, t%16]
            # But for PV output, the "column" is in the HD dimension:
            # PV tile vs covers HD[vs*16 : vs*16+16]
            # Element j in tile vs at lane t -> O_row = (t/16)*4+j, O_col(hd) = t%16 + ???
            # Actually: P@V MFMA has A=P (16x16) and B=V(16x16_hd_chunk).
            # Output D[lane] = C[(lane/16)*4+j, lane%16]
            # For PV: this is O[(lane/16)*4+j, lane%16 + vs*16]... No.
            #
            # Let me think again. P is f32x4 representing 16x16 tile in MFMA layout.
            # V chunk vs is V[:, vs*16:(vs+1)*16]. P@V_vs gives 16x16 output.
            # Output element at lane t = O[(t/16)*4+j, t%16] for this V chunk.
            # But we want this to contribute to output row m_start + q_row, column hd_col.
            # q_row comes from the P matrix (rows of P = rows of Q).
            # hd_col comes from V chunk number vs: column = t%16 + vs*16? No.
            #
            # MFMA P@V: output[i,j] = sum_k P[i,k] * V[k, vs*16+j]
            # At lane t: i = (t/16)*4..+4, j = t%16
            # So output is at row (t/16)*4+j, column t%16 of this 16x16 tile.
            # The V chunk vs covers HD columns vs*16 : vs*16+16.
            # So the actual output column = t%16 + vs*16? No -- j in the MFMA is 0-15
            # within this tile, which maps to VS's column indices.
            #
            # Actually: MFMA(P, V_vs) gives D where D[i,j] = sum_k P[i,k]*V_vs[k,j]
            # with i,j in [0,16). So the j maps to the j-th column of V_vs,
            # which is the (vs*16 + j)-th column of the full V.
            # At lane t: i = (t/16)*4+0..3, j = t%16.
            # So the output element is at O[q_row_for_i, vs*16 + t%16]
            # where q_row_for_i = (t/16)*4+element_idx.
            #
            # q_row_for_i maps to output row m_start + (t/16)*4+j

            # Output is f32 buffer: flat index = (batch*sq*nh*hd) + row*nh*hd + head*hd + d
            # In (batch, seqlen, heads, hdim) layout:
            # o_flat_f32[batch * sq * nh * hd + row * nh * hd + head * hd + d]
            o_stride_s_f32 = c_nh * c_hd  # heads * hdim (in f32 elements)
            o_stride_b_f32 = c_sq * o_stride_s_f32

            for vs in range_constexpr(K_STEPS):
                normalized = final_accs[vs] * inv_sum_v
                hd_col = arith.constant(vs * MFMA_K, type=T.i32) + m_idx

                for j in range_constexpr(4):
                    q_row = n_group * c4 + arith.constant(j, type=T.i32)
                    global_row = m_start + q_row
                    # f32 flat offset
                    o_f32_idx = arith.unwrap(batch_i * o_stride_b_f32 + global_row * o_stride_s_f32 + head_i * c_hd + hd_col)
                    elem_f32 = vector.extract(normalized, static_position=[j])
                    buffer_ops.buffer_store(elem_f32, o_rsrc, o_f32_idx)

        @flir.jit
        def launch(self: flir.T.i64,
                   arg_o: lambda: memref(DYN, T.f32),
                   arg_q: lambda: memref(DYN, T.bf16),
                   arg_k: lambda: memref(DYN, T.bf16),
                   arg_v: lambda: memref(DYN, T.bf16),
                   sp: flir.T.i64):
            st = stream_ptr_to_async_token(sp)
            gm = arith.constant(grid_m, index=True)
            gh = arith.constant(_nh, index=True)
            gb = arith.constant(_batch, index=True)
            c64 = arith.constant(WARP_SIZE, index=True)
            c1 = arith.constant(1, index=True)
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "flash_kernel"],
                grid_size=(gm, gh, gb),
                block_size=(c64, c1, c1),
                kernel_operands=[arg_o, arg_q, arg_k, arg_v],
                async_dependencies=[st],
            )

    m = _FA()
    return flydsl.compile(m)


def main():
    batch, seqlen, nh, hd = 1, 16, 1, HD
    scale = 1.0 / math.sqrt(hd)

    print("=" * 60)
    print("FlyDSL MFMA-Tiled Flash Attention")
    print("=" * 60)
    print(f"Shape: b={batch} s={seqlen} h={nh} d={hd}")

    exe = build_mfma_flash_attn(seqlen, nh, batch, scale)
    print("Compile: OK")

    torch.manual_seed(42)
    q = torch.randn(batch, seqlen, nh, hd, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, seqlen, nh, hd, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, seqlen, nh, hd, dtype=torch.bfloat16, device="cuda")
    o_f32 = torch.zeros(batch, seqlen, nh, hd, dtype=torch.float32, device="cuda")

    stream = torch.cuda.current_stream()
    exe.launch(o_f32.view(-1), q.view(-1), k.view(-1), v.view(-1), stream.cuda_stream)
    torch.cuda.synchronize()
    print("Launch: OK")

    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(),
        v.transpose(1, 2).float(), is_causal=False, scale=scale
    ).transpose(1, 2)

    err = (o_f32 - ref).abs().max().item()
    mean_err = (o_f32 - ref).abs().mean().item()
    has_nan = torch.isnan(o_f32).any().item()
    print(f"Max error: {err:.6f}, Mean: {mean_err:.6f}, NaN: {has_nan}")
    print(f"O[0,0,0,:8]:   {o_f32[0, 0, 0, :8].tolist()}")
    print(f"ref[0,0,0,:8]: {ref[0, 0, 0, :8].tolist()}")
    correct = err < 0.5 and not has_nan
    print(f"Correct: {correct}")

    if correct and seqlen >= 64:
        torch.cuda.synchronize()
        for _ in range(10):
            exe.launch(o_f32.view(-1), q.view(-1), k.view(-1), v.view(-1), stream.cuda_stream)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N = 50
        for _ in range(N):
            exe.launch(o_f32.view(-1), q.view(-1), k.view(-1), v.view(-1), stream.cuda_stream)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        lat = elapsed / N * 1e6
        flops = 4 * batch * seqlen * seqlen * nh * hd
        tf = flops / (elapsed / N) / 1e12
        print(f"TFLOPS: {tf:.4f} | Latency: {lat:.0f} us")
    else:
        tf, lat = 0.0, 0.0

    result = {"correct": correct, "error": round(err, 6), "tflops": round(tf, 4)}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
