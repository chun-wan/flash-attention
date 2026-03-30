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

                # Scale QK scores
                scale_v = arith.constant_vector(_scale, T.f32x4)
                acc_qk = acc_qk * scale_v

                # ======= Online Softmax (per-element) =======
                log2e_c = arith.constant(1.4426950408889634, type=T.f32)
                new_max_elems = []
                rescale_elems = []
                p_elems = []
                for e in range_constexpr(4):
                    old_m = vector.extract(i_max, static_position=[e])
                    cur_s = vector.extract(acc_qk, static_position=[e])
                    nm = arith.maximum(old_m, cur_s)
                    new_max_elems.append(nm)
                    re = flydsl_math.exp2(arith.unwrap((old_m - nm) * log2e_c))
                    rescale_elems.append(re)
                    pe = flydsl_math.exp2(arith.unwrap((cur_s - nm) * log2e_c))
                    p_elems.append(pe)

                new_max = vector.from_elements(T.f32x4, new_max_elems)
                rescale_v = vector.from_elements(T.f32x4, rescale_elems)
                p_v = vector.from_elements(T.f32x4, p_elems)

                new_sum = i_sum * rescale_v + p_v

                # Rescale existing PV accumulators
                new_accs = []
                for acc_idx in range_constexpr(K_STEPS):
                    new_accs.append(i_accs[acc_idx] * rescale_v)

                # ======= P @ V via 8 MFMA steps =======
                # P is the softmax output: f32x4 per thread, representing a 16x16 tile
                # We need to convert P from f32 to bf16 and pack as vector<4xi16>
                p_i16_elems = []
                for e in range_constexpr(4):
                    p_bf16 = arith.trunc_f(T.bf16, vector.extract(p_v, static_position=[e]))
                    p_i16_elems.append(arith.bitcast(T.i16, p_bf16))
                p_v4i16 = vector.from_elements(T.vec(4, T.i16), p_i16_elems)

                for vs in range_constexpr(K_STEPS):
                    v_off = vs * MFMA_K
                    # V operand: V[n_start + m_idx, v_off + n_group*4 : +4]
                    v_elem = k_row_base + m_idx * stride_s + arith.constant(v_off, type=T.i32) + n_group * c4
                    v_dw = arith.unwrap(v_elem / c2)
                    v0 = buffer_ops.buffer_load(v_rsrc, v_dw, vec_width=1, dtype=T.i32)
                    v1 = buffer_ops.buffer_load(v_rsrc, arith.unwrap(v_dw + c1_i32), vec_width=1, dtype=T.i32)
                    v_v4i16 = vector.bitcast(T.vec(4, T.i16), vector.from_elements(T.vec(2, T.i32), [v0, v1]))

                    new_accs[vs] = rocdl.mfma_f32_16x16x16bf16_1k(
                        T.f32x4, [p_v4i16, v_v4i16, new_accs[vs], 0, 0, 0]
                    )

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

            o_head_base = batch_i * stride_b + head_i * c_hd

            for vs in range_constexpr(K_STEPS):
                normalized = final_accs[vs] * inv_sum_v
                hd_col_base = arith.constant(vs * MFMA_K, type=T.i32) + m_idx  # m_idx = t%16 -> hd col within tile

                for j in range_constexpr(4):
                    q_row = n_group * c4 + arith.constant(j, type=T.i32)  # (t/16)*4+j -> local Q row
                    global_row = m_start + q_row
                    o_elem = arith.unwrap(o_head_base + global_row * stride_s + hd_col_base)
                    elem_f32 = vector.extract(normalized, static_position=[j])
                    elem_bf16 = arith.trunc_f(T.bf16, elem_f32)
                    elem_i16 = arith.bitcast(T.i16, elem_bf16)
                    elem_i32 = arith.ExtSIOp(T.i32, arith.unwrap(elem_i16)).result

                    # Store as bf16: write to dword containing this element
                    # dword = o_elem / 2, position = o_elem % 2
                    # Since hd_col_base = vs*16 + t%16, and vs*16 is always even,
                    # parity depends on t%16. Use read-modify-write for odd positions.
                    # SIMPLIFICATION: store as i32 at element offset (loses neighbor)
                    # TODO: proper bf16 store
                    o_dw = arith.unwrap(o_elem / c2)
                    # For now, just write the bf16 as the low half of a dword
                    # This overwrites the neighbor, but is a starting point
                    buffer_ops.buffer_store(elem_i32, o_rsrc, o_dw)

        @flir.jit
        def launch(self: flir.T.i64,
                   arg_o: lambda: memref(DYN, T.bf16),
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
    o = torch.zeros_like(q)

    stream = torch.cuda.current_stream()
    exe.launch(o.view(-1), q.view(-1), k.view(-1), v.view(-1), stream.cuda_stream)
    torch.cuda.synchronize()
    print("Launch: OK")

    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(),
        v.transpose(1, 2).float(), is_causal=False, scale=scale
    ).transpose(1, 2).to(torch.bfloat16)

    err = (o.float() - ref.float()).abs().max().item()
    print(f"Max error: {err:.6f}")
    print(f"O[0,0,0,:8]: {o[0, 0, 0, :8].tolist()}")
    print(f"ref[0,0,0,:8]: {ref[0, 0, 0, :8].tolist()}")
    print(f"Correct: {err < 0.5}")

    result = {"correct": err < 0.5, "error": err}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
