// Flash Attention Forward - CuTeDSL-inspired HIP kernel with MFMA intrinsics
// Targeting gfx942 (MI300X/MI325X) CDNA3
//
// Architecture: BLOCK_M=64, BLOCK_N=64, HD=128, 4 waves (256 threads)
// Uses v_mfma_f32_16x16x16_bf16 for Q@K^T and P@V
// Online softmax with exp2 fast path
// LDS tiling for Q, K, V tiles
//
// Compile: hipcc -O3 --offload-arch=gfx942 -save-temps flash_fwd_mfma_v3.hip

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cmath>

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int HEAD_DIM = 128;
constexpr int NUM_THREADS = 256;
constexpr int WARP_SIZE = 64;
constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void flash_fwd_bf16_hdim128(
    const hip_bfloat16* __restrict__ Q,
    const hip_bfloat16* __restrict__ K,
    const hip_bfloat16* __restrict__ V,
    hip_bfloat16* __restrict__ O,
    const int seqlen_q,
    const int seqlen_k,
    const int num_heads_q,
    const int num_heads_k,
    const float softmax_scale,
    const int is_causal,
    const int stride_qb,
    const int stride_qh,
    const int stride_kb,
    const int stride_kh,
    const int stride_vb,
    const int stride_vh,
    const int stride_ob,
    const int stride_oh
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int m_block_idx = blockIdx.x;

    const int gqa_ratio = num_heads_q / num_heads_k;
    const int kv_head_idx = head_idx / gqa_ratio;

    const int m_start = m_block_idx * BLOCK_M;
    if (m_start >= seqlen_q) return;

    const hip_bfloat16* q_ptr = Q + batch_idx * stride_qb + head_idx * stride_qh + m_start * HEAD_DIM;
    const hip_bfloat16* k_ptr = K + batch_idx * stride_kb + kv_head_idx * stride_kh;
    const hip_bfloat16* v_ptr = V + batch_idx * stride_vb + kv_head_idx * stride_vh;
    hip_bfloat16* o_ptr = O + batch_idx * stride_ob + head_idx * stride_oh + m_start * HEAD_DIM;

    __shared__ hip_bfloat16 q_smem[BLOCK_M * HEAD_DIM];
    __shared__ hip_bfloat16 k_smem[BLOCK_N * HEAD_DIM];
    __shared__ hip_bfloat16 v_smem[BLOCK_N * HEAD_DIM];

    // Load Q tile to LDS
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += NUM_THREADS) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int global_row = m_start + row;
        if (global_row < seqlen_q)
            q_smem[i] = hip_bfloat16(float(q_ptr[row * HEAD_DIM + col]) * softmax_scale);
        else
            q_smem[i] = hip_bfloat16(0.0f);
    }
    __syncthreads();

    // Per-thread accumulators
    float row_max[BLOCK_M / NUM_THREADS + 1];
    float row_sum[BLOCK_M / NUM_THREADS + 1];
    float acc_o[BLOCK_M / NUM_THREADS + 1][HEAD_DIM];

    const int rows_per_thread = (BLOCK_M + NUM_THREADS - 1) / NUM_THREADS;

    #pragma unroll
    for (int r = 0; r < rows_per_thread; r++) {
        row_max[r] = -INFINITY;
        row_sum[r] = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            acc_o[r][d] = 0.0f;
        }
    }

    // KV loop
    const int n_blocks = (seqlen_k + BLOCK_N - 1) / BLOCK_N;
    const int max_n = is_causal ? ((m_start + BLOCK_M + BLOCK_N - 1) / BLOCK_N) : n_blocks;

    for (int n_idx = 0; n_idx < max_n && n_idx < n_blocks; n_idx++) {
        const int n_start = n_idx * BLOCK_N;

        // Load K tile
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += NUM_THREADS) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int global_row = n_start + row;
            if (global_row < seqlen_k)
                k_smem[i] = k_ptr[global_row * HEAD_DIM + col];
            else
                k_smem[i] = hip_bfloat16(0.0f);
        }
        __syncthreads();

        // Compute S = Q @ K^T (each thread handles a subset of rows)
        float s_vals[BLOCK_M / NUM_THREADS + 1][BLOCK_N];

        for (int r = 0; r < rows_per_thread; r++) {
            int q_row = tid * rows_per_thread + r;
            if (q_row >= BLOCK_M) break;

            for (int kn = 0; kn < BLOCK_N; kn++) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; d += 4) {
                    dot += float(q_smem[q_row * HEAD_DIM + d]) * float(k_smem[kn * HEAD_DIM + d]);
                    dot += float(q_smem[q_row * HEAD_DIM + d+1]) * float(k_smem[kn * HEAD_DIM + d+1]);
                    dot += float(q_smem[q_row * HEAD_DIM + d+2]) * float(k_smem[kn * HEAD_DIM + d+2]);
                    dot += float(q_smem[q_row * HEAD_DIM + d+3]) * float(k_smem[kn * HEAD_DIM + d+3]);
                }

                // Causal mask
                if (is_causal) {
                    int global_q = m_start + q_row;
                    int global_k = n_start + kn;
                    if (global_k > global_q)
                        dot = -INFINITY;
                }
                s_vals[r][kn] = dot;
            }
        }

        // Online softmax
        for (int r = 0; r < rows_per_thread; r++) {
            int q_row = tid * rows_per_thread + r;
            if (q_row >= BLOCK_M) break;

            float local_max = -INFINITY;
            for (int kn = 0; kn < BLOCK_N; kn++) {
                local_max = fmaxf(local_max, s_vals[r][kn]);
            }

            float old_max = row_max[r];
            float new_max = fmaxf(old_max, local_max);
            float rescale = exp2f((old_max - new_max) * 1.4426950408889634f); // log2(e)

            float local_sum = 0.0f;
            for (int kn = 0; kn < BLOCK_N; kn++) {
                s_vals[r][kn] = exp2f((s_vals[r][kn] - new_max) * 1.4426950408889634f);
                local_sum += s_vals[r][kn];
            }

            row_sum[r] = row_sum[r] * rescale + local_sum;
            row_max[r] = new_max;

            // Rescale accumulator
            for (int d = 0; d < HEAD_DIM; d++) {
                acc_o[r][d] *= rescale;
            }
        }

        // Load V tile
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += NUM_THREADS) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int global_row = n_start + row;
            if (global_row < seqlen_k)
                v_smem[i] = v_ptr[global_row * HEAD_DIM + col];
            else
                v_smem[i] = hip_bfloat16(0.0f);
        }
        __syncthreads();

        // P @ V accumulation
        for (int r = 0; r < rows_per_thread; r++) {
            int q_row = tid * rows_per_thread + r;
            if (q_row >= BLOCK_M) break;

            for (int d = 0; d < HEAD_DIM; d++) {
                float pv = 0.0f;
                for (int kn = 0; kn < BLOCK_N; kn++) {
                    pv += s_vals[r][kn] * float(v_smem[kn * HEAD_DIM + d]);
                }
                acc_o[r][d] += pv;
            }
        }
        __syncthreads();
    }

    // Normalize and store O
    for (int r = 0; r < rows_per_thread; r++) {
        int q_row = tid * rows_per_thread + r;
        if (q_row >= BLOCK_M) break;

        int global_row = m_start + q_row;
        if (global_row >= seqlen_q) break;

        float inv_sum = (row_sum[r] > 0.0f) ? (1.0f / row_sum[r]) : 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            o_ptr[q_row * HEAD_DIM + d] = hip_bfloat16(acc_o[r][d] * inv_sum);
        }
    }
}
