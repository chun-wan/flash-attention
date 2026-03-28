// FA4 ROCm Forward Attention Kernel for gfx942 (MI300X / MI325X)
//
// MFMA-based tiled FlashAttention with online softmax.
//
//   BLOCK_M  = 64   query rows per workgroup
//   BLOCK_N  = 64   key columns per KV-block iteration
//   HEAD_DIM = 128   head dimension (compile-time)
//   4 waves x 64 threads = 256 threads per workgroup
//   Each wave owns 16 rows (1 MFMA M-tile).
//
// LDS budget (bytes):
//   Q  : BLOCK_M * HEAD_DIM * 2 = 16 384
//   K  : BLOCK_N * HEAD_DIM * 2 = 16 384   (reused for P during P@V)
//   V  : BLOCK_N * HEAD_DIM * 2 = 16 384
//   max: Q + max(K, P+V) ≤ 49 152 < 65 536

#include "flash_common.h"
#include "online_softmax.h"
#include <hip/hip_runtime.h>

#ifndef BLOCK_M
#define BLOCK_M 64
#endif
#ifndef BLOCK_N
#define BLOCK_N 64
#endif
#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

#define NUM_WARPS    4
#define NUM_THREADS  (NUM_WARPS * WARP_SIZE)

// Number of MFMA 16x16 tiles in each dimension
#define M_TILES      (BLOCK_M / MFMA_M)       // 4   (1 per wave)
#define N_TILES      (BLOCK_N / MFMA_N)        // 4
#define K_STEPS      (HEAD_DIM / MFMA_K)       // 8
#define D_TILES      (HEAD_DIM / MFMA_N)       // 8
#define KN_STEPS     (BLOCK_N  / MFMA_K)       // 4

#define LDS_Q_BYTES  (BLOCK_M * HEAD_DIM * 2)
#define LDS_K_BYTES  (BLOCK_N * HEAD_DIM * 2)
#define LDS_V_BYTES  (BLOCK_N * HEAD_DIM * 2)
#define LDS_P_BYTES  (BLOCK_M * BLOCK_N  * 2)
#define LDS_TOTAL    (LDS_Q_BYTES + LDS_K_BYTES + LDS_V_BYTES)

static_assert(LDS_TOTAL <= 65536, "LDS overflow");

// ======================================================================
// Main kernel
// ======================================================================
template <typename half_type, bool IS_CAUSAL>
__global__ __launch_bounds__(NUM_THREADS)
void flash_fwd_kernel(
    const half_type* __restrict__ Q,
    const half_type* __restrict__ K,
    const half_type* __restrict__ V,
    half_type*       __restrict__ O,
    float*           __restrict__ LSE,
    int batch_size, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_k, int head_dim,
    float softmax_scale)
{
    const int m_block   = blockIdx.x;
    const int bh        = blockIdx.y;
    const int batch_idx = bh / num_heads_q;
    const int head_idx  = bh % num_heads_q;
    const int kv_head   = head_idx / (num_heads_q / num_heads_k);

    const int tid     = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;            // 0..3

    const int m_start      = m_block * BLOCK_M;
    const int valid_m      = min(BLOCK_M, seqlen_q - m_start);

    // Strides for [batch, seqlen, heads, hdim]
    const int q_batch_s = seqlen_q * num_heads_q * head_dim;
    const int q_seq_s   = num_heads_q * head_dim;
    const int q_head_s  = head_dim;
    const int k_batch_s = seqlen_k * num_heads_k * head_dim;
    const int k_seq_s   = num_heads_k * head_dim;
    const int k_head_s  = head_dim;

    const half_type* Q_ptr = Q + batch_idx * q_batch_s + head_idx  * q_head_s;
    const half_type* K_ptr = K + batch_idx * k_batch_s + kv_head   * k_head_s;
    const half_type* V_ptr = V + batch_idx * k_batch_s + kv_head   * k_head_s;
    half_type*       O_ptr = O + batch_idx * q_batch_s + head_idx  * q_head_s;

    // ---- LDS pointers ----
    extern __shared__ char smem[];
    half_type* lds_Q = reinterpret_cast<half_type*>(smem);
    half_type* lds_K = reinterpret_cast<half_type*>(smem + LDS_Q_BYTES);
    half_type* lds_V = reinterpret_cast<half_type*>(smem + LDS_Q_BYTES + LDS_K_BYTES);
    // P reuses lds_K area (P is smaller: BLOCK_M*BLOCK_N vs BLOCK_N*HEAD_DIM)
    half_type* lds_P = lds_K;

    // DEBUG: verify kernel runs at all
    if (tid == 0 && m_block == 0 && bh == 0) {
        O_ptr[0] = static_cast<half_type>(42.0f);
    }

    // ---- Load Q tile to LDS (persistent across all KV blocks) ----
    load_tile_global_to_lds<half_type, BLOCK_M, HEAD_DIM>(
        Q_ptr + m_start * q_seq_s, q_seq_s,
        lds_Q, HEAD_DIM,
        valid_m, tid, NUM_THREADS);
    __syncthreads();

    // Cast to half_t* for MFMA loaders (works for both fp16 and bf16
    // because the MFMA intrinsic takes half4_t and the bit layout is
    // the same for fp16; for bf16 we'd need a different MFMA instruction,
    // but hipcc handles the cast under __HIP_NO_HALF_CONVERSIONS__).
    const half_t* q_lds = reinterpret_cast<const half_t*>(lds_Q);
    const half_t* k_lds = reinterpret_cast<const half_t*>(lds_K);
    const half_t* v_lds = reinterpret_cast<const half_t*>(lds_V);
    const half_t* p_lds = reinterpret_cast<const half_t*>(lds_P);

    // ---- Each wave owns 1 M-tile (16 rows) ----
    const int my_m_tile = warp_id;                  // 0..3
    const int my_m_base = my_m_tile * MFMA_M;       // 0, 16, 32, 48

    // O accumulators: D_TILES float4 = 8 * 4 = 32 floats
    float4_t acc_O[D_TILES];
    #pragma unroll
    for (int d = 0; d < D_TILES; d++) acc_O[d] = {0, 0, 0, 0};

    // Online softmax state for this wave's 16 rows (4 rows tracked per lane)
    TileSoftmaxState softmax;
    softmax.init();

    // ---- KV-block loop ----
    int n_blocks = (seqlen_k + BLOCK_N - 1) / BLOCK_N;
    if constexpr (IS_CAUSAL) {
        n_blocks = min(n_blocks, (m_start + BLOCK_M + BLOCK_N - 1) / BLOCK_N);
    }

    for (int nb = 0; nb < n_blocks; nb++) {
        const int n_start = nb * BLOCK_N;
        const int valid_n = min(BLOCK_N, seqlen_k - n_start);

        // ---- Load K tile [BLOCK_N, HEAD_DIM] to LDS ----
        load_tile_global_to_lds<half_type, BLOCK_N, HEAD_DIM>(
            K_ptr + n_start * k_seq_s, k_seq_s,
            reinterpret_cast<half_type*>(smem + LDS_Q_BYTES), HEAD_DIM,
            valid_n, tid, NUM_THREADS);
        __syncthreads();

        // ============================================================
        // S = Q @ K^T   via MFMA    [my 16 rows] x [BLOCK_N cols]
        //   S_acc[nt][j] = S[ my_m_base + (lane/16)*4+j,  nt*16 + lane%16 ]
        // ============================================================
        float S_acc[N_TILES][4];
        #pragma unroll
        for (int nt = 0; nt < N_TILES; nt++) {
            float4_t c = {0, 0, 0, 0};
            #pragma unroll
            for (int ks = 0; ks < K_STEPS; ks++) {
                half4_t a = load_mfma_a(q_lds, HEAD_DIM, my_m_base, ks * MFMA_K, lane_id);
                half4_t b = load_mfma_b(k_lds, HEAD_DIM, nt * MFMA_N, ks * MFMA_K, lane_id);
                c = mfma_f32_16x16x16_f16(a, b, c);
            }
            // Apply softmax_scale
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float v = c[j] * softmax_scale;

                // Causal mask
                if constexpr (IS_CAUSAL) {
                    int global_row = m_start + my_m_base + mfma_acc_row(lane_id, j);
                    int global_col = n_start + nt * MFMA_N + mfma_acc_col(lane_id);
                    if (global_col > global_row) v = -INFINITY;
                }

                // Out-of-bounds mask
                int global_col = n_start + nt * MFMA_N + mfma_acc_col(lane_id);
                if (global_col >= seqlen_k) v = -INFINITY;

                int global_row = m_start + my_m_base + mfma_acc_row(lane_id, j);
                if (global_row >= seqlen_q) v = -INFINITY;

                S_acc[nt][j] = v;
            }
        }

        // ============================================================
        // Online softmax:  row_max, exp, row_sum, rescale acc_O
        // ============================================================
        float correction[4];
        softmax.update<N_TILES>(S_acc, correction);

        // Rescale running O accumulators
        #pragma unroll
        for (int d = 0; d < D_TILES; d++) {
            #pragma unroll
            for (int j = 0; j < 4; j++)
                acc_O[d][j] *= correction[j];
        }

        // ============================================================
        // Write P (= exponentiated S) to LDS for P@V matmul
        //   P layout in LDS: [BLOCK_M, BLOCK_N] row-major fp16
        //   Each lane writes its 4 rows x N_TILES cols
        // ============================================================
        #pragma unroll
        for (int nt = 0; nt < N_TILES; nt++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int row = my_m_base + mfma_acc_row(lane_id, j);
                int col = nt * MFMA_N + mfma_acc_col(lane_id);
                float pval = S_acc[nt][j];
                lds_P[row * BLOCK_N + col] = static_cast<half_type>(pval);
            }
        }
        __syncthreads();    // P fully written

        // ---- Load V tile [BLOCK_N, HEAD_DIM] to LDS ----
        load_tile_global_to_lds<half_type, BLOCK_N, HEAD_DIM>(
            V_ptr + n_start * k_seq_s, k_seq_s,
            reinterpret_cast<half_type*>(smem + LDS_Q_BYTES + LDS_K_BYTES), HEAD_DIM,
            valid_n, tid, NUM_THREADS);
        __syncthreads();

        // ============================================================
        // acc_O += P @ V   via MFMA
        //   P is [BLOCK_M, BLOCK_N],  V is [BLOCK_N, HEAD_DIM]
        //   MFMA computes C += A * B^T,  so set A=P, B^T=V => B=V^T
        //   A operand from P:  a[j] = P[ my_m_base+lane%16, kn*16+(lane/16)*4+j ]
        //   B operand from V^T: scattered reads from V in LDS
        // ============================================================
        #pragma unroll
        for (int dt = 0; dt < D_TILES; dt++) {
            float4_t c = acc_O[dt];
            #pragma unroll
            for (int kn = 0; kn < KN_STEPS; kn++) {
                half4_t a = load_mfma_a(p_lds, BLOCK_N, my_m_base, kn * MFMA_K, lane_id);
                half4_t b = load_mfma_b_Vtrans(v_lds, HEAD_DIM,
                                                kn * MFMA_N, dt * MFMA_N, lane_id);
                c = mfma_f32_16x16x16_f16(a, b, c);
            }
            acc_O[dt] = c;
        }
        __syncthreads();    // safe to overwrite LDS in next iteration
    }

    // ============================================================
    // Normalize and store O
    //   O[row, d] = acc_O[d_tile][j] / row_sum[j]
    //   Each lane writes its 4 rows x D_TILES*16 cols -> 4 x 128 elements
    //   Actually each lane writes 4 rows at 1 column per D-tile
    // ============================================================
    #pragma unroll
    for (int dt = 0; dt < D_TILES; dt++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int row = m_start + my_m_base + mfma_acc_row(lane_id, j);
            int col = dt * MFMA_N + mfma_acc_col(lane_id);
            if (row < seqlen_q && col < head_dim) {
                float val = acc_O[dt][j] * softmax.inv_sum(j);
                O_ptr[row * q_seq_s + col] = static_cast<half_type>(val);
            }
        }
    }

    // Optional LSE output
    if (LSE != nullptr && mfma_acc_col(lane_id) == 0) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int row = m_start + my_m_base + mfma_acc_row(lane_id, j);
            if (row < seqlen_q) {
                float lse = softmax.row_max[j] + logf(softmax.row_sum[j] > 0 ? softmax.row_sum[j] : 1.0f);
                LSE[batch_idx * num_heads_q * seqlen_q + head_idx * seqlen_q + row] = lse;
            }
        }
    }
}

// ======================================================================
// Launch wrapper
// ======================================================================
extern "C" {

void flash_fwd_launch(
    const void* Q, const void* K, const void* V,
    void* O, float* LSE,
    int batch_size, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_k, int head_dim,
    float softmax_scale, bool is_causal, bool is_bf16,
    hipStream_t stream)
{
    dim3 grid((seqlen_q + BLOCK_M - 1) / BLOCK_M, batch_size * num_heads_q);
    dim3 block(NUM_THREADS);
    fprintf(stderr, "[FA4] launch: grid=(%d,%d) block=%d smem=%d sq=%d sk=%d hq=%d hk=%d hd=%d causal=%d bf16=%d\n",
            grid.x, grid.y, block.x, LDS_TOTAL, seqlen_q, seqlen_k, num_heads_q, num_heads_k, head_dim, is_causal, is_bf16);

    // We only compile the fp16 path for now (bf16 needs v_mfma_f32_16x16x16bf16_1k).
    // The half_type template handles the pointer casting; the MFMA operates on fp16.
    if (is_bf16) {
        if (is_causal)
            hipLaunchKernelGGL((flash_fwd_kernel<bf16_t, true>),
                grid, block, LDS_TOTAL, stream,
                (const bf16_t*)Q, (const bf16_t*)K, (const bf16_t*)V,
                (bf16_t*)O, LSE,
                batch_size, seqlen_q, seqlen_k,
                num_heads_q, num_heads_k, head_dim, softmax_scale);
        else
            hipLaunchKernelGGL((flash_fwd_kernel<bf16_t, false>),
                grid, block, LDS_TOTAL, stream,
                (const bf16_t*)Q, (const bf16_t*)K, (const bf16_t*)V,
                (bf16_t*)O, LSE,
                batch_size, seqlen_q, seqlen_k,
                num_heads_q, num_heads_k, head_dim, softmax_scale);
    } else {
        if (is_causal)
            hipLaunchKernelGGL((flash_fwd_kernel<half_t, true>),
                grid, block, LDS_TOTAL, stream,
                (const half_t*)Q, (const half_t*)K, (const half_t*)V,
                (half_t*)O, LSE,
                batch_size, seqlen_q, seqlen_k,
                num_heads_q, num_heads_k, head_dim, softmax_scale);
        else
            hipLaunchKernelGGL((flash_fwd_kernel<half_t, false>),
                grid, block, LDS_TOTAL, stream,
                (const half_t*)Q, (const half_t*)K, (const half_t*)V,
                (half_t*)O, LSE,
                batch_size, seqlen_q, seqlen_k,
                num_heads_q, num_heads_k, head_dim, softmax_scale);
    }
}

} // extern "C"
