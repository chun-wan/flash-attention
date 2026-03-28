// Combined: kernel + PyTorch binding in single translation unit
// This avoids cross-TU fat binary registration issues with -fno-gpu-rdc

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
#define M_TILES      (BLOCK_M / MFMA_M)
#define N_TILES      (BLOCK_N / MFMA_N)
#define K_STEPS      (HEAD_DIM / MFMA_K)
#define D_TILES      (HEAD_DIM / MFMA_N)
#define KN_STEPS     (BLOCK_N  / MFMA_K)
#define VT_PAD       8
#define LDS_Q_BYTES  (BLOCK_M * HEAD_DIM * 2)
#define LDS_K_BYTES  (BLOCK_N * HEAD_DIM * 2)
#define LDS_VT_BYTES (HEAD_DIM * (BLOCK_N + VT_PAD) * 2)
#define LDS_TOTAL    (LDS_Q_BYTES + LDS_K_BYTES + LDS_VT_BYTES)
static_assert(LDS_TOTAL <= 65536, "LDS overflow");

template <typename half_type, bool IS_CAUSAL>
__global__ __launch_bounds__(NUM_THREADS)
void flash_fwd_kernel(
    const half_type* __restrict__ Q, const half_type* __restrict__ K,
    const half_type* __restrict__ V, half_type* __restrict__ O,
    float* __restrict__ LSE,
    int batch_size, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_k, int head_dim, float softmax_scale)
{
    const int m_block = blockIdx.x, bh = blockIdx.y;
    const int batch_idx = bh / num_heads_q, head_idx = bh % num_heads_q;
    const int kv_head = head_idx / (num_heads_q / num_heads_k);
    const int tid = threadIdx.x, lane_id = tid % WARP_SIZE, warp_id = tid / WARP_SIZE;
    const int m_start = m_block * BLOCK_M;

    const int q_batch_s = seqlen_q * num_heads_q * head_dim;
    const int q_seq_s   = num_heads_q * head_dim;
    const int k_batch_s = seqlen_k * num_heads_k * head_dim;
    const int k_seq_s   = num_heads_k * head_dim;

    const half_type* Q_ptr = Q + batch_idx * q_batch_s + head_idx * head_dim;
    const half_type* K_ptr = K + batch_idx * k_batch_s + kv_head * head_dim;
    const half_type* V_ptr = V + batch_idx * k_batch_s + kv_head * head_dim;
    half_type*       O_ptr = O + batch_idx * q_batch_s + head_idx * head_dim;

    extern __shared__ char smem[];
    auto* lds_Q = reinterpret_cast<half_type*>(smem);
    auto* lds_K = reinterpret_cast<half_type*>(smem + LDS_Q_BYTES);
    auto* lds_VT = reinterpret_cast<half_type*>(smem + LDS_Q_BYTES + LDS_K_BYTES);
    auto* lds_P = lds_K;

    const int valid_m = min(BLOCK_M, seqlen_q - m_start);

    // Load Q
    load_tile_global_to_lds<half_type, BLOCK_M, HEAD_DIM>(
        Q_ptr + m_start * q_seq_s, q_seq_s, lds_Q, HEAD_DIM, valid_m, tid, NUM_THREADS);
    __syncthreads();

    const auto* q_lds = reinterpret_cast<const half_t*>(lds_Q);
    const auto* k_lds = reinterpret_cast<const half_t*>(lds_K);
    const auto* vt_lds = reinterpret_cast<const half_t*>(lds_VT);
    const auto* p_lds = reinterpret_cast<const half_t*>(lds_P);

    const int my_m = warp_id * MFMA_M;

    float4_t acc_O[D_TILES];
    for (int d = 0; d < D_TILES; d++) acc_O[d] = {0,0,0,0};
    TileSoftmaxState softmax;
    softmax.init();

    int n_blocks = (seqlen_k + BLOCK_N - 1) / BLOCK_N;
    if (IS_CAUSAL) n_blocks = min(n_blocks, (m_start + BLOCK_M + BLOCK_N - 1) / BLOCK_N);

    for (int nb = 0; nb < n_blocks; nb++) {
        const int n_start = nb * BLOCK_N;
        const int valid_n = min(BLOCK_N, seqlen_k - n_start);

        load_tile_global_to_lds<half_type, BLOCK_N, HEAD_DIM>(
            K_ptr + n_start * k_seq_s, k_seq_s,
            reinterpret_cast<half_type*>(smem + LDS_Q_BYTES), HEAD_DIM,
            valid_n, tid, NUM_THREADS);
        __syncthreads();

        // S = Q @ K^T
        float S_acc[N_TILES][4];
        for (int nt = 0; nt < N_TILES; nt++) {
            float4_t c = {0,0,0,0};
            for (int ks = 0; ks < K_STEPS; ks++) {
                half4_t a = load_mfma_a(q_lds, HEAD_DIM, my_m, ks*MFMA_K, lane_id);
                half4_t b = load_mfma_b(k_lds, HEAD_DIM, nt*MFMA_N, ks*MFMA_K, lane_id);
                c = mfma_f32_16x16x16_f16(a, b, c);
            }
            for (int j = 0; j < 4; j++) {
                float v = c[j] * softmax_scale;
                int gr = m_start + my_m + mfma_acc_row(lane_id, j);
                int gc = n_start + nt * MFMA_N + mfma_acc_col(lane_id);
                if (IS_CAUSAL && gc > gr) v = -INFINITY;
                if (gc >= seqlen_k || gr >= seqlen_q) v = -INFINITY;
                S_acc[nt][j] = v;
            }
        }

        // Softmax
        float correction[4];
        softmax.update<N_TILES>(S_acc, correction);
        for (int d = 0; d < D_TILES; d++)
            for (int j = 0; j < 4; j++) acc_O[d][j] *= correction[j];

        // Write P to LDS
        for (int nt = 0; nt < N_TILES; nt++)
            for (int j = 0; j < 4; j++) {
                int r = my_m + mfma_acc_row(lane_id, j);
                int c = nt * MFMA_N + mfma_acc_col(lane_id);
                lds_P[r * BLOCK_N + c] = static_cast<half_type>(S_acc[nt][j]);
            }
        __syncthreads();

        // Load V transposed: V[BLOCK_N, HEAD_DIM] -> VT[HEAD_DIM, BLOCK_N+PAD]
        load_tile_transposed<half_type, BLOCK_N, HEAD_DIM, VT_PAD>(
            V_ptr + n_start * k_seq_s, k_seq_s,
            reinterpret_cast<half_type*>(smem + LDS_Q_BYTES + LDS_K_BYTES),
            BLOCK_N + VT_PAD,
            valid_n, tid, NUM_THREADS);
        __syncthreads();

        // O += P @ V  (using VT[HEAD_DIM, BLOCK_N+PAD] for contiguous B loads)
        for (int dt = 0; dt < D_TILES; dt++) {
            float4_t c = acc_O[dt];
            for (int kn = 0; kn < KN_STEPS; kn++) {
                half4_t a = load_mfma_a(p_lds, BLOCK_N, my_m, kn*MFMA_K, lane_id);
                half4_t b = load_mfma_b_from_VT(vt_lds, BLOCK_N + VT_PAD,
                                                 kn*MFMA_N, dt*MFMA_N, lane_id);
                c = mfma_f32_16x16x16_f16(a, b, c);
            }
            acc_O[dt] = c;
        }
        __syncthreads();
    }

    // Write O
    for (int dt = 0; dt < D_TILES; dt++)
        for (int j = 0; j < 4; j++) {
            int row = m_start + my_m + mfma_acc_row(lane_id, j);
            int col = dt * MFMA_N + mfma_acc_col(lane_id);
            if (row < seqlen_q && col < head_dim) {
                float val = acc_O[dt][j] * softmax.inv_sum(j);
                O_ptr[row * q_seq_s + col] = static_cast<half_type>(val);
            }
        }
}

// ============================================================
// PyTorch binding (same TU as kernel)
// ============================================================
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <vector>

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().is_cuda(), #x " must be on GPU")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

std::vector<torch::Tensor> flash_attn_fwd(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    float softmax_scale, bool is_causal, bool return_lse)
{
    CHECK_DEVICE(Q); CHECK_CONTIGUOUS(Q);
    CHECK_DEVICE(K); CHECK_CONTIGUOUS(K);
    CHECK_DEVICE(V); CHECK_CONTIGUOUS(V);
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4);

    const int bs = Q.size(0), sq = Q.size(1), hq = Q.size(2), hd = Q.size(3);
    const int sk = K.size(1), hk = K.size(2);
    TORCH_CHECK(hq % hk == 0);
    TORCH_CHECK(Q.dtype() == torch::kFloat16 || Q.dtype() == torch::kBFloat16);
    bool is_bf16 = (Q.dtype() == torch::kBFloat16);

    auto O = torch::full_like(Q, -999.0);
    float* lse_ptr = nullptr;
    torch::Tensor LSE;
    if (return_lse) {
        LSE = torch::empty({bs, hq, sq}, torch::dtype(torch::kFloat32).device(Q.device()));
        lse_ptr = LSE.data_ptr<float>();
    }

    dim3 grid((sq + BLOCK_M - 1) / BLOCK_M, bs * hq);
    dim3 block(NUM_THREADS);
    hipStream_t stream = at::hip::getCurrentHIPStream().stream();

    #define LAUNCH(HT, CAUSAL) \
        hipLaunchKernelGGL((flash_fwd_kernel<HT, CAUSAL>), grid, block, LDS_TOTAL, stream, \
            reinterpret_cast<const HT*>(Q.data_ptr()), \
            reinterpret_cast<const HT*>(K.data_ptr()), \
            reinterpret_cast<const HT*>(V.data_ptr()), \
            reinterpret_cast<HT*>(O.data_ptr()), \
            lse_ptr, bs, sq, sk, hq, hk, hd, softmax_scale)

    if (is_bf16) {
        if (is_causal) LAUNCH(bf16_t, true); else LAUNCH(bf16_t, false);
    } else {
        if (is_causal) LAUNCH(half_t, true); else LAUNCH(half_t, false);
    }
    #undef LAUNCH

    if (return_lse) return {O, LSE};
    return {O};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_fwd", &flash_attn_fwd, "Flash Attention Forward (FA4 ROCm)");
}
