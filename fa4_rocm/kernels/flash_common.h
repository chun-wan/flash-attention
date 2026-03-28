#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

constexpr int WARP_SIZE = 64;
constexpr int MFMA_M = 16;
constexpr int MFMA_N = 16;
constexpr int MFMA_K = 16;

typedef _Float16 half_t;
typedef __hip_bfloat16 bf16_t;

using float4_t = float __attribute__((ext_vector_type(4)));
using half4_t  = _Float16 __attribute__((ext_vector_type(4)));

constexpr float LOG2E_F = 1.4426950408889634f;

__device__ __forceinline__ float fast_exp(float x) {
    return exp2f(x * LOG2E_F);
}

// ---------- MFMA intrinsic ----------
// v_mfma_f32_16x16x16f16 computes C[16x16] += A[16x16] * B^T[16x16]
// Lane mapping (verified on gfx942):
//   Output acc[j]:  row = (lane/16)*4 + j,  col = lane % 16
//   Input A half4:  A[ lane%16, (lane/16)*4 + j ]      (M, K)
//   Input B half4:  B[ lane%16, (lane/16)*4 + j ]      (N, K)
//   The instruction computes C += A * B^T
__device__ __forceinline__ float4_t mfma_f32_16x16x16_f16(half4_t a, half4_t b, float4_t c) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
}

// ---------- MFMA lane mapping helpers ----------
__device__ __forceinline__ int mfma_acc_row(int lane_id, int j) {
    return (lane_id / 16) * 4 + j;
}

__device__ __forceinline__ int mfma_acc_col(int lane_id) {
    return lane_id % 16;
}

// ---------- MFMA operand loaders from LDS ----------
// Load A operand: a[j] = mat[m_base + lane%16, k_base + (lane/16)*4 + j]
// mat is row-major with leading dimension `ld`.
__device__ __forceinline__ half4_t load_mfma_a(
    const half_t* __restrict__ lds, int ld, int m_base, int k_base, int lane
) {
    int row = m_base + (lane % 16);
    int col = k_base + (lane / 16) * 4;
    return *reinterpret_cast<const half4_t*>(&lds[row * ld + col]);
}

// Load B operand (same layout as A). MFMA computes A*B^T,
// so placing K[n, d] in B gives Q@K^T naturally.
__device__ __forceinline__ half4_t load_mfma_b(
    const half_t* __restrict__ lds, int ld, int n_base, int k_base, int lane
) {
    int row = n_base + (lane % 16);
    int col = k_base + (lane / 16) * 4;
    return *reinterpret_cast<const half4_t*>(&lds[row * ld + col]);
}

// Load B operand from V for P@V.  We need C += P * V = A * B^T where B = V^T.
// b[j] = V^T[d_base + lane%16, n_base + (lane/16)*4 + j]
//       = V[ n_base + (lane/16)*4 + j,  d_base + lane%16 ]
// Scattered reads with stride HEAD_DIM between consecutive j.
__device__ __forceinline__ half4_t load_mfma_b_Vtrans(
    const half_t* __restrict__ lds_v, int ld_v, int n_base, int d_base, int lane
) {
    half4_t r;
    int d = d_base + (lane % 16);
    int n0 = n_base + (lane / 16) * 4;
    r[0] = lds_v[(n0 + 0) * ld_v + d];
    r[1] = lds_v[(n0 + 1) * ld_v + d];
    r[2] = lds_v[(n0 + 2) * ld_v + d];
    r[3] = lds_v[(n0 + 3) * ld_v + d];
    return r;
}

// Load B from V^T stored in LDS with padding for bank-conflict avoidance.
// VT is stored as [HEAD_DIM, BLOCK_N+PAD], so contiguous half4 loads work.
// b[j] = VT[ d_base + lane%16, n_base + (lane/16)*4 + j ]
__device__ __forceinline__ half4_t load_mfma_b_from_VT(
    const half_t* __restrict__ lds_vt, int ld_vt, int n_base, int d_base, int lane
) {
    int row = d_base + (lane % 16);
    int col = n_base + (lane / 16) * 4;
    return *reinterpret_cast<const half4_t*>(&lds_vt[row * ld_vt + col]);
}

// BF16 MFMA intrinsic for gfx942 (CDNA3)
// __hip_bfloat16 can't be used with ext_vector_type; use short4 as the packed type
using bf16x4_raw_t = short __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float4_t mfma_f32_16x16x16_bf16(bf16x4_raw_t a, bf16x4_raw_t b, float4_t c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

// Vectorized LDS store: write 8 fp16 (128 bits) at once
__device__ __forceinline__ void store_lds_128(half_t* dst, const half_t* src) {
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
}

// Vectorized global load: read 8 fp16 (128 bits) at once
template <typename T>
__device__ __forceinline__ void load_global_128(T* dst, const T* src) {
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
}

// Vectorized tile load: global -> LDS with 128-bit vectorized loads
template <typename T, int ROWS, int COLS>
__device__ void load_tile_vectorized(
    const T* __restrict__ src, int src_stride,
    T* __restrict__ dst, int dst_stride,
    int valid_rows, int tid, int nthreads
) {
    constexpr int ELEMS_PER_VEC = 8;
    constexpr int TOTAL_VECS = (ROWS * COLS) / ELEMS_PER_VEC;
    constexpr int VECS_PER_THREAD = (TOTAL_VECS + 255) / 256;

    #pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; i++) {
        int vec_idx = tid + i * nthreads;
        if (vec_idx < TOTAL_VECS) {
            int elem_idx = vec_idx * ELEMS_PER_VEC;
            int r = elem_idx / COLS;
            int c = elem_idx % COLS;
            if (r < valid_rows && c + ELEMS_PER_VEC <= COLS) {
                load_global_128(&dst[r * dst_stride + c], &src[r * src_stride + c]);
            } else if (r < valid_rows) {
                for (int j = 0; j < ELEMS_PER_VEC && c + j < COLS; j++)
                    dst[r * dst_stride + c + j] = src[r * src_stride + c + j];
            } else {
                for (int j = 0; j < ELEMS_PER_VEC && c + j < COLS; j++)
                    dst[r * dst_stride + c + j] = T(0);
            }
        }
    }
}

// Transposed tile load: global V[ROWS, COLS] -> LDS VT[COLS, ROWS+PAD]
template <typename T, int ROWS, int COLS, int PAD>
__device__ void load_tile_transposed(
    const T* __restrict__ src, int src_stride,
    T* __restrict__ dst_vt, int dst_stride,   // dst_stride = ROWS + PAD
    int valid_rows, int tid, int nthreads
) {
    constexpr int TOTAL = ROWS * COLS;
    constexpr int PER_THREAD = (TOTAL + 255) / 256;
    #pragma unroll
    for (int i = 0; i < PER_THREAD; i++) {
        int idx = tid + i * nthreads;
        if (idx < TOTAL) {
            int r = idx / COLS;  // V row (n dimension)
            int c = idx % COLS;  // V col (d dimension)
            T val = (r < valid_rows) ? src[r * src_stride + c] : T(0);
            dst_vt[c * dst_stride + r] = val;  // VT[d, n] with stride (ROWS+PAD)
        }
    }
}

// ---------- Intra-group reduction ----------
// Reduce across the 16 lanes within each group-of-16 (lanes sharing the
// same output rows).  Uses __shfl_xor with offsets 8, 4, 2, 1.
__device__ __forceinline__ float group16_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor(val, 8, WARP_SIZE));
    val = fmaxf(val, __shfl_xor(val, 4, WARP_SIZE));
    val = fmaxf(val, __shfl_xor(val, 2, WARP_SIZE));
    val = fmaxf(val, __shfl_xor(val, 1, WARP_SIZE));
    return val;
}

__device__ __forceinline__ float group16_reduce_sum(float val) {
    val += __shfl_xor(val, 8, WARP_SIZE);
    val += __shfl_xor(val, 4, WARP_SIZE);
    val += __shfl_xor(val, 2, WARP_SIZE);
    val += __shfl_xor(val, 1, WARP_SIZE);
    return val;
}

// ---------- Cooperative tile load: global -> LDS ----------
template <typename T, int ROWS, int COLS>
__device__ void load_tile_global_to_lds(
    const T* __restrict__ src, int src_stride,
    T* __restrict__ dst,       int dst_stride,
    int valid_rows, int tid, int nthreads
) {
    constexpr int TOTAL = ROWS * COLS;
    constexpr int PER_THREAD = (TOTAL + 255) / 256;  // assumes nthreads <= 256
    #pragma unroll
    for (int i = 0; i < PER_THREAD; i++) {
        int idx = tid + i * nthreads;
        if (idx < TOTAL) {
            int r = idx / COLS;
            int c = idx % COLS;
            dst[r * dst_stride + c] = (r < valid_rows) ? src[r * src_stride + c] : T(0);
        }
    }
}
