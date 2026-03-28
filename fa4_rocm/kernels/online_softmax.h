#pragma once
#include "flash_common.h"

// Online softmax state for one MFMA 16x16 output tile.
// Each lane owns 4 rows (acc[0..3]), with the column = lane%16.
// After QK^T MFMA, each lane holds QK_N_TILES float4 accumulators.
//
// Row-max and row-sum are tracked *per row* and reduced across the
// 16 lanes of each group-of-16 that share those rows.
//
// Storage: 4 row_max + 4 row_sum = 8 floats per lane.

struct TileSoftmaxState {
    float row_max[4];
    float row_sum[4];

    __device__ __forceinline__ void init() {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            row_max[j] = -INFINITY;
            row_sum[j] = 0.0f;
        }
    }

    // Given new QK block scores in S_acc[n_tiles][4], perform online softmax
    // update.  Returns the correction factor per row (4 values) so the caller
    // can rescale the O accumulator.
    //
    // After this call:
    //   - S_acc values are replaced with exp(S - new_max)  (= P)
    //   - row_max / row_sum are updated
    //   - correction[j] = exp(old_max[j] - new_max[j])
    template <int N_TILES>
    __device__ __forceinline__ void update(
        float S_acc[N_TILES][4], float correction[4]
    ) {
        // Step 1: per-lane local max for each of the 4 rows
        float local_max[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float m = S_acc[0][j];
            #pragma unroll
            for (int nt = 1; nt < N_TILES; nt++)
                m = fmaxf(m, S_acc[nt][j]);
            local_max[j] = m;
        }

        // Step 2: reduce max across the 16 lanes sharing these rows
        #pragma unroll
        for (int j = 0; j < 4; j++)
            local_max[j] = group16_reduce_max(local_max[j]);

        // Step 3: merge with running row_max
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float new_max = fmaxf(row_max[j], local_max[j]);
            float c = (row_max[j] == -INFINITY) ? 0.0f : fast_exp(row_max[j] - new_max);
            correction[j] = c;
            row_sum[j] *= c;
            row_max[j] = new_max;
        }

        // Step 4: exponentiate S and accumulate row_sum
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float block_sum = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < N_TILES; nt++) {
                float v = S_acc[nt][j];
                v = (v == -INFINITY) ? 0.0f : fast_exp(v - row_max[j]);
                S_acc[nt][j] = v;   // now P
                block_sum += v;
            }
            // reduce partial sums across 16 lanes of the group
            block_sum = group16_reduce_sum(block_sum);
            row_sum[j] += block_sum;
        }
    }

    __device__ __forceinline__ float inv_sum(int j) const {
        return (row_sum[j] > 0.0f) ? (1.0f / row_sum[j]) : 0.0f;
    }
};
