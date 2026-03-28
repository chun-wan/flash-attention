// CK FMHA torch extension wrapper
// Wraps CK's fmha_fwd() dispatch into a torch custom op.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha.hpp"

#include "fmha_fwd.hpp"

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

namespace {

std::string dtype_to_string(at::ScalarType dtype) {
    switch (dtype) {
        case at::kHalf: return "fp16";
        case at::kBFloat16: return "bf16";
        default: TORCH_CHECK(false, "Unsupported dtype");
    }
}

std::tuple<at::Tensor, at::Tensor>
ck_flash_attn_fwd(
    at::Tensor q,    // (batch, seqlen_q, nheads_q, hdim)
    at::Tensor k,    // (batch, seqlen_k, nheads_k, hdim)
    at::Tensor v,    // (batch, seqlen_k, nheads_k, hdim)
    float softmax_scale,
    bool is_causal,
    bool return_lse)
{
    TORCH_CHECK(q.is_cuda(), "q must be on GPU");
    TORCH_CHECK(q.dim() == 4, "q must be 4D");
    TORCH_CHECK(k.dim() == 4, "k must be 4D");
    TORCH_CHECK(v.dim() == 4, "v must be 4D");

    const int batch = q.size(0);
    const int seqlen_q = q.size(1);
    const int nheads_q = q.size(2);
    const int hdim_q = q.size(3);
    const int seqlen_k = k.size(1);
    const int nheads_k = k.size(2);
    const int hdim_v = v.size(3);

    TORCH_CHECK(nheads_q % nheads_k == 0,
                "nheads_q must be divisible by nheads_k for GQA");

    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();

    auto o = torch::empty_like(q);
    at::Tensor lse;
    if (return_lse) {
        lse = torch::empty({batch, nheads_q, seqlen_q},
                           q.options().dtype(at::kFloat));
    } else {
        lse = torch::empty({0}, q.options().dtype(at::kFloat));
    }

    std::string dtype_str = dtype_to_string(q.scalar_type());
    int mask_type = is_causal ? 2 : 0;
    bool is_group_mode = false;

    // Strides (in elements)
    int stride_q = hdim_q;
    int stride_k = hdim_q;
    int stride_v = hdim_v;
    int stride_o = hdim_q;
    int nhead_stride_q = seqlen_q * hdim_q;
    int nhead_stride_k = seqlen_k * hdim_q;
    int nhead_stride_v = seqlen_k * hdim_v;
    int nhead_stride_o = seqlen_q * hdim_q;
    int batch_stride_q = nheads_q * nhead_stride_q;
    int batch_stride_k = nheads_k * nhead_stride_k;
    int batch_stride_v = nheads_k * nhead_stride_v;
    int batch_stride_o = nheads_q * nhead_stride_o;

    // Build CK traits and args
    fmha_fwd_traits traits{
        hdim_q,
        hdim_v,
        dtype_str,
        is_group_mode,
        true,                              // is_v_rowmajor
        false,                             // logits_soft_cap > 0
        static_cast<mask_enum>(mask_type),
        bias_enum::no_bias,
        return_lse,                        // has_lse
        false,                             // has_dropout
        quant_scale_enum::none,            // qscale
        false,                             // min_seqlen_q != 0
        false                              // has_sink
    };

    fmha_fwd_args args{
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        nullptr,  // bias
        nullptr,  // q_descale
        nullptr,  // k_descale
        nullptr,  // v_descale
        nullptr,  // rand_val
        return_lse ? lse.data_ptr() : nullptr,
        o.data_ptr(),
        nullptr,  // seqstart_q
        nullptr,  // seqstart_k
        nullptr,  // seqlen_q_ptr
        nullptr,  // seqlen_k_ptr
        nullptr,  // cu_seqlen_q
        nullptr,  // cu_seqlen_k
        nullptr,  // block_scale_seqstart_q
        nullptr,  // block_scale_seqstart_k
        nullptr,  // sink
        seqlen_q,
        seqlen_k,
        batch,
        seqlen_q,  // max_seqlen_q
        hdim_q,
        hdim_v,
        nheads_q,
        nheads_k,
        softmax_scale,
        0.0f,  // logits_soft_cap
        stride_q,
        stride_k,
        stride_v,
        0,  // stride_bias
        0,  // stride_randval
        stride_o,
        nhead_stride_q,
        nhead_stride_k,
        nhead_stride_v,
        0,  // nhead_stride_bias
        0,  // nhead_stride_randval
        return_lse ? seqlen_q : 0,  // nhead_stride_lse
        nhead_stride_o,
        0,  // nhead_stride_q_descale
        0,  // nhead_stride_k_descale
        0,  // nhead_stride_v_descale
        batch_stride_q,
        batch_stride_k,
        batch_stride_v,
        0,  // batch_stride_bias
        0,  // batch_stride_randval
        return_lse ? nheads_q * seqlen_q : 0,  // batch_stride_lse
        batch_stride_o,
        0,  // batch_stride_q_descale
        0,  // batch_stride_k_descale
        0,  // batch_stride_v_descale
        -1, // window_size_left
        is_causal ? 0 : -1,  // window_size_right
        0,  // sink_size
        mask_type,
        0,  // min_seqlen_q
        0.0f,  // p_drop
        {0, 0},  // s_randval
        {0, 0},  // drop_seed_offset
        0,  // block_scale_size_q
        0   // block_scale_size_kv
    };

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    ck_tile::stream_config sc{stream};

    float ret = fmha_fwd(traits, args, sc);
    TORCH_CHECK(ret >= 0, "CK FMHA dispatch failed (unsupported config: dtype=",
                dtype_str, " hdim=", hdim_q, " mask=", mask_type, ")");

    return std::make_tuple(o, lse);
}

} // namespace

TORCH_LIBRARY_FRAGMENT(fa4_ck, m) {
    m.def("flash_attn_fwd(Tensor q, Tensor k, Tensor v, float scale, bool causal, bool return_lse) -> (Tensor, Tensor)");
    m.impl("flash_attn_fwd", ck_flash_attn_fwd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_fwd", &ck_flash_attn_fwd,
          "CK FMHA forward (batch mode)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("softmax_scale"), py::arg("is_causal"),
          py::arg("return_lse") = false);
}
