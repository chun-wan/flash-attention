// PyTorch C++ extension binding for FA4 ROCm kernels
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <vector>

extern "C" void flash_fwd_launch(
    const void* Q, const void* K, const void* V,
    void* O, float* LSE,
    int batch_size, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_k, int head_dim,
    float softmax_scale, bool is_causal, bool is_bf16,
    hipStream_t stream
);

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().is_cuda(), #x " must be on CUDA/HIP device")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_DEVICE(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> flash_attn_fwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float softmax_scale,
    bool is_causal,
    bool return_lse
) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);

    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [batch, seqlen_q, num_heads_q, head_dim]");
    TORCH_CHECK(K.dim() == 4, "K must be 4D [batch, seqlen_k, num_heads_k, head_dim]");
    TORCH_CHECK(V.dim() == 4, "V must be 4D [batch, seqlen_k, num_heads_k, head_dim]");

    const int batch_size = Q.size(0);
    const int seqlen_q = Q.size(1);
    const int num_heads_q = Q.size(2);
    const int head_dim = Q.size(3);
    const int seqlen_k = K.size(1);
    const int num_heads_k = K.size(2);

    TORCH_CHECK(K.size(0) == batch_size, "K batch size must match Q");
    TORCH_CHECK(V.size(0) == batch_size, "V batch size must match Q");
    TORCH_CHECK(V.size(1) == seqlen_k, "V seqlen must match K");
    TORCH_CHECK(V.size(2) == num_heads_k, "V num_heads must match K");
    TORCH_CHECK(K.size(3) == head_dim, "K head_dim must match Q");
    TORCH_CHECK(V.size(3) == head_dim, "V head_dim must match Q");
    TORCH_CHECK(num_heads_q % num_heads_k == 0,
                "num_heads_q must be divisible by num_heads_k for GQA/MQA");
    TORCH_CHECK(Q.dtype() == K.dtype() && Q.dtype() == V.dtype(),
                "Q, K, V must have the same dtype");
    TORCH_CHECK(Q.dtype() == torch::kFloat16 || Q.dtype() == torch::kBFloat16,
                "Only fp16 and bf16 are supported");

    bool is_bf16 = (Q.dtype() == torch::kBFloat16);

    auto O = torch::full_like(Q, -999.0);

    torch::Tensor LSE;
    float* lse_ptr = nullptr;
    if (return_lse) {
        LSE = torch::empty({batch_size, num_heads_q, seqlen_q},
                           torch::dtype(torch::kFloat32).device(Q.device()));
        lse_ptr = LSE.data_ptr<float>();
    }

    hipStream_t stream = at::hip::getCurrentHIPStream().stream();

    flash_fwd_launch(
        Q.data_ptr(),
        K.data_ptr(),
        V.data_ptr(),
        O.data_ptr(),
        lse_ptr,
        batch_size, seqlen_q, seqlen_k,
        num_heads_q, num_heads_k, head_dim,
        softmax_scale, is_causal, is_bf16,
        stream
    );

    if (return_lse) {
        return {O, LSE};
    }
    return {O};
}

// Simple test kernel defined in THIS translation unit
__global__ void trivial_kernel(float* out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) out[tid] = 42.0f + tid;
}

torch::Tensor test_trivial_launch(int n) {
    auto out = torch::zeros({n}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    hipStream_t stream = at::hip::getCurrentHIPStream().stream();
   hipLaunchKernelGGL(( trivial_kernel), dim3((n+255)/256), dim3(256), 0, stream, out.data_ptr<float>(), n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_trivial_launch", &test_trivial_launch, "Trivial kernel test");
    m.def("flash_attn_fwd", &flash_attn_fwd,
          "Flash Attention Forward (FA4 ROCm)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("softmax_scale"), py::arg("is_causal"),
          py::arg("return_lse") = false);
}
