import torch, math, sys
sys.path.insert(0, "/workspace/fa4_rocm")
from triton_kernels.flash_fwd_triton_v2 import flash_attn_triton_v2_func as fa4v2
device = "cuda:0"
torch.manual_seed(42)
q = torch.randn(2, 2048, 32, 128, dtype=torch.bfloat16, device=device)
k = torch.randn(2, 2048, 32, 128, dtype=torch.bfloat16, device=device)
v = torch.randn(2, 2048, 32, 128, dtype=torch.bfloat16, device=device)
scale = 1.0 / math.sqrt(128)
for _ in range(3):
    fa4v2(q, k, v, causal=True, softmax_scale=scale)
torch.cuda.synchronize()
for _ in range(3):
    fa4v2(q, k, v, causal=True, softmax_scale=scale)
torch.cuda.synchronize()
