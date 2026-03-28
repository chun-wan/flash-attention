#!/usr/bin/env python3
"""Basic CK FMHA correctness test."""
import torch
import sys
import math

sys.path.insert(0, "/workspace/fa4_rocm")
from ck_kernels.flash_attn_ck import ck_flash_attn_func


def test_config(batch, seqlen, nheads_q, nheads_k, hdim, dtype, causal):
    q = torch.randn(batch, seqlen, nheads_q, hdim, dtype=dtype, device="cuda")
    k = torch.randn(batch, seqlen, nheads_k, hdim, dtype=dtype, device="cuda")
    v = torch.randn(batch, seqlen, nheads_k, hdim, dtype=dtype, device="cuda")
    scale = 1.0 / math.sqrt(hdim)

    out = ck_flash_attn_func(q, k, v, causal=causal, softmax_scale=scale, mode="aiter")

    with torch.no_grad():
        qt = q.transpose(1, 2)
        kt = k.transpose(1, 2)
        vt = v.transpose(1, 2)
        if nheads_q != nheads_k:
            gqa_ratio = nheads_q // nheads_k
            kt = kt.repeat_interleave(gqa_ratio, dim=1)
            vt = vt.repeat_interleave(gqa_ratio, dim=1)
        ref = torch.nn.functional.scaled_dot_product_attention(
            qt, kt, vt, is_causal=causal
        ).transpose(1, 2)

    err = (out - ref).abs().max().item()
    dtype_str = "fp16" if dtype == torch.float16 else "bf16"
    causal_str = "causal" if causal else "noncausal"
    gqa = f"GQA{nheads_q // nheads_k}" if nheads_q != nheads_k else "MHA"
    tol = 0.005 if dtype == torch.float16 else 0.05
    ok = err < tol
    status = "PASS" if ok else "FAIL"
    print(f"{dtype_str} {causal_str} b{batch}s{seqlen}h{nheads_q}d{hdim} {gqa}: err={err:.6f} {status} (tol={tol})")
    return ok


results = []
# bf16
results.append(test_config(2, 2048, 32, 32, 128, torch.bfloat16, True))
results.append(test_config(2, 2048, 32, 32, 128, torch.bfloat16, False))
results.append(test_config(2, 2048, 32, 8, 128, torch.bfloat16, True))
# fp16
results.append(test_config(2, 2048, 32, 32, 128, torch.float16, True))
results.append(test_config(2, 2048, 32, 32, 128, torch.float16, False))
results.append(test_config(2, 2048, 32, 8, 128, torch.float16, True))
# long sequence
results.append(test_config(1, 8192, 32, 32, 128, torch.bfloat16, True))
# performance shape
results.append(test_config(2, 4096, 32, 32, 128, torch.bfloat16, True))

print(f"\nPassed: {sum(results)}/{len(results)}")
