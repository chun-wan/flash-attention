"""
Correctness tests for FA4 ROCm flash attention kernels.

Verifies numerical accuracy against torch.nn.functional.scaled_dot_product_attention
across a comprehensive matrix of dtypes, sequence lengths, head dimensions,
causal/non-causal modes, and GQA/MQA ratios.

Usage:
    pytest test_correctness.py -v
    pytest test_correctness.py -v -k "bf16 and causal"
    python test_correctness.py  # standalone mode
"""

import math
import sys
import os
from pathlib import Path
from itertools import product

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Tolerance thresholds (max absolute error)
ATOL_BF16 = 1e-2
ATOL_FP16 = 5e-3
# Fraction of elements allowed to exceed ATOL
MAX_ERROR_RATIO = 0.01


def reference_attention(q, k, v, causal=False, softmax_scale=None):
    """
    Reference attention using torch SDPA.

    Args:
        q: [batch, seqlen_q, num_heads_q, head_dim]
        k: [batch, seqlen_k, num_heads_k, head_dim]
        v: [batch, seqlen_k, num_heads_k, head_dim]
    """
    batch, seqlen_q, num_heads_q, head_dim = q.shape
    num_heads_k = k.shape[2]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # SDPA expects [batch, heads, seqlen, head_dim]
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()

    # Expand K/V for GQA
    if num_heads_q != num_heads_k:
        repeat_factor = num_heads_q // num_heads_k
        k_t = k_t.repeat_interleave(repeat_factor, dim=1)
        v_t = v_t.repeat_interleave(repeat_factor, dim=1)

    with torch.no_grad():
        ref = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, is_causal=causal, scale=softmax_scale
        )

    return ref.transpose(1, 2).contiguous()


def check_correctness(out, ref, atol, name=""):
    """Check numerical accuracy of output against reference."""
    diff = (out.float() - ref.float()).abs()
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()
    error_ratio = (diff > atol).float().mean().item()

    passed = max_abs_err < atol * 10 and error_ratio < MAX_ERROR_RATIO

    if not passed:
        print(f"\n  FAIL {name}: max_abs_err={max_abs_err:.6f}, "
              f"mean_abs_err={mean_abs_err:.6f}, "
              f"error_ratio={error_ratio:.4f} (threshold: {atol})")
    else:
        print(f"\n  PASS {name}: max_abs_err={max_abs_err:.6f}, "
              f"mean_abs_err={mean_abs_err:.6f}")

    return passed, max_abs_err, mean_abs_err, error_ratio


# Test parameter matrix
DTYPES = [
    (torch.bfloat16, ATOL_BF16, "bf16"),
    (torch.float16, ATOL_FP16, "fp16"),
]

HEAD_DIMS = [64, 96, 128]

SEQ_LENS = [
    (128, 128),
    (512, 512),
    (2048, 2048),
]

LONG_SEQ_LENS = [
    (4096, 4096),
    (8192, 8192),
]

CAUSAL_MODES = [False, True]

GQA_CONFIGS = [
    (32, 32, "mha"),
    (32, 8, "gqa4"),
    (32, 4, "gqa8"),
    (32, 1, "mqа"),
]


def _make_test_id(dtype_name, head_dim, seqlen_q, seqlen_k, causal, gqa_name):
    c = "causal" if causal else "noncausal"
    return f"{dtype_name}_d{head_dim}_sq{seqlen_q}_sk{seqlen_k}_{c}_{gqa_name}"


class TestFA4Correctness:
    """Test suite for FA4 ROCm correctness."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import the FA4 ROCm module."""
        try:
            from flash_attn_rocm import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.available = True
        except Exception as e:
            self.available = False
            self.import_error = str(e)

    def _skip_if_unavailable(self):
        if not self.available:
            pytest.skip(f"FA4 ROCm not available: {self.import_error}")

    @pytest.mark.parametrize(
        "dtype,atol,dtype_name", DTYPES, ids=[d[2] for d in DTYPES]
    )
    @pytest.mark.parametrize("head_dim", HEAD_DIMS, ids=[f"d{d}" for d in HEAD_DIMS])
    @pytest.mark.parametrize(
        "seqlen_q,seqlen_k", SEQ_LENS,
        ids=[f"sq{s[0]}_sk{s[1]}" for s in SEQ_LENS]
    )
    @pytest.mark.parametrize("causal", CAUSAL_MODES, ids=["noncausal", "causal"])
    def test_basic(self, dtype, atol, dtype_name, head_dim, seqlen_q, seqlen_k, causal):
        """Basic correctness test with MHA (num_heads_q == num_heads_k)."""
        self._skip_if_unavailable()

        batch = 2
        num_heads = 4  # Small for fast testing
        torch.manual_seed(42)

        q = torch.randn(batch, seqlen_q, num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(batch, seqlen_k, num_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(batch, seqlen_k, num_heads, head_dim, dtype=dtype, device="cuda")

        ref = reference_attention(q, k, v, causal=causal)
        out = self.flash_attn_func(q, k, v, causal=causal)

        test_name = _make_test_id(dtype_name, head_dim, seqlen_q, seqlen_k, causal, "mha")
        passed, max_err, mean_err, err_ratio = check_correctness(out, ref, atol, test_name)
        assert passed, f"Correctness failed: max_err={max_err}, ratio={err_ratio}"

    @pytest.mark.parametrize(
        "num_heads_q,num_heads_k,gqa_name", GQA_CONFIGS,
        ids=[g[2] for g in GQA_CONFIGS]
    )
    @pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
    def test_gqa(self, num_heads_q, num_heads_k, gqa_name, causal):
        """Test GQA/MQA configurations."""
        self._skip_if_unavailable()

        batch = 2
        seqlen = 512
        head_dim = 128
        dtype = torch.bfloat16
        torch.manual_seed(42)

        q = torch.randn(batch, seqlen, num_heads_q, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(batch, seqlen, num_heads_k, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(batch, seqlen, num_heads_k, head_dim, dtype=dtype, device="cuda")

        ref = reference_attention(q, k, v, causal=causal)
        out = self.flash_attn_func(q, k, v, causal=causal)

        test_name = f"gqa_{gqa_name}_{'causal' if causal else 'noncausal'}"
        passed, max_err, mean_err, err_ratio = check_correctness(out, ref, ATOL_BF16, test_name)
        assert passed, f"GQA correctness failed: max_err={max_err}"

    @pytest.mark.parametrize(
        "seqlen_q,seqlen_k", LONG_SEQ_LENS,
        ids=[f"sq{s[0]}_sk{s[1]}" for s in LONG_SEQ_LENS]
    )
    def test_long_sequences(self, seqlen_q, seqlen_k):
        """Test with longer sequences (stress test)."""
        self._skip_if_unavailable()

        batch = 1
        num_heads = 8
        head_dim = 128
        dtype = torch.bfloat16
        torch.manual_seed(42)

        q = torch.randn(batch, seqlen_q, num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(batch, seqlen_k, num_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(batch, seqlen_k, num_heads, head_dim, dtype=dtype, device="cuda")

        ref = reference_attention(q, k, v, causal=True)
        out = self.flash_attn_func(q, k, v, causal=True)

        test_name = f"long_sq{seqlen_q}_sk{seqlen_k}"
        passed, max_err, mean_err, err_ratio = check_correctness(out, ref, ATOL_BF16, test_name)
        assert passed, f"Long-seq correctness failed: max_err={max_err}"

    def test_asymmetric_seqlens(self):
        """Test with seqlen_q != seqlen_k."""
        self._skip_if_unavailable()

        batch = 2
        num_heads = 4
        head_dim = 128
        dtype = torch.bfloat16
        torch.manual_seed(42)

        for seqlen_q, seqlen_k in [(256, 1024), (1024, 256), (1, 2048)]:
            q = torch.randn(batch, seqlen_q, num_heads, head_dim, dtype=dtype, device="cuda")
            k = torch.randn(batch, seqlen_k, num_heads, head_dim, dtype=dtype, device="cuda")
            v = torch.randn(batch, seqlen_k, num_heads, head_dim, dtype=dtype, device="cuda")

            # Non-causal for asymmetric (causal with sq < sk needs special handling)
            ref = reference_attention(q, k, v, causal=False)
            out = self.flash_attn_func(q, k, v, causal=False)

            test_name = f"asym_sq{seqlen_q}_sk{seqlen_k}"
            passed, max_err, _, _ = check_correctness(out, ref, ATOL_BF16, test_name)
            assert passed, f"Asymmetric seqlen correctness failed: max_err={max_err}"

    def test_deterministic(self):
        """Verify deterministic output (same input -> same output)."""
        self._skip_if_unavailable()

        batch, seqlen, heads, hdim = 2, 512, 8, 128
        dtype = torch.bfloat16
        torch.manual_seed(42)

        q = torch.randn(batch, seqlen, heads, hdim, dtype=dtype, device="cuda")
        k = torch.randn(batch, seqlen, heads, hdim, dtype=dtype, device="cuda")
        v = torch.randn(batch, seqlen, heads, hdim, dtype=dtype, device="cuda")

        out1 = self.flash_attn_func(q, k, v, causal=True)
        out2 = self.flash_attn_func(q, k, v, causal=True)

        assert torch.equal(out1, out2), "Non-deterministic output detected"

    def test_softmax_scale(self):
        """Test with custom softmax scale."""
        self._skip_if_unavailable()

        batch, seqlen, heads, hdim = 2, 256, 4, 128
        dtype = torch.bfloat16
        torch.manual_seed(42)

        q = torch.randn(batch, seqlen, heads, hdim, dtype=dtype, device="cuda")
        k = torch.randn(batch, seqlen, heads, hdim, dtype=dtype, device="cuda")
        v = torch.randn(batch, seqlen, heads, hdim, dtype=dtype, device="cuda")

        for scale in [0.5, 1.0, 0.1]:
            ref = reference_attention(q, k, v, causal=False, softmax_scale=scale)
            out = self.flash_attn_func(q, k, v, causal=False, softmax_scale=scale)

            passed, max_err, _, _ = check_correctness(out, ref, ATOL_BF16, f"scale_{scale}")
            assert passed, f"Custom scale {scale} failed: max_err={max_err}"


def run_standalone():
    """Run a quick correctness check without pytest."""
    print("=" * 60)
    print("FA4 ROCm Correctness Test (standalone)")
    print("=" * 60)

    try:
        from flash_attn_rocm import flash_attn_func
    except ImportError as e:
        print(f"\nFA4 ROCm not available (will use torch SDPA self-test): {e}")
        print("To test the kernel, run: cd fa4_rocm && pip install -e .")
        flash_attn_func = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("No GPU available, skipping.")
        return

    total = 0
    passed = 0

    test_matrix = [
        # (batch, sq, sk, hq, hk, hdim, causal, dtype, name)
        (2, 128, 128, 4, 4, 128, False, torch.bfloat16, "bf16_short_noncausal"),
        (2, 128, 128, 4, 4, 128, True, torch.bfloat16, "bf16_short_causal"),
        (2, 512, 512, 8, 8, 128, True, torch.bfloat16, "bf16_medium_causal"),
        (2, 2048, 2048, 8, 8, 128, True, torch.bfloat16, "bf16_long_causal"),
        (2, 512, 512, 8, 2, 128, True, torch.bfloat16, "bf16_gqa4_causal"),
        (2, 512, 512, 8, 8, 64, True, torch.bfloat16, "bf16_hdim64_causal"),
        (2, 128, 128, 4, 4, 128, True, torch.float16, "fp16_short_causal"),
    ]

    for batch, sq, sk, hq, hk, hdim, causal, dtype, name in test_matrix:
        total += 1
        torch.manual_seed(42)

        q = torch.randn(batch, sq, hq, hdim, dtype=dtype, device=device)
        k = torch.randn(batch, sk, hk, hdim, dtype=dtype, device=device)
        v = torch.randn(batch, sk, hk, hdim, dtype=dtype, device=device)

        ref = reference_attention(q, k, v, causal=causal)

        if flash_attn_func is not None:
            try:
                out = flash_attn_func(q, k, v, causal=causal)
            except Exception as e:
                print(f"\n  ERROR {name}: {e}")
                continue
        else:
            out = ref  # Self-test mode

        atol = ATOL_BF16 if dtype == torch.bfloat16 else ATOL_FP16
        ok, max_err, mean_err, err_ratio = check_correctness(out, ref, atol, name)
        if ok:
            passed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_standalone()
