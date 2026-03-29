"""
FA4-specific scoring extension for the AVO agent loop.

Adds FlashAttention-specific test shapes, correctness verification against
torch SDPA, and TFLOPS-based scoring for attention kernels.

Integrates with the existing kernel_scorer.py infrastructure.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger("avo.fa4_scorer")


@dataclass
class FA4TestShape:
    """Test shape for flash attention kernels."""
    batch_size: int
    seqlen_q: int
    seqlen_k: int
    num_heads_q: int
    num_heads_k: int
    head_dim: int
    causal: bool = False
    dtype: str = "bf16"

    @property
    def flops(self) -> int:
        """Total FLOPs for this attention shape (fwd only)."""
        return 4 * self.batch_size * self.seqlen_q * self.seqlen_k * self.num_heads_q * self.head_dim

    @property
    def name(self) -> str:
        gqa = f"gqa{self.num_heads_q // self.num_heads_k}" if self.num_heads_q != self.num_heads_k else "mha"
        causal_str = "causal" if self.causal else "noncausal"
        return f"b{self.batch_size}_sq{self.seqlen_q}_sk{self.seqlen_k}_h{self.num_heads_q}_d{self.head_dim}_{gqa}_{causal_str}_{self.dtype}"


# Standard shapes covering common LLM configurations
FA4_PREFILL_SHAPES = [
    FA4TestShape(batch_size=2, seqlen_q=2048, seqlen_k=2048, num_heads_q=32, num_heads_k=32, head_dim=128, causal=True, dtype="bf16"),
    FA4TestShape(batch_size=2, seqlen_q=4096, seqlen_k=4096, num_heads_q=32, num_heads_k=32, head_dim=128, causal=True, dtype="bf16"),
    FA4TestShape(batch_size=1, seqlen_q=8192, seqlen_k=8192, num_heads_q=32, num_heads_k=32, head_dim=128, causal=True, dtype="bf16"),
    # GQA shapes (DeepSeek-style)
    FA4TestShape(batch_size=2, seqlen_q=2048, seqlen_k=2048, num_heads_q=32, num_heads_k=8, head_dim=128, causal=True, dtype="bf16"),
    # Different head dims
    FA4TestShape(batch_size=2, seqlen_q=2048, seqlen_k=2048, num_heads_q=32, num_heads_k=32, head_dim=64, causal=True, dtype="bf16"),
    FA4TestShape(batch_size=2, seqlen_q=2048, seqlen_k=2048, num_heads_q=32, num_heads_k=32, head_dim=96, causal=True, dtype="bf16"),
    # Non-causal
    FA4TestShape(batch_size=2, seqlen_q=2048, seqlen_k=2048, num_heads_q=32, num_heads_k=32, head_dim=128, causal=False, dtype="bf16"),
    # FP16
    FA4TestShape(batch_size=2, seqlen_q=2048, seqlen_k=2048, num_heads_q=32, num_heads_k=32, head_dim=128, causal=True, dtype="fp16"),
]

FA4_DECODE_SHAPES = [
    FA4TestShape(batch_size=64, seqlen_q=1, seqlen_k=2048, num_heads_q=32, num_heads_k=32, head_dim=128, causal=False, dtype="bf16"),
    FA4TestShape(batch_size=32, seqlen_q=1, seqlen_k=4096, num_heads_q=32, num_heads_k=32, head_dim=128, causal=False, dtype="bf16"),
    FA4TestShape(batch_size=16, seqlen_q=1, seqlen_k=8192, num_heads_q=32, num_heads_k=32, head_dim=128, causal=False, dtype="bf16"),
    # GQA decode
    FA4TestShape(batch_size=64, seqlen_q=1, seqlen_k=2048, num_heads_q=32, num_heads_k=8, head_dim=128, causal=False, dtype="bf16"),
]


@dataclass
class FA4ScoreResult:
    correct: bool
    tflops: float = 0.0
    latency_us: float = float("inf")
    max_abs_error: float = float("inf")
    mean_abs_error: float = float("inf")
    shape_name: str = ""
    errors: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.tflops if self.correct else 0.0

    @property
    def throughput_us(self) -> float:
        return self.latency_us


class FA4Scorer:
    """Scores FA4 attention kernels against torch SDPA reference.

    When kernel_path is provided, deploys the .co into an overlay directory
    structured as gfx942/<subdir>/MI300/<name>.co and sets AITER_ASM_DIR
    so aiter loads the modified kernel instead of the stock one.
    """

    def __init__(
        self,
        gpu_id: int = 0,
        num_warmup: int = 10,
        num_iters: int = 100,
        atol_bf16: float = 1e-2,
        atol_fp16: float = 5e-3,
        max_error_ratio: float = 0.01,
        fa4_rocm_path: str = "",
        arch: str = "gfx942",
        aiter_root: str = "/opt/aiter",
    ):
        self.gpu_id = gpu_id
        self.num_warmup = num_warmup
        self.num_iters = num_iters
        self.atol_bf16 = atol_bf16
        self.atol_fp16 = atol_fp16
        self.max_error_ratio = max_error_ratio
        self.fa4_rocm_path = fa4_rocm_path or str(Path(__file__).parent.parent.parent / "fa4_rocm")
        self.arch = arch
        self.aiter_root = aiter_root

    def _deploy_co_overlay(self, kernel_co_path: str) -> Optional[str]:
        """Deploy a .co into an overlay directory that mirrors aiter HSA layout.

        Returns the overlay root (to be used as AITER_ASM_DIR), or None on failure.
        The layout is: <overlay_root>/<arch>/<subdir>/<variant>/<name>.co
        """
        co_path = Path(kernel_co_path)
        if not co_path.exists():
            logger.warning("CO file not found: %s", kernel_co_path)
            return None

        overlay_root = co_path.parent.parent  # go up from build/ dir
        if "avo" in str(overlay_root) or "workspace" in str(overlay_root):
            overlay_root = overlay_root / "overlay"
        else:
            overlay_root = Path(tempfile.mkdtemp(prefix="avo_overlay_"))

        subdir = self._infer_subdir(co_path)
        cu_variant = self._get_cu_variant()

        deploy_dir = overlay_root / self.arch / subdir / cu_variant
        deploy_dir.mkdir(parents=True, exist_ok=True)

        deploy_path = deploy_dir / co_path.name
        shutil.copy2(str(co_path), str(deploy_path))

        stock_hsa = Path(self.aiter_root) / "hsa"
        stock_subdir = stock_hsa / self.arch / subdir / cu_variant
        if stock_subdir.exists():
            for stock_co in stock_subdir.glob("*.co"):
                target = deploy_dir / stock_co.name
                if not target.exists():
                    shutil.copy2(str(stock_co), str(target))

        csv_src = stock_hsa / self.arch / subdir
        for csv_file in csv_src.glob("*.csv"):
            csv_dst = overlay_root / self.arch / subdir / csv_file.name
            if not csv_dst.exists():
                shutil.copy2(str(csv_file), str(csv_dst))

        logger.info("Deployed %s to %s", co_path.name, deploy_dir)
        return str(overlay_root)

    def _infer_subdir(self, co_path: Path) -> str:
        parts = co_path.parts
        for i, p in enumerate(parts):
            if p == self.arch and i + 1 < len(parts):
                return parts[i + 1]
        if "fmha" in co_path.name or "fwd_hd" in co_path.name:
            return "fmha_v3_fwd"
        return "fmha_v3_fwd"

    def _get_cu_variant(self) -> str:
        try:
            import subprocess
            r = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True, text=True, timeout=5,
            )
            if "MI300" in r.stdout or "304" in r.stdout:
                return "MI300"
            if "MI308" in r.stdout or "80" in r.stdout or "64" in r.stdout:
                return "MI308"
        except Exception:
            pass
        return "MI300"

    def score_kernel(
        self,
        kernel_path: Optional[str] = None,
        kernel_name: str = "",
        shapes: Optional[List[FA4TestShape]] = None,
        test_shapes=None,
        timeout: int = 300,
    ) -> FA4ScoreResult:
        """
        Score an FA4 kernel on given shapes.

        If kernel_path points to a .co file, deploys it into an overlay and
        sets AITER_ASM_DIR so aiter loads the modified kernel.
        Returns combined score across all shapes.
        """
        if shapes is None:
            shapes = FA4_PREFILL_SHAPES[:3]

        overlay_root = None
        if kernel_path and Path(kernel_path).suffix == ".co" and Path(kernel_path).exists():
            overlay_root = self._deploy_co_overlay(kernel_path)

        results = []
        for shape in shapes:
            result = self._score_single_shape(shape, kernel_path, timeout, overlay_root)
            results.append(result)

        all_correct = all(r.correct for r in results)
        mean_tflops = sum(r.tflops for r in results) / len(results) if results else 0
        mean_latency = sum(r.latency_us for r in results) / len(results) if results else float("inf")
        all_errors = []
        for r in results:
            all_errors.extend(r.errors)

        return FA4ScoreResult(
            correct=all_correct,
            tflops=mean_tflops,
            latency_us=mean_latency,
            shape_name="aggregate",
            errors=all_errors,
        )

    def _score_single_shape(
        self,
        shape: FA4TestShape,
        kernel_path: Optional[str],
        timeout: int,
        overlay_root: Optional[str] = None,
    ) -> FA4ScoreResult:
        """Score on a single shape by running a subprocess benchmark."""
        script = self._generate_test_script(shape, kernel_path)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            env = os.environ.copy()
            env["HIP_VISIBLE_DEVICES"] = str(self.gpu_id)
            env["PYTHONPATH"] = self.fa4_rocm_path + ":" + env.get("PYTHONPATH", "")

            if overlay_root:
                env["AITER_ASM_DIR"] = overlay_root
                logger.info("AITER_ASM_DIR=%s for shape %s", overlay_root, shape.name)

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True,
                timeout=timeout, env=env,
            )

            if result.returncode != 0:
                return FA4ScoreResult(
                    correct=False,
                    shape_name=shape.name,
                    errors=[f"Script failed: {result.stderr[:500]}"],
                )

            return self._parse_results(result.stdout, shape)

        except subprocess.TimeoutExpired:
            return FA4ScoreResult(
                correct=False, shape_name=shape.name,
                errors=["Timeout"],
            )
        except Exception as e:
            return FA4ScoreResult(
                correct=False, shape_name=shape.name,
                errors=[str(e)],
            )
        finally:
            os.unlink(script_path)

    def _generate_test_script(self, shape: FA4TestShape, kernel_path: Optional[str]) -> str:
        dtype_str = "torch.bfloat16" if shape.dtype == "bf16" else "torch.float16"
        atol = self.atol_bf16 if shape.dtype == "bf16" else self.atol_fp16

        return f'''
import torch
import time
import json
import math
import sys

torch.manual_seed(42)
device = "cuda"
dtype = {dtype_str}

batch = {shape.batch_size}
seqlen_q = {shape.seqlen_q}
seqlen_k = {shape.seqlen_k}
num_heads_q = {shape.num_heads_q}
num_heads_k = {shape.num_heads_k}
head_dim = {shape.head_dim}
causal = {shape.causal}
softmax_scale = 1.0 / math.sqrt(head_dim)

q = torch.randn(batch, seqlen_q, num_heads_q, head_dim, dtype=dtype, device=device)
k = torch.randn(batch, seqlen_k, num_heads_k, head_dim, dtype=dtype, device=device)
v = torch.randn(batch, seqlen_k, num_heads_k, head_dim, dtype=dtype, device=device)

# Reference: torch SDPA
q_sdpa = q.transpose(1, 2).contiguous()
k_sdpa = k.transpose(1, 2).contiguous()
if num_heads_q != num_heads_k:
    k_sdpa = k_sdpa.repeat_interleave(num_heads_q // num_heads_k, dim=1)
v_sdpa = v.transpose(1, 2).contiguous()
if num_heads_q != num_heads_k:
    v_sdpa = v_sdpa.repeat_interleave(num_heads_q // num_heads_k, dim=1)

with torch.no_grad():
    ref = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, is_causal=causal, scale=softmax_scale
    ).transpose(1, 2).contiguous()

# Test kernel
try:
    sys.path.insert(0, "{self.fa4_rocm_path}")
    from flash_attn_rocm import flash_attn_func
    with torch.no_grad():
        out = flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale)
except Exception as e:
    print(json.dumps({{"correct": False, "error": str(e)}}))
    sys.exit(0)

# Correctness check
max_abs_err = (out.float() - ref.float()).abs().max().item()
mean_abs_err = (out.float() - ref.float()).abs().mean().item()
correct = max_abs_err < {atol}

# Throughput benchmark
if correct:
    torch.cuda.synchronize()
    for _ in range({self.num_warmup}):
        _ = flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range({self.num_iters}):
        _ = flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_us = elapsed / {self.num_iters} * 1e6
    flops = {shape.flops}
    tflops = flops / (elapsed / {self.num_iters}) / 1e12
else:
    latency_us = float("inf")
    tflops = 0.0

result = {{
    "correct": correct,
    "max_abs_error": max_abs_err,
    "mean_abs_error": mean_abs_err,
    "latency_us": latency_us,
    "tflops": tflops,
    "shape": "{shape.name}",
}}
print("__FA4_SCORE__")
print(json.dumps(result))
print("__FA4_SCORE_END__")
'''

    def _parse_results(self, stdout: str, shape: FA4TestShape) -> FA4ScoreResult:
        try:
            start = stdout.index("__FA4_SCORE__") + len("__FA4_SCORE__")
            end = stdout.index("__FA4_SCORE_END__")
            data = json.loads(stdout[start:end].strip())
            return FA4ScoreResult(
                correct=data["correct"],
                tflops=data.get("tflops", 0),
                latency_us=data.get("latency_us", float("inf")),
                max_abs_error=data.get("max_abs_error", float("inf")),
                mean_abs_error=data.get("mean_abs_error", float("inf")),
                shape_name=data.get("shape", shape.name),
            )
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            return FA4ScoreResult(
                correct=False,
                shape_name=shape.name,
                errors=[f"Failed to parse output: {e}\nStdout: {stdout[:300]}"],
            )
