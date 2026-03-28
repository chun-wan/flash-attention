"""
FA4 ROCm build script.

Install:
    pip install -e .          # editable install (JIT compiles on first use)
    python setup.py install   # pre-compiled install

Requires: ROCm 6.x+, PyTorch with ROCm support.
"""

import os
import subprocess
from pathlib import Path
from setuptools import setup, find_packages

# Try to use torch's CppExtension for AOT compilation.
# Falls back to JIT compilation at import time if this fails.
try:
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    def get_rocm_arch():
        """Detect GPU arch or default to gfx942."""
        try:
            result = subprocess.run(
                ["/opt/rocm/bin/rocminfo"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                if "gfx" in line and "Name:" in line:
                    arch = line.strip().split()[-1]
                    if arch.startswith("gfx"):
                        return arch
        except Exception:
            pass
        return "gfx942"

    rocm_arch = os.environ.get("FA4_ROCM_ARCH", get_rocm_arch())

    kernels_dir = Path(__file__).parent / "kernels"
    sources = [
        str(kernels_dir / "flash_fwd_gfx942.hip"),
        str(kernels_dir / "flash_attn_ops.hip"),
    ]

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [  # hipcc uses the nvcc key in torch's build system
            "-O3",
            f"--offload-arch={rocm_arch}",
            "-std=c++17",
            "-DHEAD_DIM=128",
            "-DBLOCK_M=64",
            "-DBLOCK_N=64",
        ],
    }

    ext_modules = [
        CppExtension(
            name="fa4_rocm.fa4_rocm_ops",
            sources=sources,
            extra_compile_args=extra_compile_args,
            include_dirs=[str(kernels_dir)],
        ),
    ]
    cmdclass = {"build_ext": BuildExtension}
except ImportError:
    ext_modules = []
    cmdclass = {}

setup(
    name="fa4_rocm",
    version="0.1.0",
    description="FlashAttention-4 for AMD ROCm GPUs (MI300X / gfx942)",
    author="AFTT AVO",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2",
    ],
)
