#!/usr/bin/env python3
"""
Build CK FMHA kernel instances as a standalone torch extension.

Uses CK's generate.py codegen to produce kernel blobs, then compiles
them alongside a thin torch C++ wrapper into a Python extension module.

Usage:
    python build_ck_fmha.py [--receipt 600] [--targets gfx942] [--output_dir build]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

AITER_ROOT = os.environ.get("AITER_ROOT", "/opt/aiter")
CK_DIR = os.environ.get("CK_DIR", f"{AITER_ROOT}/3rdparty/composable_kernel")
FMHA_EXAMPLE = f"{CK_DIR}/example/ck_tile/01_fmha"
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")

THIS_DIR = Path(__file__).parent.resolve()


def run_codegen(output_dir: Path, receipt: int, targets: str, direction: str = "fwd"):
    gen_dir = output_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        f"{FMHA_EXAMPLE}/generate.py",
        "-d", direction,
        "--receipt", str(receipt),
        "--targets", targets,
        "-o", str(gen_dir),
    ]
    print(f"[codegen] {' '.join(cmd)}")
    subprocess.check_call(cmd)

    blob_dir = gen_dir / "generated"
    if not blob_dir.exists():
        blob_dir = gen_dir
    blobs = list(blob_dir.rglob("*.cpp")) + list(blob_dir.rglob("*.cu"))
    print(f"[codegen] Generated {len(blobs)} kernel blobs")
    return blobs


def compile_extension(blobs: list, output_dir: Path, targets: str):
    """Compile generated blobs + wrapper into a torch extension."""
    build_dir = output_dir / "build_tmp"
    build_dir.mkdir(parents=True, exist_ok=True)

    wrapper_src = THIS_DIR / "ck_fmha_torch_wrapper.cu"
    if not wrapper_src.exists():
        raise FileNotFoundError(f"Wrapper not found: {wrapper_src}")

    all_sources = [str(wrapper_src)] + [str(b) for b in blobs]

    arch_flags = [f"--offload-arch={t}" for t in targets.split(",")]

    extra_include = [
        f"{CK_DIR}/include",
        f"{FMHA_EXAMPLE}",
        f"{FMHA_EXAMPLE}/codegen",
        f"{AITER_ROOT}/csrc/include",
        str(output_dir / "generated" / "generated"),
        str(output_dir / "generated"),
    ]

    extra_hip_flags = [
        "-O3",
        "-std=c++20",
        "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
        "-DFAV2_ON=1",
        f"-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT={os.environ.get('CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT', '2')}",
        "-fgpu-flush-denormals-to-zero",
        "-mllvm", "--amdgpu-kernarg-preload-count=16",
    ] + arch_flags

    include_flags = []
    for inc in extra_include:
        include_flags.extend(["-I", inc])

    print(f"[compile] Compiling {len(all_sources)} sources for {targets}...")
    print(f"[compile] This may take several minutes...")

    from torch.utils.cpp_extension import load

    module = load(
        name="ck_fmha_ext",
        sources=all_sources,
        extra_cflags=["-O3", "-std=c++20", "-DFAV2_ON=1"],
        extra_cuda_cflags=extra_hip_flags + include_flags,
        build_directory=str(build_dir),
        verbose=True,
    )
    return module


def main():
    parser = argparse.ArgumentParser(description="Build CK FMHA extension")
    parser.add_argument("--receipt", type=int, default=600,
                        help="Codegen receipt (600=aiter fwd+splitkv+bwd)")
    parser.add_argument("--targets", default="gfx942",
                        help="GPU targets (comma-separated)")
    parser.add_argument("--output_dir", default=str(THIS_DIR / "build"),
                        help="Build output directory")
    parser.add_argument("--direction", default="fwd",
                        help="API direction (fwd, bwd, fwd_splitkv)")
    parser.add_argument("--codegen-only", action="store_true",
                        help="Only run codegen, skip compilation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blobs = run_codegen(output_dir, args.receipt, args.targets, args.direction)
    print(f"[codegen] Done: {len(blobs)} blobs in {output_dir}")

    if args.codegen_only:
        print("[codegen-only] Skipping compilation")
        return

    module = compile_extension(blobs, output_dir, args.targets)
    print(f"[build] Extension built: {module}")


if __name__ == "__main__":
    main()
