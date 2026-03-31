#!/usr/bin/env python3
"""AVO 20 rounds on real FlyDSL kernel ISA."""
import os, sys, re, json, shutil, subprocess, time
from pathlib import Path

LLVM_BIN = "/opt/rocm/lib/llvm/bin"
ARCH = "gfx942"
WORKSPACE = "/tmp/avo_real_flydsl"
ISA_SRC = "/tmp/mfma_final/flash_kernel/15_final_isa.s"


def run_cmd(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def assemble(s_path, o_path):
    return run_cmd([
        f"{LLVM_BIN}/llvm-mc", "-triple", "amdgcn-amd-amdhsa",
        f"-mcpu={ARCH}", "--filetype=obj", "-o", str(o_path), str(s_path),
    ]).returncode == 0


def opt_reduce_waitcnt(asm, n=1):
    count = 0
    def reducer(m):
        nonlocal count
        val = int(m.group(1))
        if val > 0:
            count += 1
            return m.group(0).replace(f"vmcnt({val})", f"vmcnt({max(0,val-n)})")
        return m.group(0)
    result = re.sub(r's_waitcnt\s+vmcnt\((\d+)\)', reducer, asm)
    return result, f"Reduced {count} vmcnt by {n}"

def opt_reduce_lgkmcnt(asm, n=1):
    count = 0
    def reducer(m):
        nonlocal count
        val = int(m.group(1))
        if val > 0:
            count += 1
            return m.group(0).replace(f"lgkmcnt({val})", f"lgkmcnt({max(0,val-n)})")
        return m.group(0)
    result = re.sub(r's_waitcnt\s+lgkmcnt\((\d+)\)', reducer, asm)
    return result, f"Reduced {count} lgkmcnt by {n}"

def opt_remove_nops(asm, n=5):
    lines = asm.split("\n")
    new, removed = [], 0
    for l in lines:
        if l.strip().startswith("s_nop") and removed < n:
            removed += 1; continue
        new.append(l)
    return "\n".join(new), f"Removed {removed} NOPs"

def opt_remove_dup_barriers(asm):
    lines = asm.split("\n")
    new, prev, removed = [], False, 0
    for l in lines:
        is_b = l.strip() == "s_barrier"
        if is_b and prev: removed += 1; continue
        new.append(l); prev = is_b
    return "\n".join(new), f"Removed {removed} dup barriers"

def opt_remove_self_moves(asm):
    count = 0
    def rm(m):
        nonlocal count
        if m.group(1) == m.group(2): count += 1; return ""
        return m.group(0)
    r = re.sub(r'v_mov_b32\w*\s+(v\d+),\s+(v\d+)', rm, asm)
    return r, f"Removed {count} self-moves"

STRATEGIES = [
    lambda a: opt_reduce_waitcnt(a, 1),
    lambda a: opt_reduce_lgkmcnt(a, 1),
    lambda a: opt_reduce_waitcnt(a, 2),
    lambda a: opt_reduce_lgkmcnt(a, 2),
    lambda a: opt_reduce_waitcnt(a, 3),
    lambda a: opt_remove_nops(a, 3),
    lambda a: opt_remove_dup_barriers(a),
    lambda a: opt_remove_self_moves(a),
    lambda a: opt_remove_nops(a, 5),
    lambda a: opt_remove_dup_barriers(a),
    lambda a: opt_reduce_waitcnt(a, 1),
    lambda a: opt_reduce_lgkmcnt(a, 1),
    lambda a: opt_reduce_waitcnt(a, 2),
    lambda a: opt_reduce_lgkmcnt(a, 2),
    lambda a: opt_reduce_waitcnt(a, 4),
    lambda a: opt_remove_nops(a, 10),
    lambda a: opt_remove_self_moves(a),
    lambda a: opt_reduce_waitcnt(a, 1),
    lambda a: opt_reduce_lgkmcnt(a, 1),
    lambda a: opt_remove_dup_barriers(a),
]


def main():
    ws = Path(WORKSPACE)
    ws.mkdir(parents=True, exist_ok=True)
    kernels_dir = ws / "kernels"
    kernels_dir.mkdir(exist_ok=True)
    build_dir = ws / "build"
    build_dir.mkdir(exist_ok=True)

    if not Path(ISA_SRC).exists():
        print(f"ISA not found at {ISA_SRC}")
        print("Run flash_fwd_real_flydsl.py first with FLIR_DUMP_IR=1")
        return

    s_path = kernels_dir / "flash_kernel.s"
    shutil.copy2(ISA_SRC, str(s_path))
    raw_asm = s_path.read_text()
    orig_lines = raw_asm.count("\n")
    print(f"FlyDSL ISA: {orig_lines} lines")

    # Verify roundtrip
    o_path = build_dir / "flash_kernel.o"
    if assemble(s_path, o_path):
        print("Roundtrip: OK")
    else:
        print("Roundtrip: FAILED")
        return

    current_asm = raw_asm
    results = [{"round": 0, "description": "baseline", "lines": orig_lines}]

    print(f"\n{'='*60}")
    print(f"AVO 20 rounds on real FlyDSL ISA ({orig_lines} lines)")
    print(f"{'='*60}")

    for r_idx in range(20):
        r_num = r_idx + 1
        strategy = STRATEGIES[r_idx]
        modified, desc = strategy(current_asm)

        if modified == current_asm:
            print(f"R{r_num:02d}: SKIP -- {desc}")
            results.append({"round": r_num, "description": f"SKIP: {desc}", "changed": False, "build": False})
            continue

        test_s = kernels_dir / f"r{r_num}.s"
        test_s.write_text(modified)
        test_o = build_dir / f"r{r_num}.o"

        if assemble(test_s, test_o):
            new_lines = modified.count("\n")
            current_asm = modified
            print(f"R{r_num:02d}: BUILD OK | {desc} | {orig_lines}->{new_lines} lines")
            results.append({"round": r_num, "description": desc, "changed": True, "build": True, "lines": new_lines})
        else:
            print(f"R{r_num:02d}: BUILD FAIL | {desc}")
            results.append({"round": r_num, "description": desc, "changed": True, "build": False})

    final_lines = current_asm.count("\n")
    final_s = kernels_dir / "final.s"
    final_s.write_text(current_asm)

    print(f"\n{'='*60}")
    print(f"REAL FLYDSL AVO SUMMARY")
    print(f"{'='*60}")
    print(f"Original: {orig_lines} lines")
    print(f"Final:    {final_lines} lines")
    print(f"Reduction: {orig_lines - final_lines} lines ({(1 - final_lines/max(orig_lines,1))*100:.1f}%)")
    print(f"Successful builds: {sum(1 for r in results if r.get('build'))}")

    log_path = ws / "real_flydsl_avo.json"
    with open(log_path, "w") as f:
        json.dump({"orig_lines": orig_lines, "final_lines": final_lines, "rounds": results}, f, indent=2)
    print(f"\nLog: {log_path}")


if __name__ == "__main__":
    main()
