#!/usr/bin/env python3
"""Test KVTC compression on MLA-style KV cache vectors.

Simulates Kimi K2.5's MLA KV cache (576-dim per token per layer)
and measures compression ratio + decompression error.
"""
import torch
import time
import json
import math


def test_kvtc_mla():
    from sglang_kvtc_hook import SGLangKVTCManager

    # Kimi K2.5 MLA params
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    kv_dim = kv_lora_rank + qk_rope_head_dim  # 576
    num_layers = 64
    seq_len = 10240  # typical long context
    batch = 40  # concurrency

    print("=" * 60)
    print("KVTC MLA KV Cache Compression Test")
    print("=" * 60)
    print(f"KV dim: {kv_dim}, Layers: {num_layers}, Seq: {seq_len}, Batch: {batch}")

    # Simulate MLA KV cache data (randn ~ latent space)
    torch.manual_seed(42)

    results = {}
    for bits in [2, 4, 8]:
        mgr = SGLangKVTCManager(kv_cache_dim=kv_dim, target_bits=bits)

        total_err = 0.0
        total_max_err = 0.0
        n_blocks = 0

        t0 = time.perf_counter()
        for layer in range(num_layers):
            kv_data = torch.randn(seq_len, 1, kv_dim, dtype=torch.bfloat16, device="cuda")
            key = f"layer{layer}_test"
            mgr.compress_kv_block(key, kv_data)
            decompressed = mgr.decompress_kv_block(key, device="cuda")

            err = (kv_data.float() - decompressed.float()).abs()
            total_err += err.mean().item()
            total_max_err = max(total_max_err, err.max().item())
            n_blocks += 1
            mgr.evict(key)
        elapsed = time.perf_counter() - t0

        stats = mgr.stats
        mean_err = total_err / n_blocks

        # Memory savings calculation
        original_size_mb = num_layers * seq_len * kv_dim * 2 / 1e6  # bf16 = 2 bytes
        compressed_size_mb = original_size_mb / stats["compression_ratio"]
        saved_mb = original_size_mb - compressed_size_mb

        print(f"\n--- {bits}-bit quantization ---")
        print(f"  Compression ratio: {stats['compression_ratio']:.1f}x")
        print(f"  Mean error: {mean_err:.6f}")
        print(f"  Max error: {total_max_err:.6f}")
        print(f"  Time: {elapsed:.2f}s ({elapsed/num_layers*1000:.1f}ms per layer)")
        print(f"  Memory: {original_size_mb:.1f} MB -> {compressed_size_mb:.1f} MB (saved {saved_mb:.1f} MB)")

        results[f"{bits}bit"] = {
            "compression_ratio": round(stats["compression_ratio"], 1),
            "mean_error": round(mean_err, 6),
            "max_error": round(total_max_err, 6),
            "time_s": round(elapsed, 2),
            "original_mb": round(original_size_mb, 1),
            "compressed_mb": round(compressed_size_mb, 1),
            "saved_mb": round(saved_mb, 1),
        }

    # Per-batch estimate for actual serving
    print(f"\n--- Serving memory estimate (batch={batch}) ---")
    total_kv_mb = num_layers * seq_len * kv_dim * 2 * batch / 1e6
    for bits in [2, 4, 8]:
        ratio = results[f"{bits}bit"]["compression_ratio"]
        compressed_total = total_kv_mb / ratio
        print(f"  {bits}-bit: {total_kv_mb:.0f} MB -> {compressed_total:.0f} MB (saved {total_kv_mb - compressed_total:.0f} MB = {(1-1/ratio)*100:.0f}%)")

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    test_kvtc_mla()
