# SPDX-License-Identifier: Apache-2.0
"""Baseline DS4 sparse-attention verification rows before a window kernel.

This measures the current fused decode primitive at B=M for M=1..6 and checks
it against the exact composed reference. It does not claim KV sharing yet; its
JSON output is the before-state for the verify-window implementation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time

import mlx.core as mx

from omlx.custom_kernels.decode_fast import fast as decode_fast
from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

apply_deepseek_v4_patch()
dsv4 = importlib.import_module("mlx_lm.models.deepseek_v4")


def _dtype(name: str):
    return {"bf16": mx.bfloat16, "fp16": mx.float16}[name]


def _measure(call, *, warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        mx.eval(call())
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        mx.eval(call())
        samples.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(samples)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    return statistics.median(samples), p95


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,2,3,4,5,6")
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--local", type=int, default=128)
    parser.add_argument("--pooled", type=int, default=512)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    dtype = _dtype(args.dtype)
    mx.random.seed(240824)
    results = []
    for rows in (int(item) for item in args.rows.split(",") if item):
        q = mx.random.normal((rows, args.heads, 1, 512), dtype=dtype)
        local = mx.random.normal((rows, 1, args.local, 512), dtype=dtype)
        pooled = mx.random.normal((rows, 1, args.pooled, 512), dtype=dtype)
        sinks = mx.random.normal((args.heads,), dtype=mx.float32)

        def native():
            value = decode_fast.sparse_attn_decode(q, local, pooled, sinks)
            if value is None:
                raise RuntimeError("native sparse_attn_decode is unavailable")
            return value

        def reference():
            return dsv4._dspark_sparse_exact_attention(q, local, pooled, sinks)

        got, expected = native(), reference()
        singles = mx.concatenate(
            [
                decode_fast.sparse_attn_decode(
                    q[index : index + 1],
                    local[index : index + 1],
                    pooled[index : index + 1],
                    sinks,
                )
                for index in range(rows)
            ],
            axis=0,
        )
        mx.eval(got, expected, singles)
        row_invariant = bool(mx.array_equal(got, singles).item())
        max_abs_vs_composed = float(
            mx.abs(got.astype(mx.float32) - expected.astype(mx.float32)).max()
        )
        native_median, native_p95 = _measure(
            native, warmup=args.warmup, iterations=args.iterations
        )
        reference_median, reference_p95 = _measure(
            reference, warmup=args.warmup, iterations=args.iterations
        )
        bytes_per_kv_pass = rows * (args.local + args.pooled) * 512 * 2
        results.append(
            {
                "rows": rows,
                "row_invariant": row_invariant,
                "max_abs_vs_composed": max_abs_vs_composed,
                "native_median_ms": native_median,
                "native_p95_ms": native_p95,
                "reference_median_ms": reference_median,
                "reference_p95_ms": reference_p95,
                "logical_kv_bytes_per_pass": bytes_per_kv_pass,
                "shared_window_floor_bytes_per_pass": (
                    (args.local + args.pooled) * 512 * 2
                ),
            }
        )

    print(
        json.dumps(
            {
                "device": mx.device_info(),
                "dtype": args.dtype,
                "heads": args.heads,
                "local": args.local,
                "pooled": args.pooled,
                "results": results,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
