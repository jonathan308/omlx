#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate the experimental DS4 full routed-MoE primitive on a real layer.

The script maps only the safetensor shards that contain one routed expert
layer, stacks its three 256-expert banks exactly as the model sanitizer does,
and alternates the stock composed SwitchGLU with the two-dispatch candidate.
It never changes checkpoint bytes or server state.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from omlx.custom_kernels.glm_moe_dsa import fast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=1, choices=range(1, 9))
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--activation-limit", type=float, default=10.0)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument(
        "--require-gate",
        action="store_true",
        help="return nonzero unless parity is exact and min-speedup is met",
    )
    return parser.parse_args()


def load_layer(model_dir: Path, layer: int):
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    prefix = f"layers.{layer}.ffn.experts."
    keys = [key for key in index["weight_map"] if key.startswith(prefix)]
    shard_names = sorted({index["weight_map"][key] for key in keys})
    tensors = {}
    for shard_name in shard_names:
        tensors.update(mx.load(str(model_dir / shard_name)))

    def stack(projection: str, suffix: str):
        return mx.stack(
            [
                tensors[f"{prefix}{expert}.{projection}.{suffix}"]
                for expert in range(256)
            ]
        )

    # Checkpoint naming: w1=gate, w2=down, w3=up.
    gate_weight, gate_scales = stack("w1", "weight"), stack("w1", "scales")
    down_weight, down_scales = stack("w2", "weight"), stack("w2", "scales")
    up_weight, up_scales = stack("w3", "weight"), stack("w3", "scales")
    mx.eval(
        gate_weight,
        gate_scales,
        down_weight,
        down_scales,
        up_weight,
        up_scales,
    )
    mx.synchronize()
    return (
        up_weight,
        up_scales,
        gate_weight,
        gate_scales,
        down_weight,
        down_scales,
        shard_names,
    )


def reference(
    x,
    up_weight,
    up_scales,
    gate_weight,
    gate_scales,
    down_weight,
    down_scales,
    indices,
    scores,
    activation_limit,
):
    expanded = mx.expand_dims(x, (-2, -3))
    kwargs = {
        "transpose": True,
        "group_size": 32,
        "bits": 4,
        "mode": "mxfp4",
        "sorted_indices": False,
    }
    up = mx.gather_qmm(
        expanded,
        up_weight,
        up_scales,
        None,
        rhs_indices=indices,
        **kwargs,
    )
    gate = mx.gather_qmm(
        expanded,
        gate_weight,
        gate_scales,
        None,
        rhs_indices=indices,
        **kwargs,
    )
    gate = mx.minimum(gate, activation_limit)
    up = mx.clip(up, -activation_limit, activation_limit)
    activated = (gate * mx.sigmoid(gate)) * up
    down = mx.gather_qmm(
        activated,
        down_weight,
        down_scales,
        None,
        rhs_indices=indices,
        **kwargs,
    ).squeeze(-2)
    return (down * scores[..., None].astype(down.dtype)).sum(-2)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> int:
    args = parse_args()
    if not fast.is_native_available() or not fast.has_symbol(
        "deepseek_mxfp4_full_decode"
    ):
        raise RuntimeError("deepseek_mxfp4_full_decode native symbol is unavailable")

    (
        up_weight,
        up_scales,
        gate_weight,
        gate_scales,
        down_weight,
        down_scales,
        shard_names,
    ) = load_layer(args.model, args.layer)
    input_dims = up_weight.shape[2] * 8
    dtype = getattr(mx, args.dtype)
    mx.random.seed(20260822)
    x = mx.random.normal((1, args.tokens, input_dims)).astype(dtype)
    base_routes = [round(slot * 255 / 5) for slot in range(6)]
    indices = mx.array(
        [
            [
                [int((expert + row * 17) % 256) for expert in base_routes]
                for row in range(args.tokens)
            ]
        ],
        dtype=mx.uint32,
    )
    scores = mx.softmax(
        mx.random.normal((1, args.tokens, 6)).astype(mx.float32),
        axis=-1,
    )
    mx.eval(x, indices, scores)

    common = (
        x,
        up_weight,
        up_scales,
        gate_weight,
        gate_scales,
        down_weight,
        down_scales,
        indices,
        scores,
    )

    def stock():
        return reference(*common, args.activation_limit)

    def candidate():
        return fast.deepseek_mxfp4_full_decode(*common, args.activation_limit)

    for _ in range(args.warmup):
        mx.eval(stock())
        mx.synchronize()
        mx.eval(candidate())
        mx.synchronize()

    expected, actual = stock(), candidate()
    mx.eval(expected, actual)
    mx.synchronize()
    difference = mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32))
    exact = bool(mx.array_equal(expected, actual).item())
    max_abs = float(mx.max(difference).item())
    nonzero = int(mx.sum(difference != 0).item())

    timings = {"stock": [], "candidate": []}
    for iteration in range(args.iterations):
        order = ("stock", "candidate") if iteration % 2 == 0 else (
            "candidate",
            "stock",
        )
        for name in order:
            started = time.perf_counter_ns()
            mx.eval(stock() if name == "stock" else candidate())
            mx.synchronize()
            timings[name].append((time.perf_counter_ns() - started) / 1e6)

    stock_stats = summarize(timings["stock"])
    candidate_stats = summarize(timings["candidate"])
    speedup = stock_stats["median_ms"] / candidate_stats["median_ms"]
    passed = exact and speedup >= args.min_speedup
    print(
        json.dumps(
            {
                "model": str(args.model),
                "layer": args.layer,
                "shards": shard_names,
                "tokens": args.tokens,
                "dtype": args.dtype,
                "shapes": {
                    "up": list(up_weight.shape),
                    "gate": list(gate_weight.shape),
                    "down": list(down_weight.shape),
                },
                "parity": {
                    "exact": exact,
                    "max_abs": max_abs,
                    "nonzero": nonzero,
                },
                "stock": stock_stats,
                "candidate": candidate_stats,
                "speedup": speedup,
                "min_speedup": args.min_speedup,
                "passed": passed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed or not args.require_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
