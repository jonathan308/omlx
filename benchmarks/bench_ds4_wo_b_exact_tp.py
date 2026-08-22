#!/usr/bin/env python3
"""Gate an exact TP2 row-sharded DS4 ``wo_b`` decode schedule.

The currently shipped attention output path is, for two tensor ranks::

    current = qmv(W, latent_rank0) + qmv(W, latent_rank1)

Both QMV results are rounded to BF16 before JACCL's BF16 sum.  The earlier
``wo_b`` sharding probe instead summed the two 8192-wide latents first.  That
changes the MXFP8 reduction/rounding boundary and is therefore not lossless.

This benchmark measures a different schedule::

    gathered = stack(latent_rank0, latent_rank1)
    local = exact_m2_qmv(W[local_output_rows], gathered)
    local = local[0] + local[1]
    output = all_gather(local)

``exact_m2_qmv`` is the existing DSpark verification kernel.  It deliberately
matches two independent MLX M=1 QMV reductions while loading each weight byte
once for both rows.  Consequently this schedule preserves the current BF16
rounding boundary, halves the ``wo_b`` bytes read per rank, and replaces one
8 KiB all-sum per layer with a 16 KiB latent all-gather plus a 4 KiB output
all-gather.  The script does not initialize distributed MLX; it isolates the
real-weight compute/parity gate on one Mac and prices the two collective
schedules from measured link constants.

Run only while the public model is unloaded, for example::

    python benchmarks/bench_ds4_wo_b_exact_tp.py \
      ~/.lmstudio/models/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from contextlib import ExitStack
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open

from omlx.patches.deepseek_v4.verify_qmv import exact_verify_qmv


LAYERS = 43
HIDDEN_DIMS = 4096
LORA_DIMS = 8192


class _MXFP8Shard:
    """Minimum QuantizedLinear protocol consumed by ``exact_verify_qmv``."""

    bits = 8
    group_size = 32
    mode = "mxfp8"

    def __init__(self, weight: mx.array, scales: mx.array):
        self.weight = weight
        self.scales = scales

    def get(self, name: str, default=None):
        return default

    def __contains__(self, name: str) -> bool:
        return False


def _qmm(value: mx.array, weight: mx.array, scales: mx.array) -> mx.array:
    return mx.quantized_matmul(
        value,
        weight,
        scales=scales,
        biases=None,
        transpose=True,
        group_size=32,
        bits=8,
        mode="mxfp8",
    )


def _load_wo_b(model: Path):
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    stack = ExitStack()
    files = {}
    modules = []
    for layer in range(LAYERS):
        weight_key = f"layers.{layer}.attn.wo_b.weight"
        scales_key = f"layers.{layer}.attn.wo_b.scales"
        filename = index[weight_key]
        if filename not in files:
            files[filename] = stack.enter_context(
                safe_open(model / filename, framework="numpy")
            )
        source = files[filename]
        weight = mx.array(source.get_tensor(weight_key))
        scales = mx.array(source.get_tensor(scales_key))
        split = int(scales.shape[0]) // 2
        modules.append(
            (
                weight,
                scales,
                _MXFP8Shard(weight[:split], scales[:split]),
                _MXFP8Shard(weight[split:], scales[split:]),
            )
        )
    mx.eval(
        [
            value
            for weight, scales, _, _ in modules
            for value in (weight, scales)
        ]
    )
    return stack, modules


def _run_current(modules, local_latent: mx.array) -> None:
    """One rank's current full-output QMV contribution."""

    for weight, scales, _, _ in modules:
        mx.eval(_qmm(local_latent, weight, scales))


def _run_candidate(
    modules,
    gathered_latents: mx.array,
    *,
    shard: int = 0,
) -> None:
    """One rank's two-contribution, half-output exact QMV and BF16 sum."""

    for _, _, left, right in modules:
        projected = exact_verify_qmv(
            left if shard == 0 else right,
            gathered_latents,
        )
        mx.eval(projected[0] + projected[1])


def _time_abba(modules, left, gathered, *, cycles: int):
    # Compile and fault both shapes before timing, then balance allocator/cache
    # inheritance by measuring A-B-B-A in every cycle.
    _run_current(modules, left)
    _run_candidate(modules, gathered)
    mx.synchronize()
    current = []
    candidate = []
    for _ in range(cycles):
        for target, fn in (
            (current, lambda: _run_current(modules, left)),
            (candidate, lambda: _run_candidate(modules, gathered)),
            (candidate, lambda: _run_candidate(modules, gathered)),
            (current, lambda: _run_current(modules, left)),
        ):
            mx.synchronize()
            started = time.perf_counter()
            fn()
            mx.synchronize()
            target.append(time.perf_counter() - started)
    return current, candidate


def _parity(modules, left, right):
    exact = 0
    elements = 0
    max_abs = 0.0
    sum_abs = 0.0
    sum_squared = 0.0
    reference_squared = 0.0
    gathered = mx.concatenate([left, right], axis=0)
    for weight, scales, first, second in modules:
        # This is the shipped two-rank rounding boundary: each rank materializes
        # a BF16 QMV result, then the two BF16 tensors are added.
        reference = _qmm(left, weight, scales) + _qmm(right, weight, scales)
        first_rows = exact_verify_qmv(first, gathered)
        second_rows = exact_verify_qmv(second, gathered)
        candidate = mx.concatenate(
            [first_rows[0] + first_rows[1], second_rows[0] + second_rows[1]],
            axis=-1,
        )
        mx.eval(reference, candidate)
        delta = mx.abs(reference.astype(mx.float32) - candidate.astype(mx.float32))
        exact += int(mx.sum(reference == candidate).item())
        elements += int(reference.size)
        max_abs = max(max_abs, float(mx.max(delta).item()))
        sum_abs += float(mx.sum(delta).item())
        sum_squared += float(mx.sum(delta * delta).item())
        reference_squared += float(
            mx.sum(reference.astype(mx.float32) ** 2).item()
        )
    return {
        "exact_elements": exact,
        "elements": elements,
        "exact_fraction": exact / elements if elements else 1.0,
        "max_abs": max_abs,
        "mean_abs": sum_abs / elements if elements else 0.0,
        "rmse": math.sqrt(sum_squared / elements) if elements else 0.0,
        "relative_l2": (
            math.sqrt(sum_squared / reference_squared)
            if reference_squared > 0
            else 0.0
        ),
    }


def _collective_price(*, latency_s: float, bandwidth: float):
    bf16 = 2
    current_payload = HIDDEN_DIMS * bf16
    latent_payload = LORA_DIMS * bf16
    local_output_payload = (HIDDEN_DIMS // 2) * bf16
    current = LAYERS * (latency_s + current_payload / bandwidth)
    candidate = LAYERS * (
        2 * latency_s
        + (latent_payload + local_output_payload) / bandwidth
    )
    return {
        "current_seconds_43_layers": current,
        "candidate_seconds_43_layers": candidate,
        "added_seconds_43_layers": candidate - current,
        "current_collectives_per_layer": 1,
        "candidate_collectives_per_layer": 2,
        "current_payload_bytes_per_layer": current_payload,
        "candidate_payload_bytes_per_layer": (
            latent_payload + local_output_payload
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--collective-latency-us", type=float, default=31.15)
    parser.add_argument("--collective-bandwidth-gbps", type=float, default=6.178)
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()

    stack, modules = _load_wo_b(args.model.expanduser())
    try:
        mx.random.seed(17)
        left = mx.random.normal((1, LORA_DIMS)).astype(mx.bfloat16)
        right = mx.random.normal((1, LORA_DIMS)).astype(mx.bfloat16)
        gathered = mx.concatenate([left, right], axis=0)
        mx.eval(left, right, gathered)
        current, candidate = _time_abba(
            modules,
            left,
            gathered,
            cycles=args.cycles,
        )
        current_median = statistics.median(current)
        candidate_median = statistics.median(candidate)
        collective = _collective_price(
            latency_s=args.collective_latency_us * 1e-6,
            bandwidth=args.collective_bandwidth_gbps * 1e9,
        )
        full_bytes = sum(
            weight.nbytes + scales.nbytes
            for weight, scales, _, _ in modules
        )
        local_bytes = sum(
            first.weight.nbytes + first.scales.nbytes
            for _, _, first, _ in modules
        )
        result = {
            "layers": len(modules),
            "full_wo_b_bytes_per_rank_current": full_bytes,
            "local_wo_b_bytes_per_rank_candidate": local_bytes,
            "weight_bytes_saved_per_rank_per_token": full_bytes - local_bytes,
            "current_compute_seconds_43_layers": current_median,
            "candidate_compute_seconds_43_layers": candidate_median,
            "isolated_compute_speedup": current_median / candidate_median,
            "collectives": collective,
            "current_compute_plus_collective_seconds": (
                current_median + collective["current_seconds_43_layers"]
            ),
            "candidate_compute_plus_collective_seconds": (
                candidate_median + collective["candidate_seconds_43_layers"]
            ),
            "parity": None if args.skip_parity else _parity(modules, left, right),
            "current_samples_ms": [value * 1000 for value in current],
            "candidate_samples_ms": [value * 1000 for value in candidate],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        stack.close()


if __name__ == "__main__":
    main()
