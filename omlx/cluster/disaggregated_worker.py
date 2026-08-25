# SPDX-License-Identifier: Apache-2.0
"""Experimental full-replica prefill/decode disaggregation worker.

Rank 0 owns prompt processing. Rank 1 owns the handed-off decode cache. Both
ranks load the same full model; only cache tensors and the first sampled token
cross the data plane. This is intentionally a bounded worker/benchmark, not a
serving route. It proves the universal cache wire contract before scheduler and
HTTP lifecycle integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Any

from .cache_transfer import (
    prepare_cache_transfer,
    recv_cache_transfer,
    send_cache_transfer,
)


EVENT_PREFIX = "OMLX_DISAGG_EVENT:"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("jaccl", "ring"), default="jaccl")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--completion-tokens", type=int, default=32)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--prefill-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--state-dir", default="~/.omlx/cluster/runtime-disaggregated"
    )
    parser.add_argument("--deployment-id", default="disaggregated-prefill-decode")
    return parser.parse_args()


def _event(payload: dict[str, Any]) -> None:
    print(EVENT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)


def _prompt_tokens(tokenizer: Any, count: int) -> list[int]:
    if count < 2:
        raise ValueError("disaggregated prompt must contain at least two tokens")
    seed = tokenizer.encode(
        " Apple Silicon prefill decode disaggregation over RDMA.",
        add_special_tokens=False,
    )
    if not seed:
        raise RuntimeError("tokenizer produced an empty benchmark seed")
    prefix = []
    bos = getattr(tokenizer, "bos_token_id", None)
    if isinstance(bos, int) and bos >= 0:
        prefix.append(bos)
    needed = count - len(prefix)
    return prefix + (seed * math.ceil(needed / len(seed)))[:needed]


def _prefill_call_count(prompt_tokens: int, step: int) -> int:
    if prompt_tokens < 2 or step < 1:
        raise ValueError("invalid prompt/step size")
    return math.ceil((prompt_tokens - 1) / step) + 1


def _heartbeat(mx: Any, group: Any, rank: int) -> None:
    value = mx.distributed.all_sum(
        mx.array([1.0], dtype=mx.float32), group=group
    )
    mx.eval(value)
    expected = int(group.size())
    actual = float(value.item())
    if actual != float(expected):
        raise RuntimeError(
            "disaggregated control heartbeat lost a rank: "
            f"rank={rank} expected={expected} actual={actual}"
        )


def _cache_states(cache: list[Any]) -> list[Any]:
    return [entry.state for entry in cache]


def _prefill(
    mx: Any,
    model: Any,
    cache: list[Any],
    tokens: list[int],
    *,
    step: int,
    group: Any,
    rank: int,
) -> tuple[int, float, int]:
    started = time.perf_counter()
    values = mx.array(tokens, dtype=mx.int32)
    processed = 0
    calls = 0
    while len(tokens) - processed > 1:
        width = min(step, (len(tokens) - processed) - 1)
        _ = model(values[None, processed : processed + width], cache=cache)
        mx.eval(_cache_states(cache))
        processed += width
        calls += 1
        _heartbeat(mx, group, rank)
        mx.clear_cache()
    logits = model(values[None, processed:], cache=cache)[:, -1, :]
    first = mx.argmax(logits, axis=-1)
    mx.eval(first, _cache_states(cache))
    calls += 1
    _heartbeat(mx, group, rank)
    return int(first.item()), time.perf_counter() - started, calls


def _wait_for_prefill(
    mx: Any,
    group: Any,
    rank: int,
    *,
    prompt_tokens: int,
    step: int,
) -> None:
    for _ in range(_prefill_call_count(prompt_tokens, step)):
        _heartbeat(mx, group, rank)


def _fixed_greedy_decode(
    mx: Any,
    model: Any,
    cache: list[Any],
    first_token: int,
    count: int,
) -> tuple[list[int], float]:
    if count < 1:
        raise ValueError("completion token count must be positive")
    result = [int(first_token)]
    current = int(first_token)
    started = time.perf_counter()
    for _ in range(count - 1):
        value = mx.array([[current]], dtype=mx.int32)
        logits = model(value, cache=cache)[:, -1, :]
        next_token = mx.argmax(logits, axis=-1)
        mx.eval(next_token)
        current = int(next_token.item())
        result.append(current)
    mx.synchronize()
    return result, time.perf_counter() - started


def _token_hash(tokens: list[int]) -> str:
    payload = ",".join(map(str, tokens)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def run(args: argparse.Namespace) -> int:
    if args.prompt_tokens < 2 or args.completion_tokens < 1:
        raise ValueError("prompt/completion sizes are too small")
    if args.prefill_step_size < 1:
        raise ValueError("prefill step size must be positive")

    from omlx._torch_stub import install as install_torch_stub

    install_torch_stub()

    import mlx.core as mx

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    from .jaccl_lease import acquire_jaccl_communicator_lease
    from .jaccl_side_channel import init_cluster_group
    from .staging import model_identity_digest
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    model_path = Path(args.model).expanduser().resolve()
    identity = model_identity_digest(model_path)
    maybe_apply_pre_load_patches(
        model_path,
        model_settings=SimpleNamespace(mtp_enabled=False, mtp_num_draft_tokens=0),
    )
    load_started = time.perf_counter()
    model, tokenizer = load(str(model_path), lazy=False, trust_remote_code=False)
    model.eval()
    load_seconds = time.perf_counter() - load_started

    lease = (
        acquire_jaccl_communicator_lease(
            deployment_id=args.deployment_id,
            state_dir=args.state_dir,
        )
        if args.backend == "jaccl"
        else None
    )
    try:
        group = init_cluster_group(mx, backend=args.backend, strict=True)
        rank = int(group.rank())
        if group.size() != 2:
            raise RuntimeError("disaggregated prototype currently requires two ranks")
        prefill_rank = int(args.prefill_rank)
        decode_rank = 1 - prefill_rank
        _heartbeat(mx, group, rank)
        tokens = _prompt_tokens(tokenizer, args.prompt_tokens)

        _event(
            {
                "type": "rank_loaded",
                "rank": rank,
                "role": "prefill" if rank == prefill_rank else "decode",
                "model_identity": identity,
                "load_seconds": load_seconds,
                "peak_memory_bytes": int(mx.get_peak_memory()),
            }
        )

        if rank == prefill_rank:
            cache = make_prompt_cache(model)
            first_token, prefill_seconds, calls = _prefill(
                mx,
                model,
                cache,
                tokens,
                step=args.prefill_step_size,
                group=group,
                rank=rank,
            )
            prepared = prepare_cache_transfer(
                cache,
                model_identity=identity,
                prompt_tokens=len(tokens),
            )
            transfer = send_cache_transfer(
                mx, prepared, dst=decode_rank, group=group
            )
            first_array = mx.array([first_token], dtype=mx.int32)
            mx.eval(
                mx.distributed.send(first_array, decode_rank, group=group)
            )
            # Finish the prefill->decode direction before either rank posts a
            # result in the reverse direction. Lazy sibling point-to-point
            # operations may otherwise be topologically reordered.
            mx.synchronize()

            baseline_tokens, baseline_decode_seconds = _fixed_greedy_decode(
                mx,
                model,
                cache,
                first_token,
                args.completion_tokens,
            )
            remote_tokens_array = mx.distributed.recv(
                (args.completion_tokens,),
                mx.int32,
                decode_rank,
                group=group,
            )
            mx.eval(remote_tokens_array)
            mx.synchronize()
            remote_metrics = mx.distributed.recv(
                (2,), mx.float32, decode_rank, group=group
            )
            mx.eval(remote_metrics)
            mx.synchronize()
            remote_tokens = [int(value) for value in remote_tokens_array.tolist()]
            remote_decode_seconds, remote_recv_seconds = (
                float(value) for value in remote_metrics.tolist()
            )
            parity = baseline_tokens == remote_tokens
            report = {
                "type": "result",
                "backend": args.backend,
                "prefill_rank": prefill_rank,
                "decode_rank": decode_rank,
                "model": str(model_path),
                "model_identity": identity,
                "prompt_tokens": len(tokens),
                "completion_tokens": args.completion_tokens,
                "prefill_calls": calls,
                "prefill_seconds": prefill_seconds,
                "prefill_tokens_per_second": len(tokens) / prefill_seconds,
                "cache_arrays": transfer.array_count,
                "cache_tensor_bytes": transfer.tensor_bytes,
                "cache_manifest_bytes": transfer.manifest_bytes,
                "cache_send_seconds": transfer.elapsed_seconds,
                "cache_send_bytes_per_second": transfer.bytes_per_second,
                "cache_recv_seconds": remote_recv_seconds,
                "baseline_decode_seconds": baseline_decode_seconds,
                "baseline_decode_tokens_per_second": (
                    max(0, args.completion_tokens - 1) / baseline_decode_seconds
                    if baseline_decode_seconds > 0
                    else 0.0
                ),
                "remote_decode_seconds": remote_decode_seconds,
                "remote_decode_tokens_per_second": (
                    max(0, args.completion_tokens - 1) / remote_decode_seconds
                    if remote_decode_seconds > 0
                    else 0.0
                ),
                "parity": parity,
                "baseline_token_sha256": _token_hash(baseline_tokens),
                "remote_token_sha256": _token_hash(remote_tokens),
                "first_token": first_token,
                "baseline_text": tokenizer.decode(baseline_tokens),
                "remote_text": tokenizer.decode(remote_tokens),
            }
            _event(report)
            return 0 if parity else 2

        _wait_for_prefill(
            mx,
            group,
            rank,
            prompt_tokens=args.prompt_tokens,
            step=args.prefill_step_size,
        )
        cache, manifest, recv_stats = recv_cache_transfer(
            mx,
            src=prefill_rank,
            group=group,
            expected_model_identity=identity,
        )
        first_array = mx.distributed.recv(
            (1,), mx.int32, prefill_rank, group=group
        )
        mx.eval(first_array)
        mx.synchronize()
        first_token = int(first_array.item())
        remote_tokens, decode_seconds = _fixed_greedy_decode(
            mx,
            model,
            cache,
            first_token,
            args.completion_tokens,
        )
        token_array = mx.array(remote_tokens, dtype=mx.int32)
        metrics = mx.array(
            [decode_seconds, recv_stats.elapsed_seconds], dtype=mx.float32
        )
        mx.eval(
            mx.distributed.send(token_array, prefill_rank, group=group)
        )
        mx.synchronize()
        mx.eval(mx.distributed.send(metrics, prefill_rank, group=group))
        # JACCL send is an MLX primitive. Evaluation queues it, while an
        # explicit stream drain keeps the rank process and source buffers alive
        # until the peer has consumed the final result frames.
        mx.synchronize()
        _event(
            {
                "type": "decode_complete",
                "rank": rank,
                "prompt_tokens": manifest["prompt_tokens"],
                "cache_arrays": recv_stats.array_count,
                "cache_tensor_bytes": recv_stats.tensor_bytes,
                "cache_recv_seconds": recv_stats.elapsed_seconds,
                "decode_seconds": decode_seconds,
                "token_sha256": _token_hash(remote_tokens),
            }
        )
        return 0
    finally:
        if lease is not None:
            lease.close()


def main() -> int:
    try:
        return run(_arguments())
    except BaseException as exc:
        _event(
            {
                "type": "error",
                "rank": int(os.environ.get("MLX_RANK", "-1")),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
