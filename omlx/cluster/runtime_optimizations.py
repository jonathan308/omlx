# SPDX-License-Identifier: Apache-2.0
"""Capability-gated optimizations for the pinned MLX-LM pipeline worker."""

from __future__ import annotations

import importlib
import inspect
import math
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from typing import Any

from .performance import ExecutionSettings

_PREFILL_CLEAR_CACHE_EVERY_ENV = "OMLX_CLUSTER_PREFILL_CLEAR_CACHE_EVERY"
_DSV4_ADAPTIVE_PREFILL_ENV = "OMLX_DSV4_ADAPTIVE_PREFILL"
_DSV4_ADAPTIVE_PREFILL_AFTER_ENV = "OMLX_DSV4_ADAPTIVE_PREFILL_AFTER"
_DSV4_ADAPTIVE_PREFILL_STEP_ENV = "OMLX_DSV4_ADAPTIVE_PREFILL_STEP"
_DSV4_ADAPTIVE_PREFILL_MAX_BASE_ENV = "OMLX_DSV4_ADAPTIVE_PREFILL_MAX_BASE"
_DSV4_PREFILL_YIELD_ENV = "OMLX_DSV4_PREFILL_YIELD"
_DSV4_PREFILL_ASYNC_DEPTH_ENV = "OMLX_DSV4_PREFILL_ASYNC_DEPTH"
_CLUSTER_PREFILL_SHAPE_WARMUP_ENV = "OMLX_CLUSTER_PREFILL_SHAPE_WARMUP"
_DSV4_PREFILL_STEP_TRACE_ENV = "OMLX_DSV4_PREFILL_STEP_TRACE"


def _capability(
    *,
    enabled: bool,
    active: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "active": bool(active),
        "reason": reason,
    }


def _prefill_clear_cache_every() -> int:
    """Allocator-cache purge cadence; zero reuses buffers until prompt end."""

    raw = os.environ.get(_PREFILL_CLEAR_CACHE_EVERY_ENV, "0").strip()
    try:
        every = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_PREFILL_CLEAR_CACHE_EVERY_ENV} must be a non-negative integer"
        ) from exc
    if every < 0:
        raise ValueError(
            f"{_PREFILL_CLEAR_CACHE_EVERY_ENV} must be a non-negative integer"
        )
    return every


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _deepseek_v4_prefill_async_depth() -> int:
    """Return the only validated DS4 TP prefill graph depths: off or two."""

    raw = os.environ.get(_DSV4_PREFILL_ASYNC_DEPTH_ENV, "0").strip()
    try:
        depth = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_DSV4_PREFILL_ASYNC_DEPTH_ENV} must be 0 or 2"
        ) from exc
    if depth not in {0, 2}:
        raise ValueError(f"{_DSV4_PREFILL_ASYNC_DEPTH_ENV} must be 0 or 2")
    return depth


def _mtp_decode_enabled(model: Any) -> bool:
    return any(
        bool(getattr(candidate, "_omlx_mtp_decode_enabled", False))
        for candidate in (
            model,
            getattr(model, "language_model", None),
            getattr(model, "_language_model", None),
        )
        if candidate is not None
    )


def _deepseek_v4_adaptive_prefill(
    model: Any,
    execution: ExecutionSettings,
    *,
    pipeline_parallel: bool,
) -> tuple[bool, bool, int, int, int, str]:
    """Choose the measured DS4 chunk schedule that avoids context taper."""

    enabled = os.environ.get(_DSV4_ADAPTIVE_PREFILL_ENV, "1").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }
    after = _positive_env_int(_DSV4_ADAPTIVE_PREFILL_AFTER_ENV, 4096)
    step = _positive_env_int(_DSV4_ADAPTIVE_PREFILL_STEP_ENV, 1024)
    max_base = _positive_env_int(_DSV4_ADAPTIVE_PREFILL_MAX_BASE_ENV, 2048)
    layers = getattr(getattr(model, "model", None), "layers", None)
    is_ds4 = any(
        bool(getattr(getattr(layer, "attn", None), "dspark", False))
        for layer in (layers if isinstance(layers, list) else ())
    )
    base = min(int(execution.prefill_step_size), max_base)
    active = bool(enabled and is_ds4 and not pipeline_parallel and base > step)
    if active:
        reason = (
            f"DS4 uses {base}-token chunks for short prompts and "
            f"{step}-token chunks from the start above {after} tokens"
        )
    elif not enabled:
        reason = "DS4 adaptive prefill is disabled by the operator"
    elif not is_ds4:
        reason = "model is not DeepSeek V4 Flash"
    elif pipeline_parallel:
        reason = "DS4 adaptive prefill currently requires tensor parallelism"
    else:
        reason = f"configured prefill step is already {base} tokens"
    return enabled, active, after, step, base, reason


def _safe_batch_len(batch: Any) -> int:
    """Return a batch's row count without assuming a concrete MLX-LM type."""

    try:
        return max(0, int(len(batch)))
    except (AttributeError, TypeError, ValueError):
        try:
            return max(0, int(len(getattr(batch, "uids", ()))))
        except (AttributeError, TypeError, ValueError):
            return 0


def _at_generation_boundary(segments: Any) -> bool:
    """Whether a staged prompt has only its one-token decode seed left."""

    try:
        return len(segments) == 1 and len(segments[0]) == 1
    except (TypeError, IndexError):
        return False


def _known_prompt_total(instance: Any, uid: Any) -> int | None:
    """Read the rank-identical full prompt length recorded by telemetry."""

    prompt_batch = getattr(instance, "_prompt_batch", None)
    totals = getattr(prompt_batch, "_omlx_total_prompt_lengths", None)
    if isinstance(totals, dict) and uid in totals:
        try:
            return max(0, int(totals[uid]))
        except (TypeError, ValueError):
            pass

    # ``install_server_telemetry`` keeps the exact full token sequence for
    # every uid.  Its length is independent of how much prefix cache each rank
    # restored, so it is safe to use for a lockstep scheduling decision.
    tokens = getattr(instance, "_omlx_tokens", None)
    if isinstance(tokens, dict) and uid in tokens:
        try:
            return max(0, int(len(tokens[uid])))
        except (TypeError, ValueError):
            pass
    return None


def _current_prompt_total(
    instance: Any,
    uid: Any,
    index: int,
    staged: Any,
) -> int | None:
    """Recover one current row's original length without telemetry metadata."""

    known = _known_prompt_total(instance, uid)
    if known is not None:
        return known
    try:
        processed = max(0, int(staged[1]))
        remaining_total = max(0, int(staged[2]))
        stored = getattr(instance._prompt_batch, "tokens", ())[index]
        cached_prefix = max(0, len(stored) - processed)
        return cached_prefix + remaining_total
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _pending_prompt_total(instance: Any, staged: Any) -> int | None:
    """Recover one queued row's original length from BatchGenerator state."""

    try:
        uid = staged[0]
    except (IndexError, TypeError):
        return None
    known = _known_prompt_total(instance, uid)
    if known is not None:
        return known
    try:
        segments = staged[1]
        cached_prefix = staged[4]
        return len(cached_prefix) + sum(len(segment) for segment in segments)
    except (IndexError, TypeError):
        return None


def _deepseek_v4_outer_prefill_step(
    instance: Any,
    *,
    long_after: int,
    kernel_step: int,
) -> int:
    """Choose the DS4 outer scheduler slice for this exact batch turn.

    Long DS4 prompts intentionally use 1024-token model calls: that width is
    substantially faster at high context than the 2048-token route.  MLX-LM's
    outer scheduler normally hands ``PromptProcessingBatch.prompt`` 2048 (or
    4096) tokens, however, so the adaptive prompt loop executes two (or four)
    expensive 1024-token calls before returning control to an active decode.

    Cap only that *outer* slice to one kernel call when this turn contains both
    a decode and long-prompt prefill work.  Idle long prompts retain the wider
    outer slice and allocator reuse.  The decision uses only mirrored batch
    structure and full request lengths -- never local clocks -- so TP ranks
    stay in collective lockstep.
    """

    try:
        configured = max(1, int(instance.prefill_step_size))
        kernel_step = max(1, int(kernel_step))
        long_after = max(0, int(long_after))
    except (AttributeError, TypeError, ValueError):
        return 1
    if configured <= kernel_step:
        return configured

    generation_batch = getattr(instance, "_generation_batch", None)
    generation_rows = _safe_batch_len(generation_batch)
    try:
        completion_limit = max(1, int(instance.completion_batch_size))
    except (AttributeError, TypeError, ValueError):
        completion_limit = max(1, generation_rows + 1)

    # Pinned MLX-LM returns after the generation step when this batch is full,
    # so no prefill slice runs in that turn and there is nothing to cap.
    if generation_rows >= completion_limit:
        return configured

    prompt_batch = getattr(instance, "_prompt_batch", None)
    prompt_uids = list(getattr(prompt_batch, "uids", ()) or ())
    current = list(getattr(instance, "_currently_processing", ()) or ())
    will_decode = generation_rows > 0
    has_long_prefill = False

    for index, uid in enumerate(prompt_uids):
        if index >= len(current):
            total = _known_prompt_total(instance, uid)
            has_long_prefill |= total is not None and total > long_after
            continue
        staged = current[index]
        try:
            segments = staged[0]
        except (IndexError, TypeError):
            segments = ()
        if _at_generation_boundary(segments):
            # BatchGenerator promotes this row before processing prompt rows,
            # so another long row in the same turn already contends with a
            # latency-sensitive decode even if generation_batch is empty now.
            will_decode = True
            continue
        total = _current_prompt_total(instance, uid, index, staged)
        has_long_prefill |= total is not None and total > long_after

    pending = getattr(instance, "_unprocessed_sequences", ()) or ()
    try:
        prefill_limit = max(0, int(instance.prefill_batch_size))
    except (AttributeError, TypeError, ValueError):
        prefill_limit = len(prompt_uids)
    pending_slots = max(0, prefill_limit - len(prompt_uids))
    pending_slots = min(
        pending_slots,
        max(0, completion_limit - generation_rows),
    )
    for staged in islice(iter(pending), pending_slots):
        try:
            segments = staged[1]
        except (IndexError, TypeError):
            segments = ()
        if _at_generation_boundary(segments):
            will_decode = True
            continue
        total = _pending_prompt_total(instance, staged)
        has_long_prefill |= total is not None and total > long_after

    if will_decode and has_long_prefill:
        return min(configured, kernel_step)
    return configured


def _supports_coordinator_sampling(
    pipeline_model: Any,
    *,
    batchable: bool,
    world_size: int,
) -> tuple[bool, str]:
    if world_size < 2:
        return False, "requires more than one pipeline rank"
    if not batchable:
        return False, "model is not compatible with MLX-LM continuous batching"
    call = type(pipeline_model).__dict__.get("__call__")
    if not callable(call):
        return False, "pipeline model has no callable forward path"
    try:
        source = inspect.getsource(call)
    except (OSError, TypeError):
        return False, "pipeline forward source is unavailable for validation"
    required = (
        "pipeline_rank",
        "pipeline_size",
        "distributed.all_gather",
        "distributed.send",
    )
    if any(token not in source for token in required):
        return False, "pipeline forward does not match the validated output contract"
    if source.count("distributed.all_gather") != 1:
        return False, "pipeline forward has an ambiguous collective output path"
    if source.count("distributed.send") != 1:
        return False, "pipeline forward has an ambiguous send path"
    return True, "validated final hidden-state gather replaced by token all-sum"


def _supports_native_async_step(generation_batch: Any) -> bool:
    step = getattr(generation_batch, "_step", None)
    try:
        source = inspect.getsource(step)
    except (OSError, TypeError):
        return False
    return "async_eval" in source and "_next_tokens" in source


def _supports_rank_zero_logits(model: Any) -> tuple[bool, int, str]:
    """Validate that worker ranks may advance the model without an LM head."""

    if not getattr(model, "_omlx_supports_rank_zero_logits", False):
        return False, 0, "model adapter has no rank-zero logits contract"
    call = type(model).__dict__.get("__call__")
    if not callable(call):
        return False, 0, "model adapter has no direct callable forward path"
    try:
        signature = inspect.signature(call)
    except (TypeError, ValueError):
        return False, 0, "model adapter forward signature is unavailable"
    if "skip_logits" not in signature.parameters:
        return False, 0, "model adapter does not explicitly accept skip_logits"
    try:
        vocab_size = int(model._omlx_output_vocab_size)
    except (AttributeError, TypeError, ValueError):
        return False, 0, "model adapter does not declare its output vocabulary"
    if vocab_size < 1:
        return False, 0, "model adapter declared an invalid output vocabulary"
    return (
        True,
        vocab_size,
        "worker ranks skip the vocabulary projection and log-softmax",
    )


def _supports_pipeline_prompt(prompt_batch: Any) -> tuple[bool, str]:
    """Validate the exact MLX-LM prompt loop this module replaces.

    This deliberately checks behavior-bearing source tokens instead of merely
    checking that a method with the right name exists. A future MLX-LM release
    can change cache preparation/finalization or prompt ownership without
    silently running a stale monkeypatch.
    """

    prompt = getattr(prompt_batch, "prompt", None)
    base_prompt = getattr(prompt_batch, "_omlx_base_prompt", prompt)
    known_wrapper = getattr(prompt_batch, "_omlx_prompt_wrapper", prompt)
    if prompt not in {base_prompt, known_wrapper}:
        return False, "another prompt-processing patch owns this model's prefill"
    try:
        source = inspect.getsource(base_prompt)
    except (OSError, TypeError):
        return False, "pinned prompt-processing source is unavailable"
    required = (
        "_right_pad_prompts",
        "self.prefill_step_size",
        "self.model(",
        "c.prepare(",
        "c.finalize()",
        "mx.eval([c.state for c in self.prompt_cache])",
    )
    if any(token not in source for token in required):
        return False, "MLX-LM prompt loop does not match the validated contract"
    return True, "validated staggered chunk scheduler and queued inter-stage sends"


def _supports_prefill_logits_skip(model: Any) -> tuple[bool, str]:
    """Prove the model explicitly accepts the no-output prefill contract."""

    call = getattr(model, "__call__", None)
    if not callable(call):
        return False, "model has no callable forward path"
    try:
        signature = inspect.signature(call)
    except (TypeError, ValueError):
        return False, "model forward signature is unavailable"
    if "skip_lm_head" not in signature.parameters:
        return False, "model does not explicitly accept skip_lm_head"
    return True, "prefill advances caches without projecting discarded logits"


def _indexer_row_parallel_capability(
    model: Any,
    *,
    world_size: int,
) -> tuple[bool, bool, str]:
    """Report the DS4 sparse-indexer row split installed by its TP adapter."""

    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, list):
        return False, False, "model has no sparse-indexer row-parallel contract"
    indexers = [
        indexer
        for layer in layers
        if layer is not None
        for indexer in (getattr(getattr(layer, "attn", None), "indexer", None),)
        if indexer is not None
    ]
    if not indexers:
        return False, False, "model has no sparse prefill indexer"
    enabled = os.environ.get("OMLX_DSV4_INDEXER_ROW_TP", "1").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }
    installed = world_size > 1 and all(
        getattr(indexer, "row_sharding_group", None) is not None
        for indexer in indexers
    )
    try:
        threshold = max(
            0,
            int(os.environ.get("OMLX_DSV4_INDEXER_ROW_TP_MIN_POOL", "2048")),
        )
    except ValueError:
        threshold = 2048
    active = enabled and installed
    return (
        enabled,
        active,
        (
            f"{len(indexers)} sparse indexers split prompt rows after "
            f"{threshold} pooled entries and gather only top-k indices"
            if active
            else (
                "sparse-indexer row parallelism is disabled by the operator"
                if not enabled
                else "sparse-indexer row parallelism requires tensor sharding"
            )
        ),
    )


def _deepseek_v4_fused_decode_capability(
    model: Any,
) -> tuple[bool, bool, str]:
    """Report the non-MTP DS4 fused-attention decode contract."""

    layers = getattr(getattr(model, "model", None), "layers", None)
    attention_modules = [
        attention
        for layer in (layers if isinstance(layers, list) else ())
        for attention in (getattr(layer, "attn", None),)
        if attention is not None and getattr(attention, "dspark", False)
    ]
    forced_exact = os.environ.get(
        "OMLX_DSV4_EXACT_DECODE", "0"
    ).strip().lower() in {"1", "true", "on"}
    enabled = not forced_exact
    active = bool(
        attention_modules
        and enabled
        and not any(
            getattr(attention, "_omlx_decode_consistent", False)
            for attention in attention_modules
        )
    )
    if not attention_modules:
        reason = "model has no DeepSeek V4 DSpark attention modules"
    elif forced_exact:
        reason = "OMLX_DSV4_EXACT_DECODE forces the legacy exact decode path"
    elif not active:
        reason = "MTP requires bit-identical one-token decode and batched verification"
    else:
        reason = (
            "non-MTP decode uses fused SDPA while sparse ratio-4 layers retain "
            "their fused exact kernel"
        )
    return enabled, active, reason


def _supports_vocab_parallel_sampling(
    model: Any,
    *,
    batchable: bool,
    world_size: int,
) -> tuple[bool, Any | None, int, str]:
    """Validate local vocabulary shards for coordinator-only reconstruction."""

    if world_size < 2:
        return False, None, 0, "requires more than one tensor rank"
    if not batchable:
        return False, None, 0, "model is not compatible with continuous batching"
    candidates = (
        model,
        getattr(model, "language_model", None),
        getattr(model, "_language_model", None),
    )
    mtp_model = next(
        (
            candidate
            for candidate in candidates
            if candidate is not None
            and bool(getattr(candidate, "_omlx_mtp_decode_enabled", False))
        ),
        None,
    )
    if mtp_model is not None and not bool(
        getattr(mtp_model, "_omlx_distributed_mtp_vocab_ready", False)
    ):
        return (
            False,
            None,
            0,
            "distributed MTP requires matching local shards for every "
            "vocabulary projection",
        )
    if mtp_model is not None:
        auxiliary_heads = tuple(
            getattr(mtp_model, "_omlx_vocab_parallel_aux_heads", ()) or ()
        )
        if not auxiliary_heads or any(
            not getattr(head, "_omlx_vocab_parallel", False)
            for head in auxiliary_heads
        ):
            return (
                False,
                None,
                0,
                "distributed MTP auxiliary vocabulary shards are incomplete",
            )
    if any(
        bool(getattr(candidate, "_omlx_mtp_decode_enabled", False))
        for candidate in candidates
        if candidate is not None
    ) and not bool(getattr(mtp_model, "_omlx_mtp_chain", False)):
        return False, None, 0, "distributed vocabulary MTP requires chain verification"
    head = next(
        (
            getattr(candidate, "lm_head", None)
            for candidate in candidates
            if candidate is not None
            and getattr(getattr(candidate, "lm_head", None), "_omlx_vocab_parallel", False)
        ),
        None,
    )
    if head is None:
        return False, None, 0, "model has no validated vocabulary-parallel head"
    try:
        output_dims = int(head._omlx_output_dims)
        local_dims = int(head.weight.shape[0])
    except (AttributeError, TypeError, ValueError):
        return False, None, 0, "vocabulary-parallel head dimensions are unavailable"
    if output_dims < 1 or local_dims * world_size != output_dims:
        return False, None, 0, "vocabulary-parallel head shards are incomplete"
    reason = (
        "rank zero reconstructs MTP vocabulary shards and broadcasts only "
        "draft/verification decisions"
        if mtp_model is not None
        else "rank zero reconstructs vocabulary shards and broadcasts only sampled token IDs"
    )
    return (
        True,
        head,
        output_dims,
        reason,
    )


class _MTPVocabCoordinator:
    """Point-to-point vocab reconstruction plus tiny decision collectives."""

    def __init__(self, mx: Any, group: Any, output_size: int) -> None:
        self.mx = mx
        self.group = group
        self.rank = int(group.rank())
        self.world_size = int(group.size())
        self.output_size = int(output_size)
        self.is_coordinator = self.rank == 0

    def gather_logits(self, local_logits: Any) -> Any | None:
        if self.is_coordinator:
            parts = [local_logits]
            for source in range(1, self.world_size):
                parts.append(self.mx.distributed.recv_like(local_logits, source))
            return self.mx.concatenate(parts, axis=-1)
        sent = self.mx.distributed.send(local_logits, 0)
        # Enter the following token/decision collective only after the local
        # projection is on the wire. This keeps JACCL operation order stable
        # even when one Apple Silicon rank is substantially faster.
        self.mx.eval(sent)
        return None

    def _broadcast_int32(self, value: Any | None, shape: tuple[int, ...]) -> Any:
        """Broadcast one tiny rank-zero decision without a JACCL reduction.

        These values are decisions, not partial results. Expressing a broadcast
        as ``all_sum(rank0_value, worker_zeros)`` needlessly routes them through
        JACCL's reduction kernel. More importantly, repeated 1--7 element
        reductions can return an uninitialised high-bit value on one peer under
        sustained MTP traffic even though the large tensor collectives remain
        healthy. A chained point-to-point send is both less work and matches
        the actual protocol: rank zero owns the decision and every worker
        consumes that exact value.

        Keep the send result in the returned lazy graph. This avoids adding a
        host synchronization to the decode loop while ensuring every send is
        evaluated before a caller can consume the decision.
        """

        if self.is_coordinator:
            if value is None or tuple(value.shape) != shape:
                raise RuntimeError("MTP coordinator produced an invalid decision")
            broadcast = value.astype(self.mx.int32)
            for target in range(1, self.world_size):
                broadcast = self.mx.distributed.send(broadcast, target)
            return broadcast

        template = self.mx.zeros(shape, dtype=self.mx.int32)
        return self.mx.distributed.recv_like(template, 0)

    def sync_tokens(self, proposal: Any | None, shape: tuple[int, ...]) -> Any:
        if self.is_coordinator:
            if proposal is None:
                raise RuntimeError("MTP coordinator produced no token proposal")
            value = proposal
        else:
            value = None
        return self._broadcast_int32(value, shape).astype(self.mx.uint32)

    def sync_packet(self, packet: Any | None, length: int) -> list[int]:
        value = self._broadcast_int32(packet, (length,))
        return [int(item) for item in value.tolist()]

    def sync_scalar(self, value: int) -> int:
        packet = (
            self.mx.array([int(value)], dtype=self.mx.int32)
            if self.is_coordinator
            else None
        )
        packet = self._broadcast_int32(packet, (1,))
        return int(packet.item())


@dataclass(frozen=True)
class PrefillSlot:
    """One rank's work in a fill/drain pipeline timeline."""

    iteration: int
    start: int | None
    end: int | None

    @property
    def is_real(self) -> bool:
        return self.start is not None


def pipeline_prefill_schedule(
    token_count: int,
    prefill_step_size: int,
    *,
    rank: int,
    world_size: int,
) -> tuple[PrefillSlot, ...]:
    """Return the Exo-style staggered fill/steady/drain schedule for one rank.

    Dummy slots do not issue a collective. Pipeline ``recv``/``send`` calls
    provide the actual dependency between adjacent ranks; issuing a different
    collective in a dummy slot would reorder the distributed graph and can
    deadlock. The slots remain explicit so telemetry and tests can prove every
    rank has the same total timeline and the expected offset.
    """

    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    if prefill_step_size < 1:
        raise ValueError("prefill_step_size must be positive")
    if world_size < 2:
        raise ValueError("pipeline prefill requires at least two ranks")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be inside the pipeline world")

    chunks = int(math.ceil(token_count / prefill_step_size))
    slots: list[PrefillSlot] = []
    total = chunks + world_size - 1
    # MLX-LM's pipeline flows from the highest rank to rank zero: rank r
    # receives from r+1 and sends to r-1. Exo's native pipeline numbers the
    # source stage as rank zero, so its ``rank`` leading-dummy formula must be
    # mirrored here. Using it verbatim makes rank zero block in recv while the
    # highest rank is still in a dummy slot.
    leading = world_size - 1 - rank
    for iteration in range(total):
        chunk = iteration - leading
        if 0 <= chunk < chunks:
            start = chunk * prefill_step_size
            slots.append(
                PrefillSlot(
                    iteration=iteration,
                    start=start,
                    end=min(start + prefill_step_size, token_count),
                )
            )
        else:
            slots.append(PrefillSlot(iteration=iteration, start=None, end=None))
    return tuple(slots)


@contextmanager
def install_runtime_optimizations(
    model: Any,
    group: Any,
    execution: ExecutionSettings,
    *,
    batchable: bool,
    pipeline_parallel: bool = True,
) -> Iterator[dict[str, dict[str, Any]]]:
    """Install opt-in token-only output while reporting every capability."""

    import mlx.core as mx

    mlx_generate = importlib.import_module("mlx_lm.generate")

    pipeline_model = getattr(model, "model", None)
    world_size = int(group.size())
    vocab_sampling_supported = False
    vocab_parallel_head = None
    vocab_output_size = 0
    vocab_sampling_reason = "requires pure tensor parallelism"
    if pipeline_parallel:
        sampling_supported, sampling_reason = _supports_coordinator_sampling(
            pipeline_model,
            batchable=batchable,
            world_size=world_size,
        )
    else:
        (
            vocab_sampling_supported,
            vocab_parallel_head,
            vocab_output_size,
            vocab_sampling_reason,
        ) = _supports_vocab_parallel_sampling(
            model,
            batchable=batchable,
            world_size=world_size,
        )
        sampling_supported = vocab_sampling_supported
        sampling_reason = vocab_sampling_reason
    generation_batch_cls = getattr(mlx_generate, "GenerationBatch", None)
    prompt_batch_cls = getattr(mlx_generate, "PromptProcessingBatch", None)
    native_async = (
        _supports_native_async_step(generation_batch_cls)
        if generation_batch_cls
        else False
    )
    prompt_supported, prompt_reason = (
        _supports_pipeline_prompt(prompt_batch_cls)
        if prompt_batch_cls
        else (False, "MLX-LM has no prompt-processing batch")
    )
    skip_logits_supported, skip_logits_reason = _supports_prefill_logits_skip(model)
    skip_logits_active = prompt_supported and skip_logits_supported
    prefill_clear_every = _prefill_clear_cache_every()
    prefill_async_depth = _deepseek_v4_prefill_async_depth()
    (
        indexer_row_parallel_enabled,
        indexer_row_parallel_active,
        indexer_row_parallel_reason,
    ) = _indexer_row_parallel_capability(model, world_size=world_size)
    (
        fused_decode_enabled,
        fused_decode_active,
        fused_decode_reason,
    ) = _deepseek_v4_fused_decode_capability(model)
    (
        adaptive_prefill_enabled,
        adaptive_prefill_active,
        adaptive_prefill_after,
        adaptive_prefill_step,
        adaptive_prefill_base,
        adaptive_prefill_reason,
    ) = _deepseek_v4_adaptive_prefill(
        model,
        execution,
        pipeline_parallel=pipeline_parallel,
    )
    raw_indexer_owner = os.environ.get(
        "OMLX_DSV4_INDEXER_DECODE_OWNER_RANK", ""
    ).strip().lower()
    try:
        indexer_owner_rank = int(raw_indexer_owner)
    except ValueError:
        indexer_owner_rank = -1
    indexer_present = not indexer_row_parallel_reason.startswith("model has no")
    indexer_owner_active = bool(
        not pipeline_parallel
        and indexer_present
        and 0 <= indexer_owner_rank < world_size
    )
    try:
        from omlx.custom_kernels.glm_moe_dsa import fast as _indexer_fast

        native_fp32_topk = _indexer_fast.has_symbol("dspark_fp32_topk_indices")
    except Exception:
        native_fp32_topk = False
    raw_shard_weights = os.environ.get("OMLX_TP_SHARD_WEIGHTS", "")
    try:
        tensor_shard_weights = tuple(
            int(item.strip()) for item in raw_shard_weights.split(",") if item.strip()
        )
    except ValueError:
        tensor_shard_weights = ()
    asymmetric_tensor_active = bool(
        not pipeline_parallel
        and len(tensor_shard_weights) == world_size
        and all(weight > 0 for weight in tensor_shard_weights)
        and len(set(tensor_shard_weights)) > 1
    )
    local_shard_rank = int(group.rank())
    asymmetric_tensor_reason = (
        f"rank {local_shard_rank} holds "
        f"{tensor_shard_weights[local_shard_rank]}/{sum(tensor_shard_weights)} "
        "of each adapter-declared tensor segment"
        if asymmetric_tensor_active
        else "the signed tensor partition is equal across ranks"
    )
    sampling_active = execution.sampling_rank_only and sampling_supported
    vocab_sampling_active = sampling_active and vocab_sampling_supported
    (
        rank_zero_logits_supported,
        output_vocab_size,
        rank_zero_logits_reason,
    ) = _supports_rank_zero_logits(model)
    placeholder_vocab_size = (
        vocab_output_size if vocab_sampling_active else output_vocab_size
    )
    rank_zero_logits_active = (
        sampling_active
        and pipeline_parallel
        and rank_zero_logits_supported
    )
    prefill_active = (
        execution.async_overlap
        and sampling_active
        and prompt_supported
        and pipeline_parallel
        and execution.prefill_step_size > 1
    )
    outer_prefill_yield_enabled = os.environ.get(
        _DSV4_PREFILL_YIELD_ENV, "1"
    ).strip().lower() in {"1", "true", "on", "yes"}
    outer_prefill_yield_active = bool(
        outer_prefill_yield_enabled
        and adaptive_prefill_active
        and skip_logits_active
        and callable(getattr(mlx_generate.BatchGenerator, "next", None))
    )
    dsv4_module = sys.modules.get("mlx_lm.models.deepseek_v4")
    defer_dsv4_cache_materialization = getattr(
        dsv4_module,
        "_defer_cache_materialization",
        None,
    )
    mtp_decode_enabled = _mtp_decode_enabled(model)
    prefill_async_active = bool(
        prefill_async_depth == 2
        and execution.async_overlap
        and not pipeline_parallel
        and world_size == 2
        and adaptive_prefill_active
        and adaptive_prefill_step == 1024
        and skip_logits_active
        and not mtp_decode_enabled
        and prefill_clear_every == 0
        and callable(defer_dsv4_cache_materialization)
    )
    if prefill_async_active:
        prefill_async_reason = (
            "pure TP2 may queue two lossless 1K DS4 cache graphs for one "
            "long unpadded row; one-chunk live-decode fairness slices stay "
            "synchronous"
        )
    elif prefill_async_depth == 0:
        prefill_async_reason = (
            "DS4 TP prefill graph overlap is disabled by the operator"
        )
    elif not execution.async_overlap:
        prefill_async_reason = "execution async overlap is disabled"
    elif pipeline_parallel or world_size != 2:
        prefill_async_reason = "depth-two DS4 prefill overlap requires pure TP2"
    elif not adaptive_prefill_active:
        prefill_async_reason = adaptive_prefill_reason
    elif adaptive_prefill_step != 1024:
        prefill_async_reason = (
            "depth-two DS4 prefill overlap requires 1024-token chunks"
        )
    elif not skip_logits_active:
        prefill_async_reason = skip_logits_reason
    elif mtp_decode_enabled:
        prefill_async_reason = "depth-two DS4 prefill overlap is not validated with MTP"
    elif prefill_clear_every != 0:
        prefill_async_reason = (
            "depth-two DS4 prefill overlap requires clear-cache cadence 0"
        )
    else:
        prefill_async_reason = (
            "DeepSeek V4 cache-materialization capture is unavailable"
        )
    batching_enabled = execution.pipeline_microbatch_size > 1
    batching_active = batching_enabled and batchable
    capabilities = {
        "coalesced_batching": _capability(
            enabled=batching_enabled,
            active=batching_active,
            reason=(
                "MLX-LM continuous batching coalesces up to "
                f"{execution.pipeline_microbatch_size} requests per target batch"
                if batchable
                else (
                    "this model's KV cache cannot be merged, so MLX-LM serves "
                    "requests sequentially"
                )
            ),
        ),
        "sampling_rank_only": _capability(
            enabled=execution.sampling_rank_only,
            active=sampling_active,
            reason=(
                sampling_reason
                if execution.sampling_rank_only
                else "experimental optimization is disabled"
            ),
        ),
        "rank_zero_logits": _capability(
            enabled=execution.sampling_rank_only,
            active=rank_zero_logits_active,
            reason=rank_zero_logits_reason,
        ),
        "async_overlap": _capability(
            enabled=execution.async_overlap,
            active=execution.async_overlap and native_async,
            reason=(
                "pinned MLX-LM GenerationBatch dispatches the next token with "
                "mx.async_eval"
                if native_async
                else "pinned generation step has no validated async dispatch"
            ),
        ),
        "cache_affinity": _capability(
            enabled=execution.cache_affinity,
            active=execution.cache_affinity,
            reason=(
                "all requests for this model stay on one persistent deployment "
                "and its rank-local prompt caches"
                if execution.cache_affinity
                else "deployment cache affinity is disabled"
            ),
        ),
        "pipeline_prefill_overlap": _capability(
            enabled=execution.async_overlap and execution.sampling_rank_only,
            active=prefill_active,
            reason=(
                prompt_reason
                if prefill_active
                else (
                    prompt_reason
                    if sampling_active
                    else (
                        "requires the validated rank-zero sampling path; this model "
                        "keeps MLX-LM's synchronized prefill"
                    )
                )
            ),
        ),
        "prefill_logits_skip": _capability(
            enabled=True,
            active=skip_logits_active,
            reason=(
                skip_logits_reason
                if prompt_supported
                else prompt_reason
            ),
        ),
        "prefill_allocator_reuse": _capability(
            enabled=prefill_clear_every != 1,
            active=(
                (prefill_active or skip_logits_active)
                and prefill_clear_every != 1
            ),
            reason=(
                "same-shaped prefill buffers are reused until the prompt ends"
                if prefill_clear_every == 0
                else (
                    "the MLX allocator cache is cleared every "
                    f"{prefill_clear_every} chunks and at prompt end"
                    if prefill_clear_every > 1
                    else "the MLX allocator cache is cleared after every chunk"
                )
            ),
        ),
        "vocab_parallel_sampling": _capability(
            enabled=execution.sampling_rank_only,
            active=vocab_sampling_active,
            reason=vocab_sampling_reason,
        ),
        "sparse_indexer_row_parallel": _capability(
            enabled=indexer_row_parallel_enabled,
            active=indexer_row_parallel_active,
            reason=indexer_row_parallel_reason,
        ),
        "sparse_indexer_decode_owner": _capability(
            enabled=raw_indexer_owner not in {"", "off", "false", "disabled"},
            active=indexer_owner_active,
            reason=(
                f"rank {indexer_owner_rank} computes decode top-k once and "
                "broadcasts the compact index vector"
                if indexer_owner_active
                else "decode indexer ownership requires DS4 tensor sharding"
            ),
        ),
        "sparse_indexer_native_topk": _capability(
            enabled=True,
            active=indexer_present and native_fp32_topk,
            reason=(
                "deterministic native FP32 top-k serves decode and verification"
                if native_fp32_topk
                else "native FP32 top-k symbol is unavailable; stable MLX fallback"
            ),
        ),
        "asymmetric_tensor_parallel": _capability(
            enabled=not pipeline_parallel,
            active=asymmetric_tensor_active,
            reason=asymmetric_tensor_reason,
        ),
        "deepseek_v4_fused_decode_attention": _capability(
            enabled=fused_decode_enabled,
            active=fused_decode_active,
            reason=fused_decode_reason,
        ),
        "deepseek_v4_adaptive_prefill": _capability(
            enabled=adaptive_prefill_enabled,
            active=adaptive_prefill_active,
            reason=adaptive_prefill_reason,
        ),
        "deepseek_v4_prefill_yield": _capability(
            enabled=outer_prefill_yield_enabled,
            active=outer_prefill_yield_active,
            reason=(
                f"long DS4 outer slices yield after one "
                f"{adaptive_prefill_step}-token model call while decode is live"
                if outer_prefill_yield_active
                else (
                    "DS4 contended-prefill yielding is disabled by the operator"
                    if not outer_prefill_yield_enabled
                    else (
                        adaptive_prefill_reason
                        if not adaptive_prefill_active
                        else prompt_reason
                    )
                )
            ),
        ),
        "deepseek_v4_prefill_async": _capability(
            enabled=prefill_async_depth == 2,
            active=prefill_async_active,
            reason=prefill_async_reason,
        ),
    }
    # A long DS4 request repeatedly executes this exact inner model-call
    # width.  Publish it as machine-readable launch metadata so the rank
    # worker can compile the shape after every rank has loaded, but before the
    # HTTP listener accepts a user request.  Keeping the warmup outside this
    # context manager's setup is deliberate: the first rank to finish loading
    # must not enter a collective while a peer is still materializing weights.
    shape_warmup_enabled = os.environ.get(
        _CLUSTER_PREFILL_SHAPE_WARMUP_ENV, "1"
    ).strip().lower() in {"1", "true", "on", "yes"}
    capabilities["deepseek_v4_adaptive_prefill"]["shape_warmup_tokens"] = (
        adaptive_prefill_step
        if shape_warmup_enabled and adaptive_prefill_active and skip_logits_active
        else 0
    )
    if not sampling_active and not skip_logits_active:
        yield capabilities
        return

    original_all_gather = mx.distributed.all_gather
    original_send = mx.distributed.send
    original_pipeline_call = type(pipeline_model).__call__
    original_generation_step = mlx_generate.GenerationBatch._step
    original_batch_next = (
        mlx_generate.BatchGenerator.next if outer_prefill_yield_active else None
    )
    original_prompt = (
        mlx_generate.PromptProcessingBatch.prompt
        if prefill_active or skip_logits_active
        else None
    )
    previous_vocab_gather = (
        getattr(vocab_parallel_head, "_omlx_gather_vocab_logits", True)
        if vocab_sampling_active
        else None
    )
    mtp_vocab_model = next(
        (
            candidate
            for candidate in (
                model,
                getattr(model, "language_model", None),
                getattr(model, "_language_model", None),
            )
            if candidate is not None
            and bool(getattr(candidate, "_omlx_mtp_decode_enabled", False))
        ),
        None,
    )
    mtp_auxiliary_heads = tuple(
        getattr(mtp_vocab_model, "_omlx_vocab_parallel_aux_heads", ()) or ()
    )
    previous_auxiliary_gathers = tuple(
        getattr(head, "_omlx_gather_vocab_logits", True)
        for head in mtp_auxiliary_heads
    )
    mtp_vocab_active = bool(
        vocab_sampling_active
        and any(
            bool(getattr(candidate, "_omlx_mtp_decode_enabled", False))
            for candidate in (
                model,
                getattr(model, "language_model", None),
                getattr(model, "_language_model", None),
            )
            if candidate is not None
        )
    )
    missing_mtp_coordinator = object()
    previous_mtp_coordinator = getattr(
        model, "_omlx_mtp_vocab_coordinator", missing_mtp_coordinator
    )
    mtp_vocab_coordinator = (
        _MTPVocabCoordinator(mx, group, vocab_output_size)
        if mtp_vocab_active
        else None
    )
    local_state = threading.local()

    def selective_all_gather(value: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(local_state, "skip_final_gather", False):
            return value
        return original_all_gather(value, *args, **kwargs)

    def local_pipeline_output(
        instance: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        previous = getattr(local_state, "skip_final_gather", False)
        local_state.skip_final_gather = True
        try:
            return original_pipeline_call(instance, *args, **kwargs)
        finally:
            local_state.skip_final_gather = previous

    def queued_pipeline_send(value: Any, *args: Any, **kwargs: Any) -> Any:
        """Materialize a stage output and defer only the transport operation."""

        if not getattr(local_state, "queue_prefill_sends", False):
            return original_send(value, *args, **kwargs)
        # Breaking the graph here is essential. Without it, the send remains
        # entangled with the entire layer graph and the downstream recv cannot
        # make forward progress while this rank starts its next chunk.
        mx.eval(value)
        pending = getattr(local_state, "pending_prefill_sends", None)
        if pending is None:
            pending = []
            local_state.pending_prefill_sends = pending
        pending.append((value, args, kwargs))
        return value

    def flush_prefill_sends() -> None:
        pending = getattr(local_state, "pending_prefill_sends", [])
        local_state.pending_prefill_sends = []
        for value, args, kwargs in pending:
            sent = original_send(value, *args, **kwargs)
            mx.async_eval(sent)

    def finish_prefill_chunk(chunk_index: int, *, final: bool) -> None:
        """Retain reusable scratch between chunks, then release it at the end."""

        if final or (
            prefill_clear_every > 0
            and chunk_index % prefill_clear_every == 0
        ):
            mx.clear_cache()

    def staggered_pipeline_prompt(instance: Any, tokens: list[list[int]]) -> None:
        """Pinned PromptProcessingBatch.prompt with pipeline fill/drain."""

        if len(instance.uids) != len(tokens):
            raise ValueError("The batch length doesn't match the number of inputs")
        if not tokens:
            return
        before_prompt = getattr(instance, "_omlx_before_prompt", None)
        if callable(before_prompt):
            before_prompt()

        for stored, incoming in zip(instance.tokens, tokens):
            stored += incoming

        lengths = [len(prompt) for prompt in tokens]
        max_length = max(lengths)
        padding = [max_length - length for length in lengths]
        max_padding = max(padding)
        if max_padding > 0:
            tokens_array = mlx_generate._right_pad_prompts(
                tokens,
                max_length=max_length,
            )
            for cache in instance.prompt_cache:
                cache.prepare(lengths=lengths, right_padding=padding)
        else:
            tokens_array = mx.array(tokens)

        # ``prefill_step_size`` is already the memory-admitted chunk size used
        # by MLX-LM and the rank prefill guard. Dividing it by the world size
        # here made a two-rank 4096-token deployment execute 2048-token chunks,
        # doubling every cache-state barrier and send boundary on long prompts.
        #
        # Staggering still overlaps adjacent stages: it is the rank offset in
        # ``pipeline_prefill_schedule`` that creates fill/steady/drain, not a
        # private reduction of the guarded compute chunk.
        step = max(1, int(instance.prefill_step_size))
        schedule = pipeline_prefill_schedule(
            int(tokens_array.shape[1]),
            step,
            rank=int(group.rank()),
            world_size=world_size,
        )
        real_chunks = sum(slot.is_real for slot in schedule)
        chunk_index = 0
        local_state.pending_prefill_sends = []
        try:
            for slot in schedule:
                if not slot.is_real:
                    continue
                chunk_index += 1
                local_state.queue_prefill_sends = True
                try:
                    model_kwargs = {"skip_lm_head": True} if skip_logits_active else {}
                    instance.model(
                        tokens_array[:, slot.start : slot.end],
                        cache=instance.prompt_cache,
                        **model_kwargs,
                    )
                finally:
                    local_state.queue_prefill_sends = False
                flush_prefill_sends()
                mx.eval([cache.state for cache in instance.prompt_cache])
                finish_prefill_chunk(
                    chunk_index,
                    final=chunk_index == real_chunks and max_padding == 0,
                )
        finally:
            local_state.queue_prefill_sends = False
            # A cancelled/failed prefill must never leak an old activation into
            # the next request.
            local_state.pending_prefill_sends = []

        if max_padding > 0:
            for cache in instance.prompt_cache:
                cache.finalize()
            mx.eval([cache.state for cache in instance.prompt_cache])
            finish_prefill_chunk(chunk_index, final=True)

    def skip_prefill_logits_prompt(instance: Any, tokens: list[list[int]]) -> None:
        """Pinned prompt loop that omits its otherwise-discarded LM head."""

        if len(instance.uids) != len(tokens):
            raise ValueError("The batch length doesn't match the number of inputs")
        if not tokens:
            return
        before_prompt = getattr(instance, "_omlx_before_prompt", None)
        if callable(before_prompt):
            before_prompt()

        # ``PromptProcessingBatch.prompt`` is called once per outer MLX-LM
        # chunk, not once per complete request. The accumulated token history
        # is therefore the persistent context position for the adaptive switch
        # (and already includes a restored prefix-cache key when present).
        processed_tokens = min(
            (len(stored) for stored in instance.tokens),
            default=0,
        )
        for stored, incoming in zip(instance.tokens, tokens):
            stored += incoming

        lengths = [len(prompt) for prompt in tokens]
        max_length = max(lengths)
        padding = [max_length - length for length in lengths]
        max_padding = max(padding)
        if max_padding > 0:
            tokens_array = mlx_generate._right_pad_prompts(
                tokens,
                max_length=max_length,
            )
            for cache in instance.prompt_cache:
                cache.prepare(lengths=lengths, right_padding=padding)
        else:
            tokens_array = mx.array(tokens)

        chunk_index = 0
        prompt_lengths = getattr(instance, "_omlx_total_prompt_lengths", {})
        long_request = bool(
            adaptive_prefill_active
            and isinstance(prompt_lengths, dict)
            and any(
                int(prompt_lengths.get(uid, 0) or 0) > adaptive_prefill_after
                for uid in instance.uids
            )
        )
        async_this_prompt = bool(
            prefill_async_active
            and len(tokens) == 1
            and max_padding == 0
            and long_request
            and math.ceil(
                int(tokens_array.shape[1]) / adaptive_prefill_step
            )
            >= 2
        )
        pending_async_cache_arrays: tuple[Any, ...] | None = None

        def drain_async_cache_arrays() -> None:
            nonlocal pending_async_cache_arrays
            pending = pending_async_cache_arrays
            pending_async_cache_arrays = None
            if pending:
                mx.eval(*pending)

        try:
            while tokens_array.shape[1] > 0:
                chunk_index += 1
                if adaptive_prefill_active:
                    if long_request:
                        width = adaptive_prefill_step
                    elif processed_tokens < adaptive_prefill_after:
                        width = min(
                            adaptive_prefill_base,
                            adaptive_prefill_after - processed_tokens,
                        )
                    else:
                        width = adaptive_prefill_step
                else:
                    width = instance.prefill_step_size
                width = min(width, tokens_array.shape[1])

                if async_this_prompt:
                    captured_cache_arrays: list[Any] = []
                    try:
                        with defer_dsv4_cache_materialization() as captured:
                            captured_cache_arrays = captured
                            instance.model(
                                tokens_array[:, :width],
                                cache=instance.prompt_cache,
                                skip_lm_head=True,
                            )
                    except BaseException:
                        # If the forward failed after updating its cache, make
                        # both the previous graph and that partial current
                        # graph safe, in dependency order, before control
                        # escapes the prompt loop.
                        drain_async_cache_arrays()
                        if captured_cache_arrays:
                            mx.eval(*captured_cache_arrays)
                        raise
                    current_cache_arrays = tuple(captured_cache_arrays)
                    if not current_cache_arrays:
                        raise RuntimeError(
                            "DeepSeek V4 async prefill captured no cache arrays"
                        )
                    previous_cache_arrays = pending_async_cache_arrays
                    pending_async_cache_arrays = current_cache_arrays
                    # Keep at most the previous and current 1K graphs alive:
                    # queue the current work first, then complete the prior
                    # graph to expose command-buffer overlap without allowing
                    # an unbounded lazy cache chain.
                    mx.async_eval(*current_cache_arrays)
                    if previous_cache_arrays:
                        mx.eval(*previous_cache_arrays)
                else:
                    step_started = time.perf_counter()
                    instance.model(
                        tokens_array[:, :width],
                        cache=instance.prompt_cache,
                        skip_lm_head=True,
                    )
                    mx.eval([cache.state for cache in instance.prompt_cache])
                    if os.environ.get(
                        _DSV4_PREFILL_STEP_TRACE_ENV, "0"
                    ).strip().lower() in {"1", "true", "on", "yes"}:
                        print(
                            "DSV4_PREFILL_STEP_TRACE "
                            f"rank={os.getenv('MLX_RANK', '?')} "
                            f"start={processed_tokens} width={width} "
                            f"wall={time.perf_counter() - step_started:.6f} "
                            f"active_gib={mx.get_active_memory() / 1024**3:.3f} "
                            f"cache_gib={mx.get_cache_memory() / 1024**3:.3f}",
                            flush=True,
                        )

                tokens_array = tokens_array[:, width:]
                processed_tokens += width
                final_chunk = tokens_array.shape[1] == 0 and max_padding == 0
                if async_this_prompt and final_chunk:
                    # ``finish_prefill_chunk`` may clear the allocator. Never
                    # let it race the final in-flight cache graph.
                    drain_async_cache_arrays()
                finish_prefill_chunk(chunk_index, final=final_chunk)
        finally:
            # Cancellation, a model exception, or an mx.eval failure must not
            # leak an in-flight cache graph into decode or the next request.
            drain_async_cache_arrays()

        if max_padding > 0:
            for cache in instance.prompt_cache:
                cache.finalize()
            mx.eval([cache.state for cache in instance.prompt_cache])
            finish_prefill_chunk(chunk_index, final=True)

    def yielding_batch_next(instance: Any, *args: Any, **kwargs: Any) -> Any:
        """Run one MLX-LM turn with a contention-only DS4 outer slice cap."""

        previous_step = instance.prefill_step_size
        effective_step = _deepseek_v4_outer_prefill_step(
            instance,
            long_after=adaptive_prefill_after,
            kernel_step=adaptive_prefill_step,
        )
        if effective_step >= int(previous_step):
            return original_batch_next(instance, *args, **kwargs)
        # Only the outer extraction width changes. PromptProcessingBatch keeps
        # its configured width, and the adaptive DS4 loop still owns the exact
        # 1024-token model-call schedule. The generation thread is the sole
        # owner of this BatchGenerator, so the temporary value cannot race.
        instance.prefill_step_size = effective_step
        try:
            return original_batch_next(instance, *args, **kwargs)
        finally:
            instance.prefill_step_size = previous_step

    def coordinator_generation_step(instance: Any) -> Any:
        """Pinned GenerationBatch._step with one token collective per batch."""

        instance._current_tokens = instance._next_tokens
        instance._current_logprobs = instance._next_logprobs
        inputs = instance._current_tokens
        coordinator = int(group.rank()) == 0

        if vocab_sampling_active:
            logits = instance.model(inputs[:, None], cache=instance.prompt_cache)
            logits = logits[:, -1, :]
            if coordinator:
                parts = [logits]
                for source in range(1, world_size):
                    parts.append(mx.distributed.recv_like(logits, source))
                logits = mx.concatenate(parts, axis=-1)
            else:
                # Materialize the point-to-point send before entering the
                # token all-sum, keeping the collective order identical on
                # heterogeneous ranks.
                sent = mx.distributed.send(logits, 0)
                mx.eval(sent)
                logits = None
        elif coordinator or not rank_zero_logits_active:
            logits = instance.model(inputs[:, None], cache=instance.prompt_cache)
            logits = logits[:, -1, :]
        else:
            instance.model(
                inputs[:, None],
                cache=instance.prompt_cache,
                skip_logits=True,
            )
            # The token all-sum must be issued after this rank's stage send.
            # MiniMax anchors that lazy send in its last KV cache entry, so
            # materializing the cache state both advances the cache and fixes
            # the distributed operation order without paying for an LM head.
            cache_states = [cache.state for cache in instance.prompt_cache]
            if not cache_states:
                raise RuntimeError(
                    "rank-zero logits requires a cache state to anchor the "
                    "worker-stage send"
                )
            mx.eval(cache_states)
            logits = None

        token_context = []
        if any(instance.logits_processors):
            token_context = [
                token_buffer.update_and_fetch(inputs[index : index + 1])
                for index, token_buffer in enumerate(instance._token_context)
            ]
            if logits is not None:
                processed_logits = []
                for index in range(len(instance.uids)):
                    sample_logits = logits[index : index + 1]
                    for processor in instance.logits_processors[index]:
                        sample_logits = processor(
                            token_context[index],
                            sample_logits,
                        )
                    processed_logits.append(sample_logits)
                logits = mx.concatenate(processed_logits, axis=0)

        if logits is not None:
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        else:
            # Worker ResponseGenerator instances still index this vector while
            # draining their private response queues. Its values never leave
            # the worker, but it must retain the real vocabulary width. A
            # broadcast scalar preserves arbitrary token indexing without
            # allocating and clearing a full float32 vocabulary per request on
            # every decode step (about 0.5 MiB/token for DS4 Flash).
            logprobs = mx.broadcast_to(
                mx.zeros((), dtype=mx.float32),
                (len(instance.uids), placeholder_vocab_size),
            )

        if coordinator:
            if any(instance.samplers):
                all_samples = []
                for index in range(len(instance.uids)):
                    sampler = instance.samplers[index] or instance.fallback_sampler
                    all_samples.append(sampler(logprobs[index : index + 1]))
                sampled = mx.concatenate(all_samples, axis=0)
            else:
                sampled = instance.fallback_sampler(logprobs)
            sampled = sampled.astype(mx.int32)
        else:
            sampled = mx.zeros((len(instance.uids),), dtype=mx.int32)

        # Pipeline decisions have one owner. A tiny JACCL int32 all_sum can
        # return an uninitialised high-bit value after the readiness canary
        # even while large activation collectives remain healthy. Broadcast
        # the rank-zero token explicitly instead. Materializing the worker's
        # stage output before recv keeps the activation send ahead of this
        # decision edge; the explicit eval also gives terminal one-token
        # requests a shared boundary before GenerationBatch filters the row.
        if pipeline_parallel:
            if coordinator:
                decision = sampled
                sends = []
                for target in range(1, world_size):
                    sends.append(
                        mx.distributed.send(
                        decision,
                        target,
                    )
                    )
                # A transport send's returned array is an execution handle,
                # not a guaranteed alias of its input on every JACCL build.
                # Force the sends, but keep rank zero's original sampled value
                # as its local next token.
                if sends:
                    mx.eval(*sends)
                synchronized = decision
            else:
                if logits is not None:
                    mx.eval(logits)
                synchronized = mx.distributed.recv_like(
                    mx.zeros(sampled.shape, dtype=mx.int32),
                    0,
                )
            sampled = synchronized
        else:
            # Pure TP reductions carry partial tensor results throughout the
            # model and retain the proven synchronized sampler ordering.
            sampled = mx.distributed.all_sum(sampled, group=group)
        mx.eval(sampled)
        instance._next_tokens = sampled.astype(mx.uint32)
        instance._next_logprobs = list(logprobs)
        mx.async_eval(
            instance._next_tokens,
            instance._next_logprobs,
            token_context,
        )

        mx.eval(inputs, instance._current_logprobs)
        # MLX uint32 scalar/list values are valid model tokens but are not
        # valid Python indexing objects on every MLX build. The private
        # server later indexes ``r.logprobs[r.token]`` and pipeline canaries
        # exposed this as "Slice indices must be 32-bit integers" after the
        # fourth token. Flatten the batch decision and publish ordinary Python
        # ints at the Response boundary; the device-side next-token array stays
        # uint32 and all collective ordering is unchanged.
        input_values = [int(token) for token in inputs.reshape(-1).tolist()]
        for sequence_tokens, token in zip(instance.tokens, input_values):
            sequence_tokens.append(token)
        return input_values, instance._current_logprobs

    if vocab_sampling_active:
        vocab_parallel_head._omlx_gather_vocab_logits = False
        for head in mtp_auxiliary_heads:
            head._omlx_gather_vocab_logits = False
    if mtp_vocab_coordinator is not None:
        model._omlx_mtp_vocab_coordinator = mtp_vocab_coordinator
    if sampling_active and pipeline_parallel:
        mx.distributed.all_gather = selective_all_gather
        mx.distributed.send = queued_pipeline_send
        type(pipeline_model).__call__ = local_pipeline_output
    if sampling_active:
        mlx_generate.GenerationBatch._step = coordinator_generation_step
    if outer_prefill_yield_active:
        mlx_generate.BatchGenerator.next = yielding_batch_next
    if prefill_active:
        mlx_generate.PromptProcessingBatch.prompt = staggered_pipeline_prompt
    elif skip_logits_active:
        mlx_generate.PromptProcessingBatch.prompt = skip_prefill_logits_prompt
    try:
        yield capabilities
    finally:
        if original_prompt is not None:
            mlx_generate.PromptProcessingBatch.prompt = original_prompt
        if original_batch_next is not None:
            mlx_generate.BatchGenerator.next = original_batch_next
        if sampling_active:
            mlx_generate.GenerationBatch._step = original_generation_step
        if sampling_active and pipeline_parallel:
            type(pipeline_model).__call__ = original_pipeline_call
            mx.distributed.send = original_send
            mx.distributed.all_gather = original_all_gather
        if vocab_sampling_active:
            vocab_parallel_head._omlx_gather_vocab_logits = previous_vocab_gather
            for head, previous in zip(
                mtp_auxiliary_heads, previous_auxiliary_gathers
            ):
                head._omlx_gather_vocab_logits = previous
        if mtp_vocab_coordinator is not None:
            if previous_mtp_coordinator is missing_mtp_coordinator:
                delattr(model, "_omlx_mtp_vocab_coordinator")
            else:
                model._omlx_mtp_vocab_coordinator = previous_mtp_coordinator
