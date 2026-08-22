# SPDX-License-Identifier: Apache-2.0
"""Tests for rank-local, end-to-end distributed inference telemetry."""

import threading
import time
from types import SimpleNamespace

from omlx.cluster.performance import execution_profile
from omlx.cluster.planner import PipelineAssignment
from omlx.cluster.telemetry import (
    RuntimeTelemetry,
    _TelemetryQueue,
    install_server_telemetry,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class _Marker:
    def __init__(self) -> None:
        self.updates = []

    def update(self, phase, **extra):
        self.updates.append((phase, extra))


class _Queue:
    def __init__(self) -> None:
        self.items = []

    def put(self, item, *args, **kwargs):
        self.items.append((item, args, kwargs))
        return "queued"


def test_telemetry_calculates_ttft_prefill_and_decode_rates():
    clock = _Clock()
    marker = _Marker()
    telemetry = RuntimeTelemetry(marker, clock=clock, publish_interval=0)
    request_id = telemetry.begin_request()

    clock.value = 0.5
    telemetry.observe_context(
        request_id,
        prompt_tokens=10,
        cached_tokens=2,
    )
    clock.value = 1.0
    telemetry.observe_token(request_id)
    clock.value = 2.0
    telemetry.observe_token(request_id)
    clock.value = 3.0
    telemetry.finish_request(request_id)

    snapshot = telemetry.snapshot()
    request = snapshot["last_request"]
    assert snapshot["scope"] == "end_to_end_pipeline"
    assert snapshot["active_requests"] == 0
    assert snapshot["requests_completed"] == 1
    assert snapshot["requests_cancelled"] == 0
    assert snapshot["prompt_tokens_total"] == 10
    assert snapshot["completion_tokens_total"] == 2
    assert request["ttft_seconds"] == 1.0
    assert request["prefill_tps"] == 8.0
    assert request["decode_tps"] == 0.5
    assert request["end_to_end_tps"] == 2 / 3
    assert marker.updates[-1][0] == "ready"


def test_telemetry_publishes_structured_mtp_economics(monkeypatch):
    from omlx.patches.mlx_lm_mtp import batch_generator

    expected = {
        "sequences": 1,
        "tokens": 120,
        "cycles": 40,
        "accepted_draft_tokens": 80,
        "drafted_tokens": 100,
        "zero_depth_cycles": 0,
        "acceptance_ratio": 0.8,
        "tokens_per_cycle": 3.0,
        "depth_drafted": [40, 35, 25],
        "depth_accepted": [38, 30, 12],
        "timing_ms": {
            "backbone": 800.0,
            "mtp_head": 200.0,
            "sampling": 20.0,
            "cache_ops": 10.0,
        },
        "last_finish_reason": "length",
    }
    monkeypatch.setattr(batch_generator, "mtp_runtime_stats_snapshot", lambda: expected)

    snapshot = RuntimeTelemetry(_Marker(), clock=_Clock()).snapshot()

    assert snapshot["mtp"] == expected


def test_aggregate_decode_tps_uses_decode_time_not_uptime():
    """Idle uptime must not dilute the aggregate decode rate.

    The old divisor (process uptime) reported ~2 tok/s on a deployment whose
    requests decoded at ~23 tok/s, because a serving rank is mostly idle
    between requests.
    """
    clock = _Clock()
    marker = _Marker()
    telemetry = RuntimeTelemetry(marker, clock=clock, publish_interval=0)
    request_id = telemetry.begin_request()
    telemetry.observe_context(request_id, prompt_tokens=10, cached_tokens=0)

    clock.value = 10.0  # queue + prefill before the first decoded token
    telemetry.observe_token(request_id)
    clock.value = 11.0
    telemetry.observe_token(request_id)
    clock.value = 12.0
    telemetry.finish_request(request_id)

    # 90 s of idle uptime before anyone reads the marker.
    clock.value = 102.0
    snapshot = telemetry.snapshot()
    # 2 tokens over the 2 s decode window (first token -> finish).
    assert snapshot["aggregate_decode_tps"] == 1.0
    # The old semantic survives under an honest name.
    assert snapshot["aggregate_wall_tps"] == 2 / 102.0

    # An in-flight request contributes its decode window while active.
    request_id = telemetry.begin_request()
    telemetry.observe_context(request_id, prompt_tokens=4, cached_tokens=0)
    clock.value = 202.0
    telemetry.observe_token(request_id)
    clock.value = 204.0
    telemetry.observe_token(request_id)
    snapshot = telemetry.snapshot()
    # (2 finished tokens + 2 active) / (2 s finished + 2 s active).
    assert snapshot["aggregate_decode_tps"] == 1.0


def test_telemetry_publishes_live_mlx_lm_prefill_progress():
    clock = _Clock()
    marker = _Marker()
    telemetry = RuntimeTelemetry(marker, clock=clock, publish_interval=0)
    request_id = telemetry.begin_request()

    clock.value = 0.25
    telemetry.observe_context(
        request_id,
        prompt_tokens=12_000,
        cached_tokens=4_000,
    )
    telemetry.mark_pending_uid(request_id)
    telemetry.bind_pending_uid((73,))

    clock.value = 2.25
    telemetry.observe_prefill_progress(
        73,
        processed_tokens=2_000,
        total_tokens=8_000,
    )

    request = telemetry.snapshot()["last_request"]
    progress = request["prefill_progress"]
    assert request["status"] == "running"
    assert request["ttft_seconds"] is None
    assert request["decode_tps"] == 0.0
    assert request["prefill_tps"] == 1_000.0
    assert progress == {
        "active": True,
        "processed": 2_000,
        "total": 8_000,
        "speed": 1_000.0,
        "average_speed": 1_000.0,
        "eta": 6.0,
        "elapsed": 2.0,
    }

    clock.value = 4.25
    telemetry.observe_prefill_progress(
        73,
        processed_tokens=4_000,
        total_tokens=8_000,
    )
    progress = telemetry.snapshot()["last_request"]["prefill_progress"]
    assert progress["processed"] == 4_000
    assert progress["speed"] == 1_000.0
    assert progress["average_speed"] == 1_000.0
    assert progress["eta"] == 4.0

    clock.value = 8.25
    telemetry.observe_prefill_progress(
        73,
        processed_tokens=8_000,
        total_tokens=8_000,
    )
    telemetry.observe_token(request_id)
    request = telemetry.snapshot()["last_request"]
    assert request["prefill_progress"]["active"] is False
    assert request["prefill_progress"]["processed"] == 8_000
    assert request["ttft_seconds"] == 8.25


def test_completed_prefill_progress_clock_stops_at_prefill_end():
    """Decode time must not leak into prefill_progress elapsed/average.

    Observed on the live TP=2 deployment: a request with ttft 2.49 s reported
    prefill_progress.elapsed 25.3 s and average_speed 16 tok/s because the
    snapshot divided processed tokens by (finish_time - prefill_start).
    """

    clock = _Clock()
    telemetry = RuntimeTelemetry(_Marker(), clock=clock, publish_interval=0)
    request_id = telemetry.begin_request()

    clock.value = 0.5
    telemetry.observe_context(request_id, prompt_tokens=407, cached_tokens=0)
    telemetry.mark_pending_uid(request_id)
    telemetry.bind_pending_uid((73,))

    # Single 407-token chunk finishes prefill at t=2.99 (≈163 tok/s).
    clock.value = 2.99
    telemetry.observe_prefill_progress(73, processed_tokens=407, total_tokens=407)
    telemetry.observe_token(request_id)

    # Decode runs for ~23 s, then the request finishes.
    clock.value = 26.0
    for _ in range(545):
        telemetry.observe_token(request_id)
    telemetry.finish_request(request_id)

    request = telemetry.snapshot()["last_request"]
    progress = request["prefill_progress"]
    assert request["ttft_seconds"] == 2.99
    assert request["elapsed_seconds"] == 26.0
    assert progress["active"] is False
    # Frozen at the final chunk callback, not at finish time.
    assert progress["elapsed"] == 2.49
    assert progress["average_speed"] == 407 / 2.49
    assert request["prefill_tps"] == 407 / 2.99

    # A later heartbeat re-snapshot must not age the completed prefill either.
    clock.value = 60.0
    progress = telemetry.snapshot()["last_request"]["prefill_progress"]
    assert progress["elapsed"] == 2.49
    assert progress["average_speed"] == 407 / 2.49


def test_live_prefill_separates_recent_chunk_rate_from_sustained_average():
    """A slow later chunk must not relabel the whole request as 200 tok/s."""

    clock = _Clock()
    telemetry = RuntimeTelemetry(_Marker(), clock=clock, publish_interval=0)
    request_id = telemetry.begin_request()
    telemetry.observe_context(
        request_id,
        prompt_tokens=8_000,
        cached_tokens=0,
    )
    telemetry.mark_pending_uid(request_id)
    telemetry.bind_pending_uid((73,))

    clock.value = 2.0
    telemetry.observe_prefill_progress(
        73,
        processed_tokens=2_000,
        total_tokens=8_000,
    )
    clock.value = 12.0
    telemetry.observe_prefill_progress(
        73,
        processed_tokens=4_000,
        total_tokens=8_000,
    )

    request = telemetry.snapshot()["last_request"]
    progress = request["prefill_progress"]
    assert progress["speed"] == 200.0
    assert progress["average_speed"] == 4_000 / 12
    assert request["prefill_tps"] == 4_000 / 12
    assert progress["eta"] == 20.0


def test_queue_observer_preserves_mlx_lm_queue_contract():
    marker = _Marker()
    telemetry = RuntimeTelemetry(marker, publish_interval=0)
    target = _Queue()
    queue = _TelemetryQueue(target, telemetry)
    context = SimpleNamespace(prompt=[1, 2, 3, 4], prompt_cache_count=1)
    token = SimpleNamespace(token=7, finish_reason=None)

    assert queue.put(context, False) == "queued"
    assert queue.put(token) == "queued"
    assert queue.put(None) == "queued"

    snapshot = telemetry.snapshot()
    assert [item[0] for item in target.items] == [context, token, None]
    assert target.items[0][1] == (False,)
    assert snapshot["active_requests"] == 0
    assert snapshot["requests_completed"] == 1
    assert snapshot["prompt_tokens_total"] == 4
    assert snapshot["cached_tokens_total"] == 1
    assert snapshot["completion_tokens_total"] == 1


def test_telemetry_marker_failure_never_interrupts_inference():
    class BrokenMarker:
        def update(self, phase, **extra):
            raise OSError("disk unavailable")

    telemetry = RuntimeTelemetry(BrokenMarker(), publish_interval=0)

    request_id = telemetry.begin_request()
    telemetry.observe_context(request_id, prompt_tokens=2, cached_tokens=0)
    telemetry.observe_token(request_id)
    telemetry.finish_request(request_id)

    assert telemetry.snapshot()["requests_completed"] == 1


def test_telemetry_reports_coalescing_cache_affinity_and_stage_prediction():
    clock = _Clock()
    marker = _Marker()
    assignment = PipelineAssignment(
        "local",
        0,
        2,
        6,
        40,
        5,
        10,
        100,
        predicted_compute_seconds=0.2,
        predicted_send_seconds=0.01,
        predicted_stage_seconds=0.21,
    )
    telemetry = RuntimeTelemetry(
        marker,
        clock=clock,
        publish_interval=0,
        execution=execution_profile("balanced"),
        assignment=assignment,
    )
    clock.value = 1.0
    telemetry.observe_batch_step(
        prompt_responses=2,
        generation_responses=4,
        elapsed_seconds=0.25,
    )
    telemetry.observe_cache_lookup(
        prompt_tokens=100,
        remaining_tokens=25,
        entries=3,
        nbytes=4096,
    )

    snapshot = telemetry.snapshot()

    assert snapshot["pipeline"]["last_batch"]["coalesced_batch_size"] == 4
    assert snapshot["pipeline"]["microbatch_target"] == 4
    assert snapshot["pipeline"]["utilization"] == 0.25
    assert snapshot["cache"]["affinity"] == "deployment"
    assert snapshot["cache"]["hit_rate"] == 1.0
    assert snapshot["cache"]["tokens_reused"] == 75
    assert snapshot["stage"]["predicted_stage_seconds"] == 0.21
    assert snapshot["stage"]["observed_step_seconds"] == 0.25


def test_batch_uid_cancellation_closes_request_on_every_rank():
    marker = _Marker()
    telemetry = RuntimeTelemetry(marker, publish_interval=0)
    request_id = telemetry.begin_request()
    telemetry.observe_context(request_id, prompt_tokens=8, cached_tokens=2)
    telemetry.mark_pending_uid(request_id)
    telemetry.bind_pending_uid((42,))

    telemetry.cancel_uids([42])

    snapshot = telemetry.snapshot()
    assert snapshot["active_requests"] == 0
    assert snapshot["requests_completed"] == 0
    assert snapshot["requests_cancelled"] == 1
    assert snapshot["last_request"]["status"] == "cancelled"


def test_server_patch_binds_batch_uid_and_restores_mlx_lm_classes(monkeypatch):
    import mlx_lm.server as mlx_server

    class FakeResponseGenerator:
        def __init__(self):
            self.model_provider = SimpleNamespace(model_key="model")
            self.prompt_cache = mlx_server.LRUPromptCache()

        def _share_request(self, request):
            return request

        def _tokenize(self, _tokenizer, _request, _args):
            prompt = [1, 2, 3, 4]
            return prompt, [prompt], ["assistant"], "normal"

    class FakeBatchGenerator:
        def __init__(self):
            self.removed = []

        def insert_segments(self, *args, **kwargs):
            return (73,)

        def next(self):
            return (
                [SimpleNamespace(uid=73, progress=(2, 3))],
                [],
            )

        def remove(self, uids):
            self.removed.extend(uids)
            return "removed"

    class FakePromptCache:
        def fetch_nearest_cache(self, _model, tokens):
            return "cache", tokens[2:]

        def insert_cache(self, *args, **kwargs):
            return None

        def __len__(self):
            return 1

        @property
        def nbytes(self):
            return 64

    monkeypatch.setattr(
        mlx_server,
        "ResponseGenerator",
        FakeResponseGenerator,
    )
    monkeypatch.setattr(mlx_server, "BatchGenerator", FakeBatchGenerator)
    monkeypatch.setattr(mlx_server, "LRUPromptCache", FakePromptCache)
    marker = _Marker()
    target = _Queue()
    guard_calls = []
    guard = SimpleNamespace(
        check_collective=lambda *args, **kwargs: guard_calls.append((args, kwargs))
    )

    with install_server_telemetry(marker, prefill_guard=guard) as telemetry:
        generator = mlx_server.ResponseGenerator()
        queue, request, args = generator._share_request((target, "request", "args"))
        queue.put(
            SimpleNamespace(
                prompt=[1, 2, 3],
                prompt_cache_count=1,
            )
        )
        batch = mlx_server.BatchGenerator()
        assert batch.insert_segments() == (73,)
        batch.next()
        progress = telemetry.snapshot()["last_request"]["prefill_progress"]
        assert progress["processed"] == 2
        assert progress["total"] == 3
        assert progress["active"] is True
        assert batch.remove([73]) == "removed"
        assert generator._tokenize(None, None, None)[0] == [1, 2, 3, 4]
        assert generator.prompt_cache.fetch_nearest_cache(
            "model", [1, 2, 3, 4]
        ) == (
            "cache",
            [3, 4],
        )
        assert guard_calls[0][0] == (4,)
        assert guard_calls[0][1]["cached_tokens"] == 2
        assert guard_calls[0][1]["mx_module"] is not None
        assert request == "request"
        assert args == "args"
        assert telemetry.snapshot()["requests_cancelled"] == 1

    assert mlx_server.ResponseGenerator is FakeResponseGenerator
    assert mlx_server.BatchGenerator is FakeBatchGenerator


def test_server_patch_broadcasts_distributed_request_point_to_point(monkeypatch):
    import mlx.core as mx
    import mlx_lm.server as mlx_server

    class Group:
        @staticmethod
        def rank():
            return 0

        @staticmethod
        def size():
            return 2

    class FakeResponseGenerator:
        def __init__(self):
            self._is_distributed = True
            self._rank = 0

        def _share_object(self, _obj):
            raise AssertionError("distributed path must not call MLX-LM all-sum share")

        def _share_request(self, request):
            shareable = self._share_object(request[1:] if request else None)
            return None if shareable is None else (request[0], *shareable)

    sent = []
    monkeypatch.setattr(mlx_server, "ResponseGenerator", FakeResponseGenerator)
    monkeypatch.setattr(mx.distributed, "init", lambda: Group())
    monkeypatch.setattr(
        mx.distributed,
        "send",
        lambda value, target: sent.append((target, value.dtype, value.tolist()))
        or value,
    )
    monkeypatch.setattr(mx, "eval", lambda *_values: None)
    target = _Queue()

    with install_server_telemetry(_Marker(), prefill_guard=None):
        generator = mlx_server.ResponseGenerator()
        queue, request, args = generator._share_request(
            (target, "request", {"max_tokens": 7})
        )

    assert queue._queue is target
    assert request == "request"
    assert args == {"max_tokens": 7}
    assert len(sent) == 2
    assert sent[0][0:2] == (1, mx.int32)
    assert sent[0][2][0] == len(bytes(sent[1][2]))
    assert sent[1][0:2] == (1, mx.uint8)


def test_server_patch_receives_distributed_request_point_to_point(monkeypatch):
    import pickle

    import mlx.core as mx
    import mlx_lm.server as mlx_server

    payload = pickle.dumps(("request", {"max_tokens": 7}))
    received = [
        mx.array([len(payload)], dtype=mx.int32),
        mx.array(payload, dtype=mx.uint8),
    ]

    class Group:
        @staticmethod
        def rank():
            return 1

        @staticmethod
        def size():
            return 2

    class FakeResponseGenerator:
        def __init__(self):
            self._is_distributed = True
            self._rank = 1

        def _share_object(self, _obj):
            raise AssertionError("distributed path must not call MLX-LM all-sum share")

        def _share_request(self, request):
            from queue import Queue

            shareable = self._share_object(request[1:] if request else None)
            return None if shareable is None else (Queue(), *shareable)

    monkeypatch.setattr(mlx_server, "ResponseGenerator", FakeResponseGenerator)
    monkeypatch.setattr(mx.distributed, "init", lambda: Group())
    monkeypatch.setattr(
        mx.distributed,
        "recv_like",
        lambda _value, source: received.pop(0) if source == 0 else None,
    )

    with install_server_telemetry(_Marker(), prefill_guard=None):
        generator = mlx_server.ResponseGenerator()
        queue, request, args = generator._share_request(None)

    assert isinstance(queue, _TelemetryQueue)
    assert request == "request"
    assert args == {"max_tokens": 7}
    assert received == []


def test_sequential_distributed_cancellation_exits_all_ranks_without_upstream_error(
    monkeypatch,
):
    """The pinned server raises NotImplementedError here without our patch."""

    import mlx_lm.server as mlx_server

    observed = []

    class FakeResponseGenerator:
        def __init__(self):
            self._is_distributed = True

        def _serve_single(self, _request):
            ctx = mlx_server.GenerationContext(
                has_tool_calling=False,
                has_thinking=False,
                tool_parser=lambda *_args: {},
                sequences={},
                prompt=[],
            )
            ctx.stop()
            if ctx._should_stop:
                if self._is_distributed:
                    raise NotImplementedError()
                observed.append("cancelled")

    class FakeBatchGenerator:
        pass

    original_context = mlx_server.GenerationContext
    monkeypatch.setattr(mlx_server, "ResponseGenerator", FakeResponseGenerator)
    monkeypatch.setattr(mlx_server, "BatchGenerator", FakeBatchGenerator)

    with install_server_telemetry(_Marker()):
        generator = mlx_server.ResponseGenerator()
        generator._serve_single(("queue", "request", "args"))
        assert generator._is_distributed is True
        assert mlx_server.GenerationContext is not original_context

    assert observed == ["cancelled"]
    assert mlx_server.GenerationContext is original_context


# ---------------------------------------------------------------------------
# The idle heartbeat.
#
# Every publish here used to be request-driven, so an idle rank's marker simply
# stopped ageing. The peer watchdog reads that timestamp and calls anything
# older than 45 s stale, so a healthy, loaded, serving cluster killed itself
# 60 s after the last token — and in conversational use that is between every
# turn, each one paying for a full model reload.
# ---------------------------------------------------------------------------


class _CountingMarker:
    def __init__(self) -> None:
        self.updates = []
        self._event = threading.Event()
        self._lock = threading.Lock()

    def update(self, phase, **extra):
        with self._lock:
            self.updates.append((phase, extra))
        self._event.set()

    def wait_for_update(self, timeout=5.0) -> bool:
        return self._event.wait(timeout)

    def count(self) -> int:
        with self._lock:
            return len(self.updates)


def test_an_idle_rank_still_refreshes_its_marker():
    """No requests, no tokens, nothing to report — and the marker still ages."""

    marker = _CountingMarker()
    telemetry = RuntimeTelemetry(
        marker, publish_interval=0, heartbeat_interval=0.01
    )

    telemetry.start_heartbeat()
    try:
        assert marker.wait_for_update(timeout=5.0), (
            "an idle rank published nothing; the peer watchdog will call it stale"
        )
    finally:
        telemetry.stop_heartbeat()

    assert marker.updates[0][0] == "ready"
    assert marker.count() >= 1


def test_stopping_the_heartbeat_ends_the_thread():
    marker = _CountingMarker()
    telemetry = RuntimeTelemetry(
        marker, publish_interval=0, heartbeat_interval=0.01
    )
    before = set(threading.enumerate())

    telemetry.start_heartbeat()
    telemetry.start_heartbeat()  # idempotent
    assert marker.wait_for_update(timeout=5.0)
    telemetry.stop_heartbeat()
    settled = marker.count()
    time.sleep(0.1)

    assert marker.count() == settled, "the heartbeat outlived stop_heartbeat"
    leaked = {
        thread
        for thread in threading.enumerate()
        if thread not in before and thread.is_alive()
        and thread.name == "omlx-cluster-telemetry-heartbeat"
    }
    assert not leaked


def test_the_heartbeat_advances_the_timestamp_a_peer_watchdog_reads(tmp_path):
    """The writer and the reader, not two hand-typed dicts.

    ``marker_age_seconds`` is what decides "stale"; a heartbeat that refreshed
    some other field would look identical in a mock and change nothing.
    """

    from omlx.cluster.inference_worker import RuntimeMarker
    from omlx.cluster.liveness import marker_age_seconds, read_marker

    marker = RuntimeMarker(
        state_dir=str(tmp_path),
        deployment_id="d",
        rank=0,
        world_size=2,
        model="org/model",
        backend="ring",
        plan_hash="a" * 64,
    )
    marker.update("ready", start_layer=0, end_layer=4)
    first = read_marker(marker.path)["updated_at"]

    telemetry = RuntimeTelemetry(marker, publish_interval=0, heartbeat_interval=0.01)
    telemetry.start_heartbeat()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if read_marker(marker.path)["updated_at"] != first:
                break
            time.sleep(0.01)
        else:  # pragma: no cover - only on a wedged heartbeat
            raise AssertionError("the marker's updated_at never advanced")
    finally:
        telemetry.stop_heartbeat()

    payload = read_marker(marker.path)
    assert payload["phase"] == "ready"
    assert marker_age_seconds(payload) < 45.0, "still inside the staleness window"


def test_serving_starts_the_heartbeat_without_the_caller_asking(monkeypatch):
    """The seam: install_server_telemetry owns the span a rank is alive for.

    A heartbeat the worker has to remember to start is a heartbeat a refactor
    will drop, and dropping it restores the 60-second self-kill silently.
    """

    import mlx_lm.server as mlx_server

    class FakeResponseGenerator:
        pass

    class FakeBatchGenerator:
        pass

    monkeypatch.setattr(mlx_server, "ResponseGenerator", FakeResponseGenerator)
    monkeypatch.setattr(mlx_server, "BatchGenerator", FakeBatchGenerator)
    marker = _CountingMarker()

    with install_server_telemetry(marker, heartbeat_interval=0.01) as telemetry:
        assert marker.wait_for_update(timeout=5.0), (
            "serving did not refresh the marker while idle"
        )
        assert telemetry._heartbeat_thread is not None

    settled = marker.count()
    time.sleep(0.1)
    assert marker.count() == settled, "the heartbeat outlived the serving block"


class _BatchGenerator:
    def __init__(self) -> None:
        self.removed = []

    def remove(self, uids):
        self.removed.append(list(uids))


def _cancel_telemetry(tmp_path, clock=None):
    marker = _Marker()
    telemetry = RuntimeTelemetry(
        marker,
        clock=clock or _Clock(),
        publish_interval=0,
        cancel_path=tmp_path / "dep-1-cancel.json",
        cancel_deployment_id="dep-1",
    )
    return telemetry


def test_force_cancel_all_removes_active_uids_through_the_batch_loop(tmp_path):
    telemetry = _cancel_telemetry(tmp_path)
    generator = _BatchGenerator()
    telemetry.register_batch_generator(generator)
    request_id = telemetry.begin_request()
    telemetry.mark_pending_uid(request_id)
    telemetry.bind_pending_uid((73,))

    cancelled = telemetry.force_cancel_all(reason="test")

    assert cancelled == 1
    assert generator.removed == [[73]]
    # remove() routes back through cancel_uids in production; here the
    # fake does not, so the request is still tracked until it does.
    telemetry.cancel_uids([73])
    assert telemetry._requests == {}
    assert telemetry._requests_cancelled == 1


def test_force_cancel_all_without_generator_or_uids_is_a_noop(tmp_path):
    telemetry = _cancel_telemetry(tmp_path)

    assert telemetry.force_cancel_all(reason="test") == 0

    generator = _BatchGenerator()
    telemetry.register_batch_generator(generator)
    assert telemetry.force_cancel_all(reason="test") == 0
    assert generator.removed == []


def test_force_cancel_all_survives_a_failing_batch_loop(tmp_path):
    telemetry = _cancel_telemetry(tmp_path)

    class BrokenGenerator:
        def remove(self, uids):
            raise RuntimeError("wedged")

    telemetry.register_batch_generator(BrokenGenerator())
    request_id = telemetry.begin_request()
    telemetry.mark_pending_uid(request_id)
    telemetry.bind_pending_uid((5,))

    assert telemetry.force_cancel_all(reason="test") == 0
    # The request stays tracked; the coordinator's process teardown is the
    # fallback for this failure mode.
    assert request_id in telemetry._requests


def test_cancel_file_is_consumed_once_and_acked(tmp_path):
    import json

    telemetry = _cancel_telemetry(tmp_path)
    generator = _BatchGenerator()
    telemetry.register_batch_generator(generator)
    request_id = telemetry.begin_request()
    telemetry.mark_pending_uid(request_id)
    telemetry.bind_pending_uid((9,))

    cancel_path = tmp_path / "dep-1-cancel.json"
    cancel_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deployment_id": "dep-1",
                "epoch": 42,
                "scope": "all",
                "reason": "memory pressure",
            }
        ),
        encoding="utf-8",
    )

    assert telemetry.poll_cancel_requests(min_interval=0.0) == 1
    assert generator.removed == [[9]]
    ack = json.loads(
        (tmp_path / "dep-1-cancel-ack.json").read_text(encoding="utf-8")
    )
    assert ack["epoch"] == 42
    assert ack["cancelled"] == 1

    # Same epoch is not consumed twice.
    assert telemetry.poll_cancel_requests(min_interval=0.0) == 0
    assert generator.removed == [[9]]


def test_cancel_file_from_a_foreign_deployment_is_ignored(tmp_path):
    import json

    telemetry = _cancel_telemetry(tmp_path)
    generator = _BatchGenerator()
    telemetry.register_batch_generator(generator)
    (tmp_path / "dep-1-cancel.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deployment_id": "somebody-else",
                "epoch": 7,
                "scope": "all",
            }
        ),
        encoding="utf-8",
    )

    assert telemetry.poll_cancel_requests(min_interval=0.0) == 0
    assert generator.removed == []
