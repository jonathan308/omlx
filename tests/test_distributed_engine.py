# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from omlx.cluster.deployment import ClusterDeployment, ClusterHost
from omlx.cluster.performance import execution_profile
from omlx.cluster.planner import PipelineAssignment
from omlx.cluster.strategy_benchmarks import configure_strategy_benchmark_store
from omlx.engine import distributed
from omlx.engine.distributed import (
    DistributedBatchedEngine,
    DistributedInferenceError,
)


def _deployment() -> ClusterDeployment:
    return ClusterDeployment(
        deployment_id="engine-test",
        model="org/model",
        backend="ring",
        hosts=(
            ClusterHost("local", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("peer", "peer.local", ("10.0.0.2",)),
        ),
        assignments=(
            PipelineAssignment("local", 0, 2, 4, 2, 0, 0, 4),
            PipelineAssignment("peer", 1, 0, 2, 2, 0, 0, 4),
        ),
        plan_hash="d" * 64,
    )


class _Tokenizer:
    @staticmethod
    def encode(text):
        return list(text.encode())


def _ready_engine(handler) -> DistributedBatchedEngine:
    engine = DistributedBatchedEngine(_deployment())
    engine._loaded = True
    engine._tokenizer = _Tokenizer()
    engine._client = httpx.AsyncClient(
        base_url="http://127.0.0.1:1",
        transport=httpx.MockTransport(handler),
    )
    return engine


def test_backend_chat_messages_serialize_native_tool_history_once():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather",
            "content": '{"temperature_c":18}',
        },
    ]

    prepared = DistributedBatchedEngine._backend_chat_messages(messages)

    assert prepared[0]["tool_calls"][0]["function"]["arguments"] == (
        '{"city": "Paris"}'
    )
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert prepared[1] == messages[1]


@pytest.mark.asyncio
async def test_private_rank_zero_client_has_finite_inactivity_timeouts():
    engine = DistributedBatchedEngine(_deployment(), request_read_timeout=12.5)
    client = engine._new_client("http://127.0.0.1:1")
    try:
        assert client.timeout.connect == 10.0
        assert client.timeout.read == 12.5
        assert client.timeout.write == 30.0
        assert client.timeout.pool == 10.0
    finally:
        await client.aclose()


def _stalled_engine():
    def handler(request):
        raise httpx.ReadTimeout("collective stalled", request=request)

    engine = _ready_engine(handler)
    # Read timeouts now drop a rank-side cancel file; keep it out of the
    # real runtime state dir.
    engine._supervisor.state_dir = tempfile.mkdtemp(prefix="omlx-test-runtime-")
    status_calls = []

    def status():
        status_calls.append(True)
        return SimpleNamespace(
            returncode=None,
            failure_reason=None,
            phase="ready",
        )

    engine._supervisor.status = status
    return engine, status_calls


@pytest.mark.asyncio
async def test_distributed_generate_bounds_rank_zero_read_stalls():
    engine, status_calls = _stalled_engine()
    try:
        with pytest.raises(
            DistributedInferenceError,
            match="request timed out.*no rank-zero data.*cluster was ready",
        ):
            await engine.generate("hello")
    finally:
        await engine._client.aclose()

    assert len(status_calls) == 2, "availability must be rechecked after timeout"


@pytest.mark.asyncio
async def test_distributed_stream_bounds_rank_zero_read_stalls():
    engine, status_calls = _stalled_engine()
    try:
        with pytest.raises(
            DistributedInferenceError,
            match="stream timed out.*no rank-zero data.*cluster was ready",
        ):
            [output async for output in engine.stream_generate("hello")]
    finally:
        await engine._client.aclose()

    assert len(status_calls) == 2, "availability must be rechecked after timeout"


@pytest.mark.asyncio
async def test_distributed_generate_translates_backend_completion():
    def handler(request):
        body = json.loads(request.content)
        assert body["prompt"] == "Hello"
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "choices": [{"text": " world", "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 1},
                },
            },
        )

    engine = _ready_engine(handler)
    try:
        output = await engine.generate("Hello", max_tokens=8)
    finally:
        await engine._client.aclose()

    assert output.text == " world"
    assert output.prompt_tokens == 1
    assert output.completion_tokens == 2
    assert output.cached_tokens == 1
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_distributed_chat_preserves_rank_zero_tool_calls_and_reasoning():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]

    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["messages"] == [{"role": "user", "content": "Weather?"}]
        assert body["tools"] == tools
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I'll check.",
                            "reasoning": "A weather lookup is required.",
                            "tool_calls": [
                                {
                                    "id": "call_weather",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Paris"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "prompt_tokens_details": {"cached_tokens": 3},
                },
            },
        )

    engine = _ready_engine(handler)
    try:
        output = await engine.chat(
            [{"role": "user", "content": "Weather?"}],
            tools=tools,
        )
    finally:
        await engine._client.aclose()

    assert output.text == ("<think>A weather lookup is required.</think>I'll check.")
    assert output.finish_reason == "tool_calls"
    assert output.tool_calls == [
        {
            "id": "call_weather",
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
        }
    ]
    assert output.cached_tokens == 3


@pytest.mark.asyncio
async def test_distributed_stream_chat_preserves_structured_tool_calls():
    events = [
        {
            "choices": [
                {
                    "delta": {"role": "assistant", "reasoning": "Need lookup."},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Paris"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "total_tokens": 17,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        },
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"

    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=content,
        )

    engine = _ready_engine(handler)
    try:
        outputs = [
            output
            async for output in engine.stream_chat(
                [{"role": "user", "content": "Weather?"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            )
        ]
    finally:
        await engine._client.aclose()

    assert outputs[0].new_text == "<think>Need lookup."
    assert outputs[-1].new_text == ""
    assert outputs[-1].text == "<think>Need lookup.</think>"
    assert outputs[-1].finish_reason == "tool_calls"
    assert outputs[-1].tool_calls == [
        {
            "id": "call_weather",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
        }
    ]
    assert outputs[-1].prompt_tokens == 12
    assert outputs[-1].completion_tokens == 5
    assert outputs[-1].cached_tokens == 2


@pytest.mark.asyncio
async def test_distributed_stream_waits_for_usage_before_final_output():
    events = [
        {
            "choices": [
                {"text": "A", "finish_reason": None},
            ]
        },
        {
            "choices": [
                {"text": "B", "finish_reason": "length"},
            ]
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        },
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=content,
        )

    engine = _ready_engine(handler)
    try:
        outputs = [output async for output in engine.stream_generate("test")]
    finally:
        await engine._client.aclose()

    assert [output.new_text for output in outputs] == ["A", "B"]
    assert outputs[0].finished is False
    assert outputs[0].completion_tokens == 1
    assert outputs[0].generated_at is not None
    assert outputs[0].generated_until == outputs[0].generated_at
    assert outputs[-1].finished is True
    assert outputs[-1].text == "AB"
    assert outputs[-1].finish_reason == "length"
    assert outputs[-1].prompt_tokens == 4
    assert outputs[-1].completion_tokens == 2
    assert outputs[-1].cached_tokens == 3
    assert outputs[-1].generated_at == outputs[0].generated_at


@pytest.mark.asyncio
async def test_stream_records_real_prefill_and_decode_for_automatic_choice(
    monkeypatch,
    tmp_path,
):
    from omlx.engine import distributed

    events = [
        {"choices": [{"text": "A", "finish_reason": None}]},
        {"choices": [{"text": "B", "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 32,
                "completion_tokens": 2,
                "total_tokens": 34,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        },
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"
    engine = _ready_engine(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=content,
        )
    )
    store = configure_strategy_benchmark_store(tmp_path)
    # Monotonic call order in stream_generate: request_started_at(10.0),
    # _enter_request state timestamp (11.0, unused for rates), first-chunk
    # now/first_token_at(12.0), _mark_backend_finished (15.0, unused),
    # finished_at(16.0). Rates: prefill 32/(12-10)=16, decode (2-1)/(16-12)=0.25.
    ticks = iter((10.0, 11.0, 12.0, 15.0, 16.0))
    monkeypatch.setattr(
        distributed,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )
    try:
        [output async for output in engine.stream_generate("x" * 32)]
    finally:
        await engine._client.aclose()

    measurements = store.measurements(
        model="org/model",
        node_ids=("local", "peer"),
        backend="ring",
        target_context_tokens=1024,
    )
    assert measurements[1].prompt_tokens_per_second == 16.0
    assert measurements[1].decode_tokens_per_second == 0.25
    assert measurements[1].time_to_first_token_seconds == 2.0


@pytest.mark.asyncio
async def test_strategy_benchmark_buckets_total_context_but_rates_uncached_prefill(
    tmp_path, monkeypatch
):
    events = [
        {"choices": [{"text": "A", "finish_reason": None}]},
        {"choices": [{"text": "B", "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 8192,
                "completion_tokens": 2,
                "total_tokens": 8194,
                "prompt_tokens_details": {"cached_tokens": 7168},
            },
        },
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"
    engine = _ready_engine(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=content,
        )
    )
    store = configure_strategy_benchmark_store(tmp_path)
    # Same monotonic call order as the automatic-choice test above; only the
    # started/first-token/finished values feed the rates.
    ticks = iter((10.0, 11.0, 12.0, 15.0, 16.0))
    monkeypatch.setattr(
        distributed,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )
    try:
        [output async for output in engine.stream_generate("x" * 8192)]
    finally:
        await engine._client.aclose()

    measurements = store.measurements(
        model="org/model",
        node_ids=("local", "peer"),
        backend="ring",
        target_context_tokens=8192,
    )
    assert measurements[1].context_tokens == 8192
    assert measurements[1].prompt_tokens_per_second == 512.0


@pytest.mark.asyncio
async def test_distributed_stream_rejects_malformed_usage():
    event = {
        "choices": [],
        "usage": {"prompt_tokens_details": "not-an-object"},
    }

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(event)}\n\n",
        )

    engine = _ready_engine(handler)
    try:
        with pytest.raises(
            DistributedInferenceError,
            match="invalid token details",
        ):
            [output async for output in engine.stream_generate("test")]
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_distributed_engine_surfaces_bounded_backend_error():
    def handler(request):
        return httpx.Response(503, json={"error": "rank 1 failed"})

    engine = _ready_engine(handler)
    try:
        with pytest.raises(DistributedInferenceError, match="HTTP 503.*rank 1"):
            await engine.generate("hello")
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_distributed_transport_error_surfaces_peer_failure_reason():
    def handler(request):
        raise httpx.RemoteProtocolError(
            "server disconnected",
            request=request,
        )

    engine = _ready_engine(handler)
    engine._supervisor.status = lambda: SimpleNamespace(
        returncode=1,
        failure_reason=(
            "Studio stopped publishing its runtime heartbeat. "
            "Check oMLX is running on that Mac."
        ),
        phase="failed",
        stderr_tail=(),
    )
    try:
        with pytest.raises(
            DistributedInferenceError,
            match="Studio stopped publishing its runtime heartbeat",
        ):
            await engine.generate("hello")
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_distributed_transport_error_reports_bounded_launcher_exit():
    def handler(request):
        raise httpx.RemoteProtocolError(
            "server disconnected",
            request=request,
        )

    engine = _ready_engine(handler)
    engine._supervisor.status = lambda: SimpleNamespace(
        returncode=1,
        failure_reason=None,
        phase="failed",
        stderr_tail=("rank 1 out of memory",),
    )
    try:
        with pytest.raises(
            DistributedInferenceError,
            match="exited with code 1.*rank 1 out of memory",
        ):
            await engine.generate("hello")
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_distributed_engine_rejects_unimplemented_grammar():
    def handler(request):
        raise AssertionError("backend should not be called")

    engine = _ready_engine(handler)
    try:
        with pytest.raises(ValueError, match="guided grammar"):
            await engine.generate("hello", compiled_grammar=object())
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_experimental_token_only_output_rejects_seeded_single_request():
    deployment = replace(
        _deployment(),
        execution=replace(
            execution_profile("balanced"),
            sampling_rank_only=True,
        ),
    )
    engine = DistributedBatchedEngine(deployment)
    engine._loaded = True
    engine._tokenizer = _Tokenizer()
    engine._client = httpx.AsyncClient(
        base_url="http://127.0.0.1:1",
        transport=httpx.MockTransport(
            lambda request: pytest.fail("backend should not be called")
        ),
    )
    try:
        with pytest.raises(ValueError, match="sampling-rank-only"):
            await engine.generate("hello", seed=7)
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_distributed_preflight_rejects_features_before_stream_starts():
    engine = _ready_engine(lambda request: httpx.Response(500))
    try:
        with pytest.raises(ValueError, match="thinking budgets"):
            await engine.preflight_chat(
                [{"role": "user", "content": "hello"}],
                thinking_budget=512,
            )
        with pytest.raises(ValueError, match="SpecPrefill"):
            await engine.preflight_chat(
                [{"role": "user", "content": "hello"}],
                specprefill=True,
            )
    finally:
        await engine._client.aclose()


def _rank_zero_marker(active_requests: int) -> dict:
    return {
        "schema_version": 1,
        "deployment_id": "engine-test",
        "pid": 424242,
        "rank": 0,
        "world_size": 2,
        "model": "org/model",
        "backend": "ring",
        "plan_hash": "d" * 64,
        "phase": "ready",
        "updated_at": "2026-08-17T00:00:00+00:00",
        "metrics": {"active_requests": active_requests},
    }


@pytest.mark.asyncio
async def test_abort_request_flags_and_closes_only_that_request(tmp_path):
    engine = _ready_engine(lambda request: httpx.Response(200, json={}))
    engine._supervisor.state_dir = str(tmp_path)
    first = await engine._enter_request()
    second = await engine._enter_request()
    closed = []

    class FakeResponse:
        async def aclose(self):
            closed.append(True)

    engine._request_states[first].response = FakeResponse()

    assert await engine.abort_request(first, reason="test") is True
    assert engine._request_states[first].aborted is True
    assert engine._request_states[second].aborted is False
    assert closed == [True]
    assert await engine.abort_request("engine-test-999") is False

    await engine._leave_request(first)
    await engine._leave_request(second)
    await engine._client.aclose()


@pytest.mark.asyncio
async def test_aborted_stream_raises_at_the_next_yield_boundary(tmp_path):
    events = [
        {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]},
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=content,
        )

    engine = _ready_engine(handler)
    engine._supervisor.state_dir = str(tmp_path)
    try:
        stream = engine.stream_chat([{"role": "user", "content": "hi"}])
        first = await stream.__anext__()
        assert first.new_text == "hello"
        request_id = next(iter(engine._request_states))
        assert await engine.abort_request(request_id, reason="operator") is True
        with pytest.raises(distributed.DistributedRequestAborted):
            await stream.__anext__()
        # The abort unwound the generator's finally: the counter drained.
        assert engine._active_requests == 0
        assert engine.has_active_requests() is False
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_abort_all_writes_cancel_file_and_swaps_client(tmp_path):
    engine = _ready_engine(lambda request: httpx.Response(200, json={}))
    engine._supervisor.state_dir = str(tmp_path)
    # The client swap needs a supervisor endpoint; the test double never
    # launches, so set the port a real launch would have allocated.
    engine._supervisor.port = 18010
    old_client = engine._client

    count = await engine.abort_all_requests(reason="unload requested")

    assert count == 0
    assert engine._client is not old_client
    cancel_path = tmp_path / "engine-test-cancel.json"
    payload = json.loads(cancel_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["deployment_id"] == "engine-test"
    assert payload["scope"] == "all"
    assert payload["reason"] == "unload requested"
    assert payload["epoch"] > 0
    await engine._client.aclose()


@pytest.mark.asyncio
async def test_backend_drain_waits_for_rank_side_evidence(tmp_path):
    engine = _ready_engine(lambda request: httpx.Response(200, json={}))
    engine._supervisor.state_dir = str(tmp_path)
    (tmp_path / "engine-test-rank-0.json").write_text(
        json.dumps(_rank_zero_marker(0)), encoding="utf-8"
    )
    try:
        assert await engine._wait_for_backend_drain(timeout=1.0) is True

        (tmp_path / "engine-test-rank-0.json").write_text(
            json.dumps(_rank_zero_marker(2)), encoding="utf-8"
        )
        started = time.monotonic()
        assert await engine._wait_for_backend_drain(timeout=0.3) is False
        assert time.monotonic() - started >= 0.3
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_abort_all_reports_rank_side_survivors(tmp_path):
    engine = _ready_engine(lambda request: httpx.Response(200, json={}))
    engine._supervisor.state_dir = str(tmp_path)
    engine._supervisor.port = 18010  # endpoint needed for the post-abort client swap
    engine._abort_drain_timeout = 0.2
    (tmp_path / "engine-test-rank-0.json").write_text(
        json.dumps(_rank_zero_marker(1)), encoding="utf-8"
    )
    try:
        # No local requests, but the rank still reports one: the drain wait
        # must engage and expire unconfirmed rather than trust client close.
        count = await engine.abort_all_requests(reason="memory pressure")
        assert count == 0
    finally:
        await engine._client.aclose()


def test_orphan_reaper_drops_finished_but_abandoned_requests():
    engine = DistributedBatchedEngine(_deployment())
    abandoned = distributed._DistributedRequestState("a", 100.0)
    abandoned.finished_at = 100.0
    live = distributed._DistributedRequestState("b", 100.0)
    fresh = distributed._DistributedRequestState("c", 100.0)
    fresh.finished_at = 199.0
    engine._request_states.update({"a": abandoned, "b": live, "c": fresh})
    engine._active_requests = 3

    reaped = engine.reap_orphaned_generators(now=200.0, grace=5.0)

    assert reaped == 1
    assert set(engine._request_states) == {"b", "c"}
    assert engine._active_requests == 2


def test_orphan_reaper_runs_from_has_active_requests():
    engine = DistributedBatchedEngine(_deployment())
    stale = distributed._DistributedRequestState("a", 0.0)
    stale.finished_at = 0.0
    engine._request_states["a"] = stale
    engine._active_requests = 1

    assert engine.has_active_requests() is False
    assert engine._request_states == {}


@pytest.mark.asyncio
async def test_read_timeout_drops_rank_side_cancel_file(tmp_path):
    engine, _ = _stalled_engine()
    engine._supervisor.state_dir = str(tmp_path)
    try:
        with pytest.raises(DistributedInferenceError):
            await engine.generate("hello")
    finally:
        await engine._client.aclose()

    cancel_path = tmp_path / "engine-test-cancel.json"
    payload = json.loads(cancel_path.read_text(encoding="utf-8"))
    assert payload["scope"] == "all"
    assert "read timeout" in payload["reason"]
