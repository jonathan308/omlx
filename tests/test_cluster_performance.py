# SPDX-License-Identifier: Apache-2.0
"""Performance-aware planner, launch probe, and runtime capability tests."""

import importlib
import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.cluster.deployment import ClusterDeployment, ClusterHost
from omlx.cluster.launch import run_cluster_performance_probe
from omlx.cluster.performance import (
    NodePerformanceProfile,
    execution_profile,
    performance_profiles_from_records,
    tune_execution_settings,
)
from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    PipelineAssignment,
    plan_unequal_pipeline,
)
from omlx.cluster.runtime_optimizations import (
    _deepseek_v4_fused_decode_capability,
    _indexer_row_parallel_capability,
    install_runtime_optimizations,
    pipeline_prefill_schedule,
)

mlx_generate = importlib.import_module("mlx_lm.generate")


def test_deepseek_v4_fused_decode_capability_tracks_mtp_and_rollback(monkeypatch):
    attention = SimpleNamespace(dspark=True, _omlx_decode_consistent=False)
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(attn=attention)])
    )

    monkeypatch.delenv("OMLX_DSV4_EXACT_DECODE", raising=False)
    assert _deepseek_v4_fused_decode_capability(model)[:2] == (True, True)

    attention._omlx_decode_consistent = True
    enabled, active, reason = _deepseek_v4_fused_decode_capability(model)
    assert enabled is True
    assert active is False
    assert "MTP" in reason

    monkeypatch.setenv("OMLX_DSV4_EXACT_DECODE", "1")
    enabled, active, reason = _deepseek_v4_fused_decode_capability(model)
    assert enabled is False
    assert active is False
    assert "EXACT_DECODE" in reason


def test_sparse_indexer_row_parallel_capability_reports_tp_contract(monkeypatch):
    monkeypatch.setenv("OMLX_DSV4_INDEXER_ROW_TP", "1")
    monkeypatch.setenv("OMLX_DSV4_INDEXER_ROW_TP_MIN_POOL", "2048")
    group = object()
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    attn=SimpleNamespace(
                        indexer=SimpleNamespace(row_sharding_group=group)
                    )
                )
            ]
        )
    )

    enabled, active, reason = _indexer_row_parallel_capability(
        model, world_size=2
    )

    assert enabled is True
    assert active is True
    assert "2048 pooled entries" in reason


def _profile(node_id: str, rank: int, rate: float) -> NodePerformanceProfile:
    return NodePerformanceProfile(
        node_id=node_id,
        rank=rank,
        decode_weight_bytes_per_second=rate,
        prefill_weight_bytes_per_second=rate,
        collective_latency_seconds=0.001,
        collective_bandwidth_bytes_per_second=10_000,
        backend="ring",
        measured_at="2026-07-26T12:00:00+00:00",
        samples=5,
    )


def test_performance_planner_prefers_faster_node_without_exceeding_memory():
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=0,
        layer_weight_bytes=(10,) * 8,
        activation_bytes_per_token=2,
    )
    plan = plan_unequal_pipeline(
        model,
        [
            NodeBudget(
                "slow",
                100,
                rank=0,
                performance=_profile("slow", 0, 10),
            ),
            NodeBudget(
                "fast",
                100,
                rank=1,
                performance=_profile("fast", 1, 40),
            ),
        ],
    )

    slow, fast = plan.assignments
    assert plan.optimization == "performance"
    assert fast.layer_count > slow.layer_count
    assert all(item.headroom_bytes >= 0 for item in plan.assignments)
    assert all(item.predicted_stage_seconds is not None for item in plan.assignments)
    assert plan.to_dict()["strategy"].startswith("performance_aware")


def test_partial_measurements_fall_back_to_original_memory_objective():
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=0,
        layer_weight_bytes=(10,) * 8,
    )
    plan = plan_unequal_pipeline(
        model,
        [
            NodeBudget(
                "first",
                100,
                rank=0,
                performance=_profile("first", 0, 10),
            ),
            NodeBudget("second", 100, rank=1),
        ],
    )

    assert plan.optimization == "memory"
    assert [item.layer_count for item in plan.assignments] == [4, 4]
    assert all(item.predicted_stage_seconds is None for item in plan.assignments)


def test_execution_tuner_reduces_concurrency_and_synchronizes_prompt_cache():
    settings = execution_profile("throughput")
    assignments = [
        SimpleNamespace(headroom_bytes=3 * 1024**3),
        SimpleNamespace(headroom_bytes=20 * 1024**3),
    ]

    tuned = tune_execution_settings(settings, assignments, backend="jaccl")

    assert tuned.decode_concurrency == 2
    assert tuned.prompt_concurrency == 1
    assert tuned.prefill_step_size == 512
    assert tuned.pipeline_microbatch_size == 1
    assert tuned.prompt_cache_size == 1
    assert tuned.prompt_cache_bytes is None
    assert tuned.ring_connections_per_ip == 1
    assert "critical headroom" in tuned.tuning_reason
    assert "synchronized single-prefix cache" in tuned.tuning_reason


def test_prompt_cache_is_synchronized_even_when_auto_tuning_is_disabled():
    settings = replace(
        execution_profile("throughput", auto_tune=False),
        prompt_cache_size=16,
        prompt_cache_bytes=8 * 1024**3,
    )

    tuned = tune_execution_settings(
        settings,
        [
            SimpleNamespace(headroom_bytes=3 * 1024**3),
            SimpleNamespace(headroom_bytes=20 * 1024**3),
        ],
        backend="jaccl",
    )

    assert tuned.decode_concurrency == settings.decode_concurrency
    assert tuned.prompt_cache_size == 1
    assert tuned.prompt_cache_bytes is None
    assert "synchronized single-prefix cache" in tuned.tuning_reason


def test_performance_profiles_reject_nonfinite_measurements():
    payload = _profile("node", 0, 10).to_dict()
    payload["decode_weight_bytes_per_second"] = float("nan")

    with pytest.raises(ValueError, match="finite positive"):
        NodePerformanceProfile.from_dict(payload)


def _deployment() -> ClusterDeployment:
    return ClusterDeployment(
        deployment_id="probe",
        model="org/model",
        backend="ring",
        hosts=(
            ClusterHost("local", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("peer", "peer.local", ("10.0.0.2",)),
        ),
        assignments=(
            PipelineAssignment("local", 0, 2, 4, 20, 0, 0, 100),
            PipelineAssignment("peer", 1, 0, 2, 20, 0, 0, 100),
        ),
        plan_hash="a" * 64,
        execution=replace(
            execution_profile("balanced"),
            ring_connections_per_ip=3,
        ),
    )


def test_cluster_performance_probe_uses_ring_connections_and_validates_ranks():
    def runner(argv, *, timeout, env):
        assert timeout == 12.0
        assert argv[argv.index("--connections-per-ip") + 1] == "3"
        assert "omlx.cluster.performance_worker" in argv
        assert env["SSH_ASKPASS_REQUIRE"] == "never"
        records = [
            {
                "type": "performance_result",
                "rank": rank,
                "size": 2,
                "decode_weight_bytes_per_second": 100 + rank,
                "prefill_weight_bytes_per_second": 200 + rank,
                "collective_latency_seconds": 0.001,
                "collective_bandwidth_bytes_per_second": 10_000,
                "samples": 5,
                "measured_at": "2026-07-26T12:00:00+00:00",
            }
            for rank in (0, 1)
        ]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="\n".join(json.dumps(record) for record in records),
            stderr="",
        )

    report = run_cluster_performance_probe(
        _deployment(),
        timeout=12.0,
        python_executable="/opt/omlx/bin/python",
        runner=runner,
    )

    assert report["ok"] is True
    assert report["connections_per_ip"] == 3
    profiles = performance_profiles_from_records(
        [
            {"type": "noise"},
            *[
                {"type": "performance_result"} | profile
                for profile in report["profiles"]
            ],
        ],
        node_ids=("local", "peer"),
        backend="ring",
    )
    assert [profile.rank for profile in profiles] == [0, 1]


def test_cluster_performance_probe_never_passes_ring_connections_to_jaccl():
    deployment = replace(
        _deployment(),
        backend="jaccl",
        hosts=(
            ClusterHost(
                "local",
                "127.0.0.1",
                ("10.0.0.1",),
                (None, "rdma_en5"),
            ),
            ClusterHost(
                "peer",
                "peer.local",
                ("10.0.0.2",),
                ("rdma_en5", None),
            ),
        ),
    )

    def runner(argv, *, timeout, env):
        assert "--connections-per-ip" not in argv
        records = [
            {
                "type": "performance_result",
                "rank": rank,
                "size": 2,
                "decode_weight_bytes_per_second": 100 + rank,
                "prefill_weight_bytes_per_second": 200 + rank,
                "collective_latency_seconds": 0.001,
                "collective_bandwidth_bytes_per_second": 10_000,
                "samples": 5,
                "measured_at": "2026-07-26T12:00:00+00:00",
            }
            for rank in (0, 1)
        ]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="\n".join(json.dumps(record) for record in records),
            stderr="",
        )

    report = run_cluster_performance_probe(deployment, runner=runner)

    assert report["ok"] is True
    assert report["backend"] == "jaccl"
    assert report["connections_per_ip"] == 1


class _ValidatedPipeline:
    pipeline_rank = 0
    pipeline_size = 2

    def __init__(self):
        self.seen = []

    def __call__(self, value, cache=None):
        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        self.seen.append(value.tolist())
        if pipeline_rank != 0:
            value = mx.distributed.send(
                value,
                (pipeline_rank - 1) % pipeline_size,
            )
        if pipeline_size > 1:
            value = mx.distributed.all_gather(value)
        return value


class _Group:
    @staticmethod
    def rank():
        return 0

    @staticmethod
    def size():
        return 2


class _WorkerGroup:
    @staticmethod
    def rank():
        return 1

    @staticmethod
    def size():
        return 2


def test_asymmetric_tensor_capability_reports_the_local_fraction(monkeypatch):
    monkeypatch.setenv("OMLX_TP_SHARD_WEIGHTS", "3,5")
    with install_runtime_optimizations(
        SimpleNamespace(model=SimpleNamespace()),
        _WorkerGroup(),
        execution_profile("balanced"),
        batchable=False,
        pipeline_parallel=False,
    ) as capabilities:
        item = capabilities["asymmetric_tensor_parallel"]
        assert item["active"] is True
        assert "rank 1 holds 5/8" in item["reason"]


def test_sampling_rank_optimization_is_capability_gated_and_restored():
    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
    )
    model = SimpleNamespace(model=_ValidatedPipeline())
    original_gather = mx.distributed.all_gather
    original_send = mx.distributed.send
    original_call = _ValidatedPipeline.__call__
    original_step = mlx_generate.GenerationBatch._step
    original_prompt = mlx_generate.PromptProcessingBatch.prompt

    with install_runtime_optimizations(
        model,
        _Group(),
        settings,
        batchable=True,
    ) as capabilities:
        assert capabilities["sampling_rank_only"]["active"] is True
        assert capabilities["rank_zero_logits"]["active"] is False
        assert capabilities["pipeline_prefill_overlap"]["active"] is True, (
            capabilities["pipeline_prefill_overlap"]["reason"]
        )
        assert mx.distributed.all_gather is not original_gather
        assert mx.distributed.send is not original_send
        assert _ValidatedPipeline.__call__ is not original_call
        assert mlx_generate.GenerationBatch._step is not original_step
        assert mlx_generate.PromptProcessingBatch.prompt is not original_prompt

    assert mx.distributed.all_gather is original_gather
    assert mx.distributed.send is original_send
    assert _ValidatedPipeline.__call__ is original_call
    assert mlx_generate.GenerationBatch._step is original_step
    assert mlx_generate.PromptProcessingBatch.prompt is original_prompt


def test_worker_rank_skips_vocab_projection_when_adapter_declares_contract(
    monkeypatch,
):
    class Cache:
        state = mx.array([0])

    class RankLocalLogitsModel:
        _omlx_supports_rank_zero_logits = True
        _omlx_output_vocab_size = 32

        def __init__(self):
            self.model = _ValidatedPipeline()
            self.model.pipeline_rank = 1
            self.calls = []

        def __call__(self, value, cache=None, skip_logits=False):
            self.calls.append(skip_logits)
            value = self.model(value, cache=cache)
            if skip_logits:
                return None
            return mx.zeros((*value.shape, self._omlx_output_vocab_size))

    class Batch:
        def __init__(self, model):
            self.model = model
            self.uids = [1]
            self.prompt_cache = [Cache()]
            self.tokens = [[]]
            self.samplers = [None]
            self.fallback_sampler = lambda value: mx.argmax(value, axis=-1)
            self.logits_processors = [[]]
            self.state_machines = []
            self.max_tokens = [2]
            self._current_tokens = None
            self._current_logprobs = []
            self._next_tokens = mx.array([3], dtype=mx.uint32)
            self._next_logprobs = []
            self._token_context = []
            self._num_tokens = [0]
            self._matcher_states = []

    model = RankLocalLogitsModel()
    batch = Batch(model)
    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
    )
    monkeypatch.setattr(
        mx.distributed,
        "all_sum",
        lambda value, group=None: value,
    )
    monkeypatch.setattr(mx.distributed, "send", lambda value, *_a, **_k: value)
    monkeypatch.setattr(mx.distributed, "all_gather", lambda value, **_k: value)
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)

    with install_runtime_optimizations(
        model,
        _WorkerGroup(),
        settings,
        batchable=True,
    ) as capabilities:
        assert capabilities["rank_zero_logits"]["active"] is True
        mlx_generate.GenerationBatch._step(batch)

    assert model.calls == [True]
    assert len(batch._next_logprobs) == 1
    assert batch._next_logprobs[0].shape == (32,)


def test_pipeline_prefill_schedule_has_equal_fill_and_drain_timeline():
    schedules = [
        pipeline_prefill_schedule(10, 4, rank=rank, world_size=3)
        for rank in range(3)
    ]

    assert {len(schedule) for schedule in schedules} == {5}
    # MLX-LM runs the first stage on the highest rank and the final stage on
    # rank zero, so the Exo fill/drain offset is mirrored.
    assert [(slot.start, slot.end) for slot in schedules[0]] == [
        (None, None),
        (None, None),
        (0, 4),
        (4, 8),
        (8, 10),
    ]
    assert [(slot.start, slot.end) for slot in schedules[2]] == [
        (0, 4),
        (4, 8),
        (8, 10),
        (None, None),
        (None, None),
    ]
    assert all(sum(slot.is_real for slot in schedule) == 3 for schedule in schedules)


def test_staggered_prompt_queues_and_flushes_every_real_chunk(monkeypatch):
    sends = []
    gathers = []
    async_values = []
    original_prompt = mlx_generate.PromptProcessingBatch.prompt

    monkeypatch.setattr(
        mx.distributed,
        "send",
        lambda value, destination, **kwargs: sends.append(destination) or value,
    )
    monkeypatch.setattr(
        mx.distributed,
        "all_gather",
        lambda value, **kwargs: gathers.append(value) or value,
    )
    monkeypatch.setattr(mx, "async_eval", lambda *values: async_values.extend(values))

    class Cache:
        state = mx.array([0])

    class Batch:
        uids = ["request"]
        tokens = [[]]
        prompt_cache = [Cache()]
        prefill_step_size = 8

        def __init__(self):
            self.model = _ValidatedPipeline()
            self.model.pipeline_rank = 1

    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
        async_overlap=True,
        prefill_step_size=8,
    )
    model = SimpleNamespace(model=_ValidatedPipeline())
    batch = Batch()

    with install_runtime_optimizations(
        model,
        _Group(),
        settings,
        batchable=True,
    ) as capabilities:
        assert capabilities["pipeline_prefill_overlap"]["active"] is True, (
            capabilities["pipeline_prefill_overlap"]["reason"]
        )
        mlx_generate.PromptProcessingBatch.prompt(batch, [list(range(9))])

    # The scheduler honours the same eight-token step the memory guard approved,
    # so 9 tokens make two real chunks. Each chunk reaches send; the final
    # hidden-state gather is skipped.
    assert sends == [0, 0]
    assert len(async_values) == 2
    assert gathers == []
    assert batch.tokens == [list(range(9))]
    assert mlx_generate.PromptProcessingBatch.prompt is original_prompt


def test_staggered_prompt_matches_stock_chunking_padding_and_cache_lifecycle(
    monkeypatch,
):
    """The faster scheduler must preserve MLX-LM's prompt/cache contract."""

    original_prompt = mlx_generate.PromptProcessingBatch.prompt
    monkeypatch.setattr(mx.distributed, "send", lambda value, *_a, **_k: value)
    monkeypatch.setattr(mx.distributed, "all_gather", lambda value, **_k: value)
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)

    class Cache:
        def __init__(self):
            self.state = mx.array([0])
            self.events = []

        def prepare(self, *, lengths, right_padding):
            self.events.append(("prepare", tuple(lengths), tuple(right_padding)))

        def finalize(self):
            self.events.append(("finalize",))

    class Batch:
        uids = ["first", "second"]
        prefill_step_size = 8

        def __init__(self):
            self.tokens = [[], []]
            self.prompt_cache = [Cache()]
            self.model = _ValidatedPipeline()
            self.model.pipeline_rank = 1

    prompts = [list(range(9)), list(range(20, 25))]
    stock = Batch()
    original_prompt(stock, [list(prompt) for prompt in prompts])

    patched = Batch()
    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
        async_overlap=True,
        prefill_step_size=8,
    )
    with install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        _Group(),
        settings,
        batchable=True,
    ):
        mlx_generate.PromptProcessingBatch.prompt(
            patched,
            [list(prompt) for prompt in prompts],
        )

    assert patched.model.seen == stock.model.seen
    assert [len(chunk[0]) for chunk in patched.model.seen] == [8, 1]
    assert patched.tokens == stock.tokens == prompts
    assert patched.prompt_cache[0].events == stock.prompt_cache[0].events


def test_sampling_rank_optimization_keeps_normal_path_for_unvalidated_model():
    settings = replace(
        execution_profile("interactive"),
        sampling_rank_only=True,
    )
    model = SimpleNamespace(model=SimpleNamespace())
    original_gather = mx.distributed.all_gather

    with install_runtime_optimizations(
        model,
        _Group(),
        settings,
        batchable=True,
    ) as capabilities:
        assert capabilities["sampling_rank_only"]["active"] is False
        assert capabilities["pipeline_prefill_overlap"]["active"] is False
        assert mx.distributed.all_gather is original_gather


def test_tensor_prefill_skips_discarded_vocab_projection(monkeypatch):
    class Cache:
        state = mx.array([0])

    class SkipHeadModel:
        def __init__(self):
            self.calls = []

        def __call__(self, value, cache=None, skip_lm_head=False):
            self.calls.append((value.shape[1], skip_lm_head))
            return None if skip_lm_head else mx.zeros((*value.shape, 32))

    class Batch:
        uids = ["request"]
        prefill_step_size = 4

        def __init__(self, model):
            self.model = model
            self.tokens = [[]]
            self.prompt_cache = [Cache()]

    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    model = SkipHeadModel()
    batch = Batch(model)
    original_prompt = mlx_generate.PromptProcessingBatch.prompt

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        assert capabilities["prefill_logits_skip"]["active"] is True
        assert capabilities["sampling_rank_only"]["active"] is False
        mlx_generate.PromptProcessingBatch.prompt(batch, [list(range(9))])

    assert model.calls == [(4, True), (4, True), (1, True)]
    assert batch.tokens == [list(range(9))]
    assert mlx_generate.PromptProcessingBatch.prompt is original_prompt


def test_tensor_prefill_reuses_allocator_cache_until_prompt_end(monkeypatch):
    class Cache:
        state = mx.array([0])

    class Model:
        def __call__(self, value, cache=None, skip_lm_head=False):
            assert skip_lm_head is True
            return None

    class Batch:
        uids = ["request"]
        prefill_step_size = 4

        def __init__(self, model):
            self.model = model
            self.tokens = [[]]
            self.prompt_cache = [Cache()]

    clears = []
    monkeypatch.delenv("OMLX_CLUSTER_PREFILL_CLEAR_CACHE_EVERY", raising=False)
    monkeypatch.setattr(mx, "clear_cache", lambda: clears.append(True))
    model = Model()
    batch = Batch(model)

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        assert capabilities["prefill_allocator_reuse"]["active"] is True
        mlx_generate.PromptProcessingBatch.prompt(batch, [list(range(9))])

    assert clears == [True]


def test_prefill_allocator_cache_cadence_is_operator_overridable(monkeypatch):
    class Cache:
        state = mx.array([0])

    class Model:
        def __call__(self, value, cache=None, skip_lm_head=False):
            return None

    class Batch:
        uids = ["request"]
        prefill_step_size = 4

        def __init__(self, model):
            self.model = model
            self.tokens = [[]]
            self.prompt_cache = [Cache()]

    clears = []
    monkeypatch.setenv("OMLX_CLUSTER_PREFILL_CLEAR_CACHE_EVERY", "2")
    monkeypatch.setattr(mx, "clear_cache", lambda: clears.append(True))
    model = Model()

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ):
        mlx_generate.PromptProcessingBatch.prompt(
            Batch(model),
            [list(range(9))],
        )

    # Three chunks: purge after chunk 2 and unconditionally after the final one.
    assert clears == [True, True]


def test_tensor_vocab_sampling_reconstructs_logits_only_on_rank_zero(monkeypatch):
    class Head:
        _omlx_vocab_parallel = True
        _omlx_output_dims = 6
        _omlx_gather_vocab_logits = True
        weight = mx.zeros((3, 2))

    class Model:
        def __init__(self):
            self.lm_head = Head()

        def __call__(self, value, cache=None):
            assert self.lm_head._omlx_gather_vocab_logits is False
            return mx.array([[[1.0, 2.0, 3.0]]])

    class Batch:
        def __init__(self, model):
            self.model = model
            self.uids = [1]
            self.prompt_cache = []
            self.tokens = [[]]
            self.samplers = [None]
            self.fallback_sampler = lambda value: mx.argmax(value, axis=-1)
            self.logits_processors = [[]]
            self._current_tokens = None
            self._current_logprobs = []
            self._next_tokens = mx.array([3], dtype=mx.uint32)
            self._next_logprobs = []
            self._token_context = []

    model = Model()
    batch = Batch(model)
    monkeypatch.setattr(
        mx.distributed,
        "recv_like",
        lambda value, source: mx.array([[4.0, 5.0, 9.0]]),
    )
    monkeypatch.setattr(mx.distributed, "all_sum", lambda value, group=None: value)
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        assert capabilities["sampling_rank_only"]["active"] is True
        assert capabilities["vocab_parallel_sampling"]["active"] is True
        assert capabilities["rank_zero_logits"]["active"] is False
        mlx_generate.GenerationBatch._step(batch)

    assert batch._next_tokens.tolist() == [5]
    assert batch._next_logprobs[0].shape == (6,)
    assert model.lm_head._omlx_gather_vocab_logits is True


def test_tensor_vocab_sampling_worker_sends_local_shard_and_uses_rank_zero_token(
    monkeypatch,
):
    class Head:
        _omlx_vocab_parallel = True
        _omlx_output_dims = 6
        _omlx_gather_vocab_logits = True
        weight = mx.zeros((3, 2))

    class Model:
        def __init__(self):
            self.lm_head = Head()

        def __call__(self, value, cache=None):
            return mx.array([[[7.0, 8.0, 9.0]]])

    class Batch:
        def __init__(self, model):
            self.model = model
            self.uids = [1]
            self.prompt_cache = []
            self.tokens = [[]]
            self.samplers = [None]
            self.fallback_sampler = lambda value: mx.argmax(value, axis=-1)
            self.logits_processors = [[]]
            self._current_tokens = None
            self._current_logprobs = []
            self._next_tokens = mx.array([3], dtype=mx.uint32)
            self._next_logprobs = []
            self._token_context = []

    sent = []
    model = Model()
    batch = Batch(model)
    monkeypatch.setattr(
        mx.distributed,
        "send",
        lambda value, destination: sent.append((value.tolist(), destination)) or value,
    )
    monkeypatch.setattr(
        mx.distributed,
        "all_sum",
        lambda value, group=None: mx.array([5], dtype=mx.uint32),
    )
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)

    with install_runtime_optimizations(
        model,
        _WorkerGroup(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ):
        mlx_generate.GenerationBatch._step(batch)

    assert sent == [([[7.0, 8.0, 9.0]], 0)]
    assert batch._next_tokens.tolist() == [5]
    assert batch._next_logprobs[0].shape == (6,)


def test_tensor_vocab_sampling_stays_synchronized_for_mtp_model():
    head = SimpleNamespace(
        _omlx_vocab_parallel=True,
        _omlx_output_dims=6,
        weight=mx.zeros((3, 2)),
    )
    model = SimpleNamespace(
        lm_head=head,
        _omlx_mtp_decode_enabled=True,
    )

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        assert capabilities["sampling_rank_only"]["active"] is False
        assert capabilities["vocab_parallel_sampling"]["active"] is False
        assert "MTP" in capabilities["vocab_parallel_sampling"]["reason"]


def test_tensor_vocab_sampling_installs_validated_mtp_coordinator():
    head = SimpleNamespace(
        _omlx_vocab_parallel=True,
        _omlx_output_dims=6,
        _omlx_gather_vocab_logits=True,
        weight=mx.zeros((3, 2)),
    )
    auxiliary = SimpleNamespace(
        _omlx_vocab_parallel=True,
        _omlx_output_dims=6,
        _omlx_gather_vocab_logits=True,
        weight=mx.zeros((3, 2)),
    )
    model = SimpleNamespace(
        lm_head=head,
        _omlx_mtp_decode_enabled=True,
        _omlx_mtp_chain=True,
        _omlx_distributed_mtp_vocab_ready=True,
        _omlx_vocab_parallel_aux_heads=(auxiliary,),
    )

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        assert capabilities["sampling_rank_only"]["active"] is True
        assert capabilities["vocab_parallel_sampling"]["active"] is True
        assert "MTP" in capabilities["vocab_parallel_sampling"]["reason"]
        coordinator = model._omlx_mtp_vocab_coordinator
        assert coordinator.is_coordinator is True
        assert coordinator.output_size == 6
        assert head._omlx_gather_vocab_logits is False
        assert auxiliary._omlx_gather_vocab_logits is False

    assert not hasattr(model, "_omlx_mtp_vocab_coordinator")
    assert head._omlx_gather_vocab_logits is True
    assert auxiliary._omlx_gather_vocab_logits is True


def test_non_batchable_model_never_reports_continuous_batching_active():
    settings = execution_profile("balanced")
    model = SimpleNamespace(model=SimpleNamespace())

    with install_runtime_optimizations(
        model,
        _Group(),
        settings,
        batchable=False,
    ) as capabilities:
        batching = capabilities["coalesced_batching"]
        assert batching["enabled"] is True
        assert batching["active"] is False
        assert "sequentially" in batching["reason"]
