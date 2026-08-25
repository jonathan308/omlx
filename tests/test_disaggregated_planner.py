from __future__ import annotations

import pytest

from omlx.cluster.disaggregated import (
    DisaggregatedNodeProfile,
    DisaggregatedWorkload,
    plan_disaggregated_prefill_decode,
)


def _node(name, *, budget=64_000, resident=16_000, prefill, decode):
    return DisaggregatedNodeProfile(
        node_id=name,
        admission_budget_bytes=budget,
        model_resident_bytes=resident,
        prefill_tokens_per_second=prefill,
        decode_tokens_per_second=decode,
    )


def _workload(*, cache=422_379_520, fabric=6.52e9):
    return DisaggregatedWorkload(
        prompt_tokens=4096,
        completion_tokens=64,
        cache_transfer_bytes=cache,
        fabric_bytes_per_second=fabric,
    )


def test_planner_chooses_measured_m5_prefill_m3_decode_orientation():
    m3 = _node("m3", prefill=700.0, decode=31.2)
    m5 = _node("m5", prefill=1005.7, decode=31.3)
    plan = plan_disaggregated_prefill_decode([m3, m5], _workload())

    assert plan.supported is True
    assert plan.recommended is True
    assert plan.orientation.prefill_node_id == "m5"
    assert plan.orientation.decode_node_id == "m3"
    assert plan.estimated_steady_speedup > 1.4


def test_planner_rejects_model_that_does_not_fit_both_replicas():
    m3 = _node("m3", budget=256_000, resident=171_000, prefill=800, decode=30)
    m5 = _node("m5", budget=128_000, resident=171_000, prefill=1000, decode=30)
    plan = plan_disaggregated_prefill_decode([m3, m5], _workload())

    assert plan.supported is False
    assert plan.recommended is False
    assert "m5" in plan.reason


def test_planner_keeps_single_node_when_handoff_dominates():
    nodes = [
        _node("a", prefill=1000, decode=100),
        _node("b", prefill=1000, decode=100),
    ]
    plan = plan_disaggregated_prefill_decode(
        nodes,
        _workload(cache=20_000_000_000, fabric=1e9),
    )

    assert plan.supported is True
    assert plan.recommended is False
    assert plan.estimated_steady_speedup < 1.0


def test_planner_validates_profile_rates_and_threshold():
    with pytest.raises(ValueError, match="phase rates"):
        _node("bad", prefill=0, decode=1)
    nodes = [
        _node("a", prefill=1000, decode=100),
        _node("b", prefill=1000, decode=100),
    ]
    with pytest.raises(ValueError, match="minimum steady speedup"):
        plan_disaggregated_prefill_decode(
            nodes, _workload(), minimum_steady_speedup=0.99
        )
