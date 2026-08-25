from __future__ import annotations

import sys
from contextlib import contextmanager

import mlx.core as mx

from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch


apply_deepseek_v4_patch()
dm = sys.modules["mlx_lm.models.deepseek_v4"]


class _Group:
    def size(self):
        return 2


def _moe(rows):
    module = dm.DeepseekV4MoE.__new__(dm.DeepseekV4MoE)
    dm.nn.Module.__init__(module)
    module.eval()
    module.sharding_group = _Group()
    module.gate = lambda x, _ids: (
        mx.zeros((1, rows, 6), dtype=mx.int32),
        mx.ones((1, rows, 6), dtype=mx.float32),
    )
    module.switch_mlp = lambda x, _inds, scores=None: mx.zeros_like(x)
    module.shared_experts = lambda x: x + mx.array(1, dtype=x.dtype)
    return module


def test_overlap_preserves_canonical_sum_and_uses_secondary_stream(monkeypatch):
    monkeypatch.setattr(dm, "_DEEPSEEK_V4_SHARED_EXPERT_OVERLAP", True)
    monkeypatch.setattr(dm, "is_dspark_verify_armed", lambda: False)
    monkeypatch.setattr(dm, "sum_gradients", lambda _group: lambda x: x)
    monkeypatch.setattr(dm.mx.distributed, "all_sum", lambda x, group=None: x)
    entered = []

    @contextmanager
    def stream_context(stream):
        entered.append(stream)
        yield

    secondary = object()
    synchronized = []
    monkeypatch.setattr(dm, "_shared_expert_overlap_stream", lambda: secondary)
    monkeypatch.setattr(dm.mx, "stream", stream_context)
    monkeypatch.setattr(dm.mx, "synchronize", lambda stream: synchronized.append(stream))
    value = mx.zeros((1, 1024, 4096), dtype=mx.bfloat16)

    result = dm.DeepseekV4MoE.__call__(_moe(1024), value, value[:, :, 0])
    mx.eval(result)

    assert entered == [secondary]
    assert synchronized == [secondary]
    assert mx.array_equal(result, mx.ones_like(value)).item()


def test_overlap_fails_closed_outside_exact_prefill_shape(monkeypatch):
    monkeypatch.setattr(dm, "_DEEPSEEK_V4_SHARED_EXPERT_OVERLAP", True)
    monkeypatch.setattr(dm, "is_dspark_verify_armed", lambda: False)
    monkeypatch.setattr(dm, "sum_gradients", lambda _group: lambda x: x)
    monkeypatch.setattr(dm.mx.distributed, "all_sum", lambda x, group=None: x)
    monkeypatch.setattr(
        dm,
        "_shared_expert_overlap_stream",
        lambda: (_ for _ in ()).throw(AssertionError("must not create stream")),
    )
    value = mx.zeros((1, 32, 4096), dtype=mx.bfloat16)

    result = dm.DeepseekV4MoE.__call__(_moe(32), value, value[:, :, 0])
    mx.eval(result)

    assert mx.array_equal(result, mx.ones_like(value)).item()


def test_cluster_hostfile_propagates_overlap_switch(monkeypatch):
    from omlx.cluster import deployment

    monkeypatch.delenv("OMLX_DSV4_SHARED_EXPERT_OVERLAP", raising=False)
    assert "OMLX_DSV4_SHARED_EXPERT_OVERLAP=0" in deployment._hostfile_envs()
    monkeypatch.setenv("OMLX_DSV4_SHARED_EXPERT_OVERLAP", "1")
    assert "OMLX_DSV4_SHARED_EXPERT_OVERLAP=1" in deployment._hostfile_envs()
