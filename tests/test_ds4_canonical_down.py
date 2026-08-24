"""Exact activation-boundary contract for heterogeneous DS4 routed MoE."""

from __future__ import annotations

import sys

import mlx.core as mx

from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

apply_deepseek_v4_patch()
switch = sys.modules["omlx.patches.deepseek_v4.switch_layers"]


class _Group:
    def __init__(self, rank):
        self._rank = rank

    def rank(self):
        return self._rank

    def size(self):
        return 2


def _module(rank):
    module = switch.SwitchGLU(8, 8, 2)
    module._omlx_dsv4f_canonical_down = True
    module._omlx_dsv4f_moe_tp = (2, rank, (3, 5))
    module._omlx_dsv4f_global_intermediate = 2048
    module.sharding_group = _Group(rank)
    return module


def test_rank_one_sends_middle_segment_and_keeps_canonical_upper_half(monkeypatch):
    full = mx.arange(2048, dtype=mx.float32).reshape(1, 1, 2048)
    local = full[..., 768:]
    sent = []
    monkeypatch.setattr(
        switch.mx.distributed,
        "send",
        lambda value, peer, group=None: sent.append((value, peer, group)) or value,
    )

    canonical = _module(1)._canonical_down_input(local)
    mx.eval(canonical, sent[0][0])

    assert sent[0][1] == 0
    assert mx.array_equal(sent[0][0], full[..., 768:1024]).item()
    assert mx.array_equal(canonical, full[..., 1024:]).item()


def test_rank_zero_receives_middle_segment_and_rebuilds_canonical_lower_half(
    monkeypatch,
):
    full = mx.arange(2048, dtype=mx.float32).reshape(1, 1, 2048)
    local = full[..., :768]
    monkeypatch.setattr(
        switch.mx.distributed,
        "recv_like",
        lambda template, peer, group=None: full[..., 768:1024],
    )

    canonical = _module(0)._canonical_down_input(local)
    mx.eval(canonical)

    assert mx.array_equal(canonical, full[..., :1024]).item()
