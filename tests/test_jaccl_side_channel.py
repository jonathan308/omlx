# SPDX-License-Identifier: Apache-2.0
"""Tests for the bounded Python control plane used to bootstrap JACCL."""

from __future__ import annotations

import socket
import threading

import pytest

from omlx.cluster.jaccl_side_channel import (
    _coordinator_endpoint,
    init_cluster_group,
    jaccl_all_gather_factory,
)


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_socket_side_channel_orders_ranks_and_reuses_connections(monkeypatch):
    port = _free_loopback_port()
    monkeypatch.setenv("MLX_JACCL_COORDINATOR", f"127.0.0.1:{port}")
    monkeypatch.setenv("OMLX_JACCL_SIDE_CHANNEL_TIMEOUT_SECONDS", "3")
    results: dict[str, bytes] = {}
    errors: list[BaseException] = []
    server_ready = threading.Event()

    def rank_zero() -> None:
        try:
            gather = jaccl_all_gather_factory(0, 2)
            server_ready.set()
            results["first_server"] = gather(b"aa", 2)
            results["second_server"] = gather(b"cc", 2)
        except BaseException as exc:  # propagate thread failures to pytest
            errors.append(exc)
            server_ready.set()

    thread = threading.Thread(target=rank_zero)
    thread.start()
    gather = jaccl_all_gather_factory(1, 2)
    assert server_ready.wait(3)
    results["first_client"] = gather(b"bb", 2)
    results["second_client"] = gather(b"dd", 2)
    thread.join(3)

    assert not thread.is_alive()
    assert not errors
    assert results == {
        "first_server": b"aabb",
        "first_client": b"aabb",
        "second_server": b"ccdd",
        "second_client": b"ccdd",
    }


@pytest.mark.parametrize("value", ["", "host", "127.0.0.1:0", "[::1]:8000"])
def test_side_channel_requires_a_valid_ipv4_coordinator(monkeypatch, value):
    monkeypatch.setenv("MLX_JACCL_COORDINATOR", value)
    with pytest.raises(RuntimeError, match="coordinator|IPv4"):
        _coordinator_endpoint()


def test_init_cluster_group_only_injects_factory_for_enabled_jaccl(monkeypatch):
    calls: list[dict[str, object]] = []

    class Distributed:
        def init(self, **kwargs):
            calls.append(kwargs)
            return "group"

    class MX:
        distributed = Distributed()

    monkeypatch.setenv("MLX_JACCL_COORDINATOR", "10.0.0.1:8000")
    assert init_cluster_group(MX(), backend="jaccl", strict=True) == "group"
    assert calls[-1]["backend"] == "jaccl"
    assert calls[-1]["strict"] is True
    assert calls[-1]["all_gather_factory"] is jaccl_all_gather_factory

    monkeypatch.setenv("OMLX_JACCL_PYTHON_SIDE_CHANNEL", "0")
    assert init_cluster_group(MX(), backend="jaccl", strict=True) == "group"
    assert calls[-1] == {"backend": "jaccl", "strict": True}

    assert init_cluster_group(MX(), backend="ring", strict=False) == "group"
    assert calls[-1] == {"backend": "ring", "strict": False}
