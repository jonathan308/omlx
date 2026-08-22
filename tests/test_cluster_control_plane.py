# SPDX-License-Identifier: Apache-2.0
"""Reliable TCP control plane kept independent of MLX/JACCL collectives."""

import socket
import threading

from omlx.cluster.control_plane import RankControlPlane


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def test_rank_control_plane_broadcasts_objects_in_sequence():
    port = _free_port()
    token = "a" * 64
    expected = [None, ("request", {"max_tokens": 7}), [], [3, 9]]
    received = []
    failures = []

    def coordinator():
        try:
            with RankControlPlane(
                rank=0,
                world_size=2,
                host="127.0.0.1",
                port=port,
                token=token,
                connect_timeout=5,
                io_timeout=5,
            ) as control:
                for value in expected:
                    assert control.broadcast_object(value) is value
        except Exception as exc:  # pragma: no cover - relayed to main thread
            failures.append(exc)

    thread = threading.Thread(target=coordinator)
    thread.start()
    try:
        with RankControlPlane(
            rank=1,
            world_size=2,
            host="127.0.0.1",
            port=port,
            token=token,
            connect_timeout=5,
            io_timeout=5,
        ) as control:
            for _ in expected:
                received.append(control.broadcast_object(None))
    finally:
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert failures == []
    assert received == expected


def test_rank_control_plane_rejects_invalid_identity():
    try:
        RankControlPlane(
            rank=2,
            world_size=2,
            host="127.0.0.1",
            port=12345,
            token="x",
        )
    except ValueError as exc:
        assert "identity" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid rank identity was accepted")
