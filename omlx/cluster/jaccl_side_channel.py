# SPDX-License-Identifier: Apache-2.0
"""Reliable JACCL metadata exchange for distributed oMLX workers.

JACCL uses a tiny TCP all-gather only while it creates its RDMA queue pairs.
The native helper is normally sufficient, but a direct Thunderbolt interface
can occasionally leave its client bootstrap sockets in ``SYN_SENT`` even
though ordinary TCP on the same interface succeeds.  MLX exposes a supported
``all_gather_factory`` specifically for replacing that bootstrap path.  This
module supplies a bounded, ordered implementation without changing the RDMA
data path used after initialization.
"""

from __future__ import annotations

import os
import socket
import struct
import sys
import threading
import time
from collections.abc import Callable


_RANK_BYTES = struct.Struct("!I")
_DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS = 20.0
_DEFAULT_CONNECT_RETRY_SECONDS = 0.05


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _enabled() -> bool:
    """Whether to replace JACCL's native metadata side channel.

    This is default-on for oMLX JACCL jobs, with an escape hatch for precise
    upstream comparison and incident recovery.
    """

    return os.environ.get("OMLX_JACCL_PYTHON_SIDE_CHANNEL", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _coordinator_endpoint() -> tuple[str, int]:
    raw = os.environ.get("MLX_JACCL_COORDINATOR", "").strip()
    host, separator, port_text = raw.rpartition(":")
    if not separator or not host or not port_text.isdecimal():
        raise RuntimeError(
            "JACCL side channel requires MLX_JACCL_COORDINATOR as IPv4:port"
        )
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise RuntimeError("JACCL side channel coordinator port is invalid")
    try:
        socket.inet_aton(host)
    except OSError as exc:
        raise RuntimeError(
            "JACCL side channel currently requires an IPv4 coordinator"
        ) from exc
    return host, port


def _recv_exact(sock: socket.socket, n_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n_bytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("JACCL side channel peer disconnected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_all(sock: socket.socket, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = sock.send(view)
        if written <= 0:
            raise RuntimeError("JACCL side channel peer rejected metadata")
        view = view[written:]


def _trace(message: str) -> None:
    if os.environ.get("OMLX_JACCL_SIDE_CHANNEL_TRACE", "0") == "1":
        print(f"OMLX_JACCL_SIDE_CHANNEL {message}", file=sys.stderr, flush=True)


class _SocketAllGather:
    """One rank's persistent, serialized bootstrap TCP connection set."""

    def __init__(self, rank: int, size: int) -> None:
        if not 0 <= rank < size or size < 2:
            raise RuntimeError("JACCL side channel received invalid rank metadata")
        self.rank = rank
        self.size = size
        self.timeout = _positive_float(
            "OMLX_JACCL_SIDE_CHANNEL_TIMEOUT_SECONDS",
            _DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS,
        )
        self._lock = threading.Lock()
        self._peers: list[socket.socket] | socket.socket
        host, port = _coordinator_endpoint()
        if rank == 0:
            self._peers = self._accept_peers(host, port)
        else:
            self._peers = self._connect(host, port)
        _trace(f"ready rank={rank} size={size} coordinator={host}:{port}")

    def _accept_peers(self, host: str, port: int) -> list[socket.socket]:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((host, port))
            server.listen(self.size)
            server.settimeout(self.timeout)
            peers: list[socket.socket | None] = [None] * self.size
            deadline = time.monotonic() + self.timeout
            while any(peer is None for peer in peers[1:]):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "JACCL side channel timed out waiting for every rank"
                    )
                server.settimeout(remaining)
                peer, _address = server.accept()
                try:
                    peer.settimeout(self.timeout)
                    peer_rank = _RANK_BYTES.unpack(_recv_exact(peer, _RANK_BYTES.size))[0]
                    if not 1 <= peer_rank < self.size or peers[peer_rank] is not None:
                        raise RuntimeError("JACCL side channel received an invalid peer rank")
                except BaseException:
                    peer.close()
                    raise
                peers[peer_rank] = peer
            return [peer for peer in peers[1:] if peer is not None]
        except BaseException:
            for peer in locals().get("peers", []):
                if peer is not None:
                    peer.close()
            raise
        finally:
            server.close()

    def _connect(self, host: str, port: int) -> socket.socket:
        deadline = time.monotonic() + self.timeout
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            peer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                peer.settimeout(min(1.0, max(0.05, deadline - time.monotonic())))
                peer.connect((host, port))
                peer.settimeout(self.timeout)
                _send_all(peer, _RANK_BYTES.pack(self.rank))
                return peer
            except OSError as exc:
                last_error = exc
                peer.close()
                time.sleep(_DEFAULT_CONNECT_RETRY_SECONDS)
        detail = f": {last_error}" if last_error else ""
        raise RuntimeError(f"JACCL side channel could not reach coordinator{detail}")

    def __call__(self, src: bytes, n_bytes: int) -> bytes:
        if n_bytes < 0 or len(src) != n_bytes:
            raise RuntimeError("JACCL side channel received malformed metadata")
        with self._lock:
            if self.rank == 0:
                assert isinstance(self._peers, list)
                gathered = [src]
                for peer in self._peers:
                    gathered.append(_recv_exact(peer, n_bytes))
                result = b"".join(gathered)
                for peer in self._peers:
                    _send_all(peer, result)
                return result
            assert isinstance(self._peers, socket.socket)
            _send_all(self._peers, src)
            return _recv_exact(self._peers, self.size * n_bytes)


def jaccl_all_gather_factory(rank: int, size: int) -> Callable[[bytes, int], bytes]:
    """Return JACCL's supported metadata all-gather callable for one rank."""

    return _SocketAllGather(rank, size)


def init_cluster_group(
    mx: object,
    *,
    backend: str | None = None,
    strict: bool = False,
) -> object:
    """Initialize a distributed group with the resilient JACCL bootstrap.

    Ring/NCCL execution remains byte-for-byte on MLX's normal path.  For a
    JACCL worker, only connection metadata travels through this small TCP
    control channel; collectives immediately thereafter remain RDMA/JACCL.
    """

    distributed = getattr(mx, "distributed")
    selected = backend
    if selected is None and os.environ.get("MLX_JACCL_COORDINATOR"):
        selected = "jaccl"
    if selected == "jaccl" and _enabled():
        return distributed.init(
            backend="jaccl",
            strict=strict,
            all_gather_factory=jaccl_all_gather_factory,
        )
    if selected is None:
        return distributed.init(strict=strict)
    return distributed.init(backend=selected, strict=strict)
