# SPDX-License-Identifier: Apache-2.0
"""Cluster v2 discovery/identity HTTP endpoints.

``discovery_router`` is mounted WITHOUT ``require_admin`` in
``omlx/server.py`` so peers can probe ``/api/cluster/node_id`` before any
trust exists; it is still gated by the distributed-inference exposure flag
(``require_distributed_inference_enabled``) like the enrollment router. The
probe endpoint is rate-limited per source IP. ``/api/cluster/devices``
carries the trusted inventory and requires admin like the rest of the
cluster surface.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from .._version import __version__
from ..admin.auth import require_admin
from .identity import get_node_identity
from .registry import get_device_registry

discovery_router = APIRouter(prefix="/api/cluster", tags=["cluster-discovery"])

DEFAULT_CLUSTER_NAME = "omlx"


class ProbeRateLimiter:
    """Token-bucket rate limiter for the unauthenticated probe endpoint."""

    def __init__(self, rate_per_second: float = 5.0, burst: int = 10) -> None:
        self._rate = float(rate_per_second)
        self._burst = int(burst)
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, updated = self._buckets.get(key, (float(self._burst), now))
            tokens = min(
                float(self._burst), tokens + (now - updated) * self._rate
            )
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            # Bound the map: drop idle buckets once it grows past 4096 keys.
            if len(self._buckets) > 4096:
                self._buckets = {
                    k: v
                    for k, v in self._buckets.items()
                    if now - v[1] < 600.0
                }
            return True


probe_rate_limiter = ProbeRateLimiter()


def discovery_service_or_none():
    from .discovery import get_discovery_service

    try:
        return get_discovery_service()
    except RuntimeError:
        return None


def _cluster_name() -> str:
    service = discovery_service_or_none()
    if service is not None:
        return service.config.cluster_name
    return DEFAULT_CLUSTER_NAME


@discovery_router.get("/node_id")
async def cluster_node_id_probe(request: Request):
    """Public, rate-limited identity probe used by peer verification.

    Deliberately unauthenticated: a discovering node must be able to confirm
    an announced address belongs to the announced node_id before any pairing
    trust exists. It reveals only the stable node_id, the oMLX version, and
    the cluster name — no capabilities, no device inventory.
    """

    client = request.client.host if request.client else "unknown"
    if not probe_rate_limiter.allow(client):
        raise HTTPException(
            status_code=429,
            detail="probe rate limit exceeded",
            headers={"Retry-After": "1"},
        )
    try:
        identity = get_node_identity()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="cluster identity is not configured"
        ) from exc
    return {
        "node_id": identity.node_id,
        "version": __version__,
        "cluster_name": _cluster_name(),
    }


@discovery_router.get("/discovery/health")
async def cluster_discovery_health(is_admin: bool = Depends(require_admin)):
    """Local-network self-test for the wizard's checks row.

    ``multicast_rx_within_5s`` is False when no foreign HELLO arrived in the
    last five seconds — on macOS that is how a denied Local Network
    permission presents, so the UI pairs it with actionable guidance instead
    of a silently empty device list. Shape is pinned by the wizard fixture
    (tests/ui/fixtures/cluster_v2/discovery_health_ok.json).
    """

    service = discovery_service_or_none()
    if service is None:
        return {
            "multicast_rx_within_5s": False,
            "last_multicast_rx_at": None,
            "mdns_active": False,
            "transport": "disabled",
        }
    return {
        "multicast_rx_within_5s": bool(service.multicast_ok),
        "last_multicast_rx_at": service.last_multicast_rx_at,
        "mdns_active": bool(service.mdns_active),
        "transport": service.transport_summary(),
    }


@discovery_router.get("/devices")
async def cluster_devices(is_admin: bool = Depends(require_admin)):
    """Cluster device inventory for the wizard UI.

    ``multicast_ok`` is the macOS Local Network permission signal: it is
    False when no foreign HELLO was received in the last 5 seconds, which is
    how a denied/blocked local-network permission presents. The UI should
    surface "check System Settings → Privacy & Security → Local Network"
    rather than showing a silently empty device list.
    """

    try:
        identity = get_node_identity()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="cluster identity is not configured"
        ) from exc

    try:
        registry = get_device_registry()
    except RuntimeError:
        registry = None
    service = discovery_service_or_none()

    paired: list[dict[str, Any]] = registry.paired() if registry else []

    # Seam with Module B's enrollment: a paired device that completed SSH
    # TOFU enrollment carries its enrolled ssh_target, so the UI probes and
    # plans against the enrolled target instead of guessing from probe IPs.
    try:
        from .enrollment import get_cluster_enrollment

        enrollment = get_cluster_enrollment()
    except RuntimeError:
        enrollment = None
    if enrollment is not None:
        enrolled = {node.node_id: node for node in enrollment.list_nodes()}
        for row in paired:
            node = enrolled.get(row.get("node_id"))
            if node is not None:
                row["ssh_target"] = node.ssh
    discovered_records: dict[str, dict[str, Any]] = {}
    if registry:
        for entry in registry.discovered():
            discovered_records[entry["node_id"]] = entry
    if service is not None:
        for peer in service.peers():
            if peer.paired:
                continue
            discovered_records[peer.node_id] = peer.to_dict()

    # Seam with Module B: a posted pair/request must surface in the device
    # list as an awaiting_approval row so the wizard renders code entry +
    # approve/deny (fixture: tests/ui/.../devices_pending_approval.json).
    # Pending state wins over a plain discovered record for the same node.
    try:
        from .pairing import get_pairing_manager

        pairing_manager = get_pairing_manager()
    except RuntimeError:
        pairing_manager = None
    if pairing_manager is not None:
        for pending in pairing_manager.pending_requests():
            row = dict(discovered_records.get(pending["node_id"], {}))
            row.update(pending)
            row["state"] = "awaiting_approval"
            discovered_records[pending["node_id"]] = row

    self_record: dict[str, Any] = {
        "node_id": identity.node_id,
        "friendly_name": identity.friendly_name,
        "version": __version__,
        "cluster_name": _cluster_name(),
        "caps": service.config.caps.to_dict() if service is not None else {},
        "addrs": [],
        "http_port": service.config.http_port if service is not None else 0,
        "paired": True,
        "last_seen": time.time(),
        "link": "unknown",
        "state": "discovered",
    }
    return {
        "paired": paired,
        "discovered": list(discovered_records.values()),
        "self": self_record,
        "multicast_ok": bool(service.multicast_ok) if service else False,
        "mdns_available": bool(service.mdns_available) if service else False,
    }
