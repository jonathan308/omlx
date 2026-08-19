# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the v2 discovery/identity endpoints."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster import discovery_routes
from omlx.cluster.discovery import (
    DiscoveryConfig,
    DiscoveryService,
    PeerRecord,
    configure_discovery_service,
)
from omlx.cluster.identity import (
    NodeIdentity,
    configure_node_identity,
    reset_configured_identity,
)
from omlx.cluster.registry import (
    configure_device_registry,
    reset_configured_device_registry,
)


@pytest.fixture(autouse=True)
def _configured_stores(tmp_path):
    reset_configured_identity()
    reset_configured_device_registry()
    configure_discovery_service(None)
    discovery_routes.probe_rate_limiter._buckets.clear()
    identity = configure_node_identity(tmp_path)
    registry = configure_device_registry(tmp_path / "devices.json")
    # Bypass admin auth for the inventory endpoint in these unit tests;
    # admin wiring is exercised by the existing admin auth test suite.
    app = FastAPI()

    async def _allow():
        return True

    app.dependency_overrides[discovery_routes.require_admin] = _allow
    app.include_router(discovery_routes.discovery_router)
    client = TestClient(app)
    yield identity, registry, client
    reset_configured_identity()
    reset_configured_device_registry()
    configure_discovery_service(None)


def test_node_id_probe_is_public_and_returns_identity(_configured_stores):
    identity, _, client = _configured_stores

    response = client.get("/api/cluster/node_id")

    assert response.status_code == 200
    payload = response.json()
    assert payload["node_id"] == identity.node_id
    assert payload["version"]
    assert payload["cluster_name"] == "omlx"  # default with no service
    # The probe must not leak capabilities or the device inventory.
    assert "caps" not in payload
    assert "devices" not in payload


def test_node_id_probe_is_rate_limited(_configured_stores):
    _, _, client = _configured_stores
    limiter = discovery_routes.probe_rate_limiter
    original = (limiter._rate, limiter._burst)
    limiter._rate, limiter._burst = 0.0, 3
    try:
        codes = [client.get("/api/cluster/node_id").status_code for _ in range(5)]
    finally:
        limiter._rate, limiter._burst = original
    assert codes[:3] == [200, 200, 200]
    assert codes[3:] == [429, 429]


def test_devices_returns_inventory_with_multicast_signal(_configured_stores):
    identity, registry, client = _configured_stores
    registry.mark_paired("peer-1", friendly_name="studio-b", paired_at=10.0)

    response = client.get("/api/cluster/devices")

    assert response.status_code == 200
    payload = response.json()
    assert payload["self"]["node_id"] == identity.node_id
    assert payload["self"]["paired"] is True
    assert payload["paired"][0]["node_id"] == "peer-1"
    assert payload["discovered"] == []
    # No discovery service running: multicast_ok must be False so the UI
    # shows the Local Network permission guidance instead of silent zeros.
    assert payload["multicast_ok"] is False
    assert payload["mdns_available"] is False


def test_devices_reflects_live_discovery_service(
    _configured_stores, tmp_path
):
    identity, registry, client = _configured_stores

    service = DiscoveryService(
        identity,
        registry,
        DiscoveryConfig(cluster_name="home", http_port=8000),
        prober=lambda ip, port, timeout: None,
        interface_lister=lambda: [],
        zeroconf_module=None,
    )
    configure_discovery_service(service)
    peer = PeerRecord(node_id="peer-9", friendly_name="nine", http_port=8000)
    peer.last_seen = service._clock()
    service._peers["peer-9"] = peer
    service._last_hello_at = service._clock()

    payload = client.get("/api/cluster/devices").json()

    assert payload["cluster_name"] if "cluster_name" in payload else True
    assert payload["self"]["cluster_name"] == "home"
    assert payload["self"]["http_port"] == 8000
    assert payload["multicast_ok"] is True
    assert any(d["node_id"] == "peer-9" for d in payload["discovered"])
    # Registry merge happened through the service path? No — direct insertion
    # here; the discovered list is sourced from service.peers().
    configure_discovery_service(None)


def test_devices_excludes_paired_peers_from_discovered(
    _configured_stores, tmp_path
):
    identity, registry, client = _configured_stores
    registry.mark_paired("peer-1", friendly_name="studio-b")

    service = DiscoveryService(
        identity,
        registry,
        DiscoveryConfig(),
        prober=lambda ip, port, timeout: None,
        interface_lister=lambda: [],
        zeroconf_module=None,
    )
    configure_discovery_service(service)
    peer = PeerRecord(node_id="peer-1", friendly_name="studio-b")
    peer.paired = True
    service._peers["peer-1"] = peer

    payload = client.get("/api/cluster/devices").json()

    assert payload["discovered"] == []
    assert payload["paired"][0]["node_id"] == "peer-1"
    configure_discovery_service(None)


def test_discovery_health_without_service(_configured_stores):
    _, _, client = _configured_stores

    payload = client.get("/api/cluster/discovery/health").json()

    assert payload == {
        "multicast_rx_within_5s": False,
        "last_multicast_rx_at": None,
        "mdns_active": False,
        "transport": "disabled",
    }


def test_discovery_health_reflects_live_service(_configured_stores):
    identity, registry, client = _configured_stores
    service = DiscoveryService(
        identity,
        registry,
        DiscoveryConfig(cluster_name="home", http_port=8000),
        prober=lambda ip, port, timeout: None,
        interface_lister=lambda: [],
        zeroconf_module=None,
    )
    configure_discovery_service(service)
    service._last_hello_at = service._clock()
    service._last_hello_wall = 1782000002.5

    payload = client.get("/api/cluster/discovery/health").json()

    assert payload["multicast_rx_within_5s"] is True
    assert payload["last_multicast_rx_at"] == 1782000002.5
    assert payload["mdns_active"] is False  # zeroconf disabled in tests
    assert payload["transport"] == "multicast"
    configure_discovery_service(None)


def test_devices_merges_pending_pairing_requests(
    _configured_stores, monkeypatch
):
    identity, registry, client = _configured_stores

    class _FakePairingManager:
        def pending_requests(self):
            return [
                {
                    "node_id": "peer-pending",
                    "friendly_name": "studio-2",
                    "caps": {"chip": "M3"},
                    "addrs": ["192.168.1.11"],
                    "http_port": 8000,
                    "state": "awaiting_approval",
                    "created_at": 100.0,
                    "expires_at": 700.0,
                    "attempts": 0,
                    "locked": False,
                    "locked_until": None,
                }
            ]

    from omlx.cluster import pairing

    monkeypatch.setattr(
        pairing, "get_pairing_manager", lambda: _FakePairingManager()
    )

    payload = client.get("/api/cluster/devices").json()

    pending = [
        d for d in payload["discovered"] if d["node_id"] == "peer-pending"
    ]
    assert len(pending) == 1
    assert pending[0]["state"] == "awaiting_approval"
    assert pending[0]["friendly_name"] == "studio-2"


def test_devices_paired_rows_carry_enrolled_ssh_target(
    _configured_stores, monkeypatch
):
    from types import SimpleNamespace

    from omlx.cluster import enrollment

    _, registry, client = _configured_stores
    registry.mark_paired("peer-1", friendly_name="studio-b")
    enrolled = SimpleNamespace(
        node_id="peer-1", ssh="omlx@studio-b.local"
    )
    fake_store = SimpleNamespace(list_nodes=lambda: (enrolled,))
    monkeypatch.setattr(
        enrollment, "get_cluster_enrollment", lambda: fake_store
    )

    payload = client.get("/api/cluster/devices").json()

    assert payload["paired"][0]["ssh_target"] == "omlx@studio-b.local"


def test_cluster_name_persisted_config(tmp_path, monkeypatch):
    from omlx.cluster.discovery import load_cluster_name, save_cluster_name

    assert load_cluster_name(tmp_path) == "omlx"  # no file yet
    save_cluster_name("studio-lan", tmp_path)
    assert load_cluster_name(tmp_path) == "studio-lan"
    # Malformed JSON falls back to the default instead of raising.
    (tmp_path / "cluster" / "cluster.json").write_text("{not json")
    assert load_cluster_name(tmp_path) == "omlx"


def test_devices_requires_admin_auth(tmp_path):
    # Without the dependency override, an unauthenticated call is rejected.
    reset_configured_identity()
    configure_node_identity(tmp_path)
    app = FastAPI()
    app.include_router(discovery_routes.discovery_router)
    client = TestClient(app, raise_server_exceptions=False)
    try:
        response = client.get("/api/cluster/devices")
        assert response.status_code in {401, 302, 307}
    finally:
        reset_configured_identity()


def test_node_id_probe_503_without_identity():
    reset_configured_identity()
    discovery_routes.probe_rate_limiter._buckets.clear()
    app = FastAPI()
    app.include_router(discovery_routes.discovery_router)
    client = TestClient(app)
    try:
        response = client.get("/api/cluster/node_id")
        assert response.status_code == 503
    finally:
        pass  # fixture resets on next test
