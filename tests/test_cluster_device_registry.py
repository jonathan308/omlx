# SPDX-License-Identifier: Apache-2.0

import stat

import pytest

from omlx.cluster.registry import (
    DeviceRegistry,
    configure_device_registry,
    get_device_registry,
    reset_configured_device_registry,
)


def _announce(node_id: str, name: str, **extra):
    record = {
        "node_id": node_id,
        "friendly_name": name,
        "caps": {"chip": "M3 Ultra", "ram_gb": 96},
        "addrs": [{"ip": "fe80::1", "if_type": "ethernet"}],
    }
    record.update(extra)
    return record


def test_discovered_devices_are_memory_only(tmp_path):
    path = tmp_path / "devices.json"
    registry = DeviceRegistry(path)

    registry.merge(_announce("node-a", "studio-a"))

    assert registry.discovered() == [
        {
            "node_id": "node-a",
            "friendly_name": "studio-a",
            "caps": {"chip": "M3 Ultra", "ram_gb": 96},
            "last_addrs": ["fe80::1"],
        }
    ]
    assert registry.paired() == []
    # Nothing was persisted for an untrusted announcer.
    assert not path.exists()
    restored = DeviceRegistry(path)
    assert restored.discovered() == []
    assert restored.paired() == []


def test_merge_dedupes_on_node_id(tmp_path):
    registry = DeviceRegistry(tmp_path / "devices.json")

    registry.merge(_announce("node-a", "studio-a"))
    registry.merge(_announce("node-a", "studio-a-renamed", caps={"ram_gb": 128}))

    discovered = registry.discovered()
    assert len(discovered) == 1
    assert discovered[0]["friendly_name"] == "studio-a-renamed"
    assert discovered[0]["caps"] == {"ram_gb": 128}


def test_mark_paired_persists_atomically_with_private_permissions(tmp_path):
    path = tmp_path / "devices.json"
    registry = DeviceRegistry(path)
    registry.merge(_announce("node-a", "studio-a"))

    device = registry.mark_paired("node-a", paired_at=123.0)

    assert device["paired_at"] == 123.0
    assert registry.paired() == [device]
    assert registry.discovered() == []
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    restored = DeviceRegistry(path)
    assert restored.paired() == [device]
    assert restored.is_paired("node-a")


def test_merge_updates_paired_device_without_losing_trust(tmp_path):
    path = tmp_path / "devices.json"
    registry = DeviceRegistry(path)
    registry.mark_paired("node-a", friendly_name="studio-a", paired_at=1.0)

    registry.merge(_announce("node-a", "studio-a", caps={"chip": "M4"}))

    device = registry.get("node-a")
    assert device["caps"] == {"chip": "M4"}
    assert device["last_addrs"] == ["fe80::1"]
    assert device["paired_at"] == 1.0
    # The update was persisted.
    assert DeviceRegistry(path).get("node-a")["caps"] == {"chip": "M4"}


def test_merge_never_downgrades_paired_to_discovered(tmp_path):
    registry = DeviceRegistry(tmp_path / "devices.json")
    registry.mark_paired("node-a", friendly_name="studio-a")

    registry.merge({"node_id": "node-a"})

    assert registry.is_paired("node-a")
    assert registry.discovered() == []


def test_unpair_revokes_and_persists(tmp_path):
    path = tmp_path / "devices.json"
    registry = DeviceRegistry(path)
    registry.mark_paired("node-a", friendly_name="studio-a")

    assert registry.unpair("node-a") is True
    assert registry.unpair("node-a") is False
    assert registry.paired() == []
    # Device falls back to memory-only discovered so the UI can still show it.
    assert registry.discovered()[0]["node_id"] == "node-a"
    assert DeviceRegistry(path).paired() == []


def test_corruption_fails_closed_without_blocking_server(tmp_path):
    path = tmp_path / "devices.json"
    path.write_text("{broken", encoding="utf-8")

    registry = DeviceRegistry(path)

    assert registry.paired() == []
    assert "could not read" in registry.load_error


def test_duplicate_node_ids_on_disk_are_rejected(tmp_path):
    path = tmp_path / "devices.json"
    path.write_text(
        '{"schema_version": 1, "devices": ['
        '{"node_id": "a", "friendly_name": "x", "paired_at": 1},'
        '{"node_id": "a", "friendly_name": "y", "paired_at": 2}]}',
        encoding="utf-8",
    )

    registry = DeviceRegistry(path)

    assert registry.paired() == []
    assert "duplicate" in registry.load_error


def test_merge_rejects_record_without_node_id(tmp_path):
    registry = DeviceRegistry(tmp_path / "devices.json")
    with pytest.raises(ValueError, match="node_id"):
        registry.merge({"friendly_name": "ghost"})


def test_configure_and_get_process_wide_registry(tmp_path):
    reset_configured_device_registry()
    try:
        registry = configure_device_registry(tmp_path / "devices.json")
        assert get_device_registry() is registry
    finally:
        reset_configured_device_registry()


def test_get_device_registry_requires_configuration():
    reset_configured_device_registry()
    with pytest.raises(RuntimeError, match="not configured"):
        get_device_registry()
