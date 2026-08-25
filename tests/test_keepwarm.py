# SPDX-License-Identifier: Apache-2.0
"""Safety and scheduling gates for latent Metal/RDMA keepwarm."""

from types import SimpleNamespace

from omlx.keepwarm import (
    KeepwarmAction,
    KeepwarmConfig,
    KeepwarmController,
    distributed_dataplane_ping,
)


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def config(**overrides):
    values = {
        "enabled": True,
        "interval_seconds": 10.0,
        "idle_after_seconds": 2.0,
        "matrix_size": 1,
        "repeats": 1,
        "request_start_enabled": True,
        "request_start_idle_seconds": 2.0,
        "request_start_matrix_size": 128,
        "post_response_enabled": True,
        "post_response_delay_seconds": 5.0,
        "post_response_matrix_size": 128,
        "large_cache_tokens": 8192,
        "large_cache_interval_seconds": 60.0,
        "slow_threshold_seconds": 1.0,
        "slow_backoff_seconds": 60.0,
        "dataplane_ping": True,
    }
    values.update(overrides)
    return KeepwarmConfig(**values)


def test_keepwarm_is_default_off(monkeypatch):
    monkeypatch.delenv("OMLX_KEEPWARM", raising=False)
    assert KeepwarmConfig.from_env().enabled is False


def test_request_start_and_post_response_are_idle_gap_gated():
    clock = Clock()
    controller = KeepwarmController(config(), clock=clock)

    clock.value = 1.0
    assert controller.request_start_action() is None
    controller.observe_request_state(False)

    clock.value = 4.0
    action = controller.request_start_action()
    assert action is not None
    assert action.kind == "request_start"
    assert action.matrix_size == 128
    controller.record(action, elapsed_seconds=0.001, ok=True)

    clock.value = 5.0
    controller.observe_request_state(False)
    clock.value = 9.9
    assert controller.idle_action() is None
    clock.value = 10.0
    action = controller.idle_action()
    assert action is not None
    assert action.kind == "post_response"


def test_large_cache_stretches_periodic_interval():
    clock = Clock()
    controller = KeepwarmController(config(), clock=clock)
    clock.value = 2.0
    first = controller.idle_action(cache_tokens=10_000)
    assert first is not None and first.kind == "idle"
    controller.record(first, elapsed_seconds=0.001, ok=True)

    clock.value = 20.0
    assert controller.idle_action(cache_tokens=10_000) is None
    clock.value = 62.0
    assert controller.idle_action(cache_tokens=10_000) is not None


def test_slow_touch_enters_backoff():
    clock = Clock()
    controller = KeepwarmController(config(), clock=clock)
    clock.value = 2.0
    action = controller.idle_action()
    assert action is not None
    controller.record(action, elapsed_seconds=1.5, ok=True)
    clock.value = 30.0
    assert controller.idle_action() is None
    clock.value = 62.0
    assert controller.idle_action() is not None
    assert controller.snapshot()["slow_count"] == 1


class Value:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Distributed:
    def __init__(self, receives):
        self.receives = list(receives)
        self.calls = []

    def send(self, value, target, *, group):
        self.calls.append(("send", target, group))
        return value

    def recv_like(self, value, source, *, group):
        self.calls.append(("recv", source, group))
        return Value(self.receives.pop(0))


class FakeMX:
    uint32 = "uint32"

    def __init__(self, receives):
        self.distributed = Distributed(receives)
        self.evaluated = []

    @staticmethod
    def array(values, dtype):
        del dtype
        return Value(values[0])

    def eval(self, value):
        self.evaluated.append(value)


def test_rank_zero_dataplane_ping_visits_every_worker_in_complementary_order():
    mx = FakeMX([1, 2])
    group = SimpleNamespace(name="jaccl")
    distributed_dataplane_ping(mx, group, rank=0, world_size=3)
    assert mx.distributed.calls == [
        ("send", 1, group),
        ("recv", 1, group),
        ("send", 2, group),
        ("recv", 2, group),
    ]


def test_worker_dataplane_ping_receives_before_acknowledging():
    mx = FakeMX([0])
    group = SimpleNamespace(name="jaccl")
    distributed_dataplane_ping(mx, group, rank=2, world_size=3)
    assert mx.distributed.calls == [
        ("recv", 0, group),
        ("send", 0, group),
    ]


def test_action_shape_is_bounded_by_configuration_parser(monkeypatch):
    monkeypatch.setenv("OMLX_KEEPWARM", "1")
    monkeypatch.setenv("OMLX_KEEPWARM_MATRIX_SIZE", "99999")
    monkeypatch.setenv("OMLX_KEEPWARM_REPEATS", "999")
    parsed = KeepwarmConfig.from_env()
    assert parsed.enabled is True
    assert parsed.matrix_size == 1024
    assert parsed.repeats == 16


def test_action_is_an_immutable_transport_value():
    action = KeepwarmAction("idle", 1, 1, 2.0)
    assert action.kind == "idle"
