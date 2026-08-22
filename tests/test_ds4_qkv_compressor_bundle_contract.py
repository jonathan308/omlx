from __future__ import annotations

from pathlib import Path

from benchmarks.bench_ds4_qkv_compressor_bundle import (
    LAYER_RATIOS,
    NATIVE_B1_SYMBOL,
    byte_ledger,
    dispatch_ledger,
    packed_slices,
    promotion_contract,
    projections,
)


def test_real_ds4_layer_schedule_and_projection_extents():
    assert {ratio: LAYER_RATIOS.count(ratio) for ratio in (0, 4, 128)} == {
        0: 2,
        4: 21,
        128: 20,
    }
    assert [item.rows for item in projections(4)] == [1024, 512, 1024, 1024, 256, 256]
    assert [item.rows for item in projections(128)] == [1024, 512, 512, 512]
    assert packed_slices(4)["indexer.compressor.wgate"] == (3840, 4096)


def test_checkpoint_and_runtime_byte_ledger_is_exact():
    ledger = byte_ledger()
    assert ledger["per_layer"]["0"]["checkpoint_bytes"] == 6_291_840
    assert ledger["per_layer"]["4"]["checkpoint_bytes"] == 27_263_360
    assert ledger["per_layer"]["128"]["checkpoint_bytes"] == 14_680_448
    assert ledger["all_layers_checkpoint_bytes"] == 878_723_200
    assert ledger["all_layers_runtime_bytes"] == 887_160_832


def test_dispatch_ledger_is_per_rank_and_collective_neutral():
    ledger = dispatch_ledger()
    assert ledger["current_projection_dispatches_per_rank"] == 210
    assert ledger["two_bank_dispatches_per_rank"] == 84
    assert ledger["full_bundle_dispatches_per_rank"] == 43
    assert ledger["full_bundle_dispatches_saved_per_rank"] == 167
    assert ledger["collectives_changed"] == 0


def test_first_native_abi_stops_before_ape_and_cache_mutation():
    contract = promotion_contract()
    assert contract["first_native_symbol"] == NATIVE_B1_SYMBOL
    assert contract["shape"] == {"batch": 1, "rows": 1, "hidden": 4096, "ratio": 4}
    assert contract["output"] == "packed_bf16[1,4096]"
    assert contract["forbidden_inputs"] == [
        "ape",
        "cache",
        "position",
        "distributed_group",
    ]
    assert contract["parity"]["decode_positions"] == [0, 1, 2, 3]


def test_symbol_remains_outside_production_until_gate_passes():
    root = Path(__file__).parents[1]
    hits = []
    for path in (root / "omlx").rglob("*.py"):
        if NATIVE_B1_SYMBOL in path.read_text():
            hits.append(path)
    assert hits == []

