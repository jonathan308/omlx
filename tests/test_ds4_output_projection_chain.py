from pathlib import Path

from benchmarks import bench_ds4_output_projection_chain as campaign

ROOT = Path(__file__).resolve().parents[1]


def test_abba_order_is_balanced(monkeypatch):
    calls = []
    monkeypatch.setattr(campaign, "_evaluate", lambda value: None)

    def stock():
        calls.append("A")

    def native():
        calls.append("B")

    campaign._abba(stock, native, warmup=0, cycles=2)
    assert calls == ["A", "B", "B", "A", "A", "B", "B", "A"]


def test_campaign_covers_all_steel_tile_pairings():
    assert campaign.VARIANTS == {
        0: {"o_a_bm": 32, "o_b_bm": 32},
        1: {"o_a_bm": 64, "o_b_bm": 64},
        2: {"o_a_bm": 64, "o_b_bm": 32},
        3: {"o_a_bm": 32, "o_b_bm": 64},
    }


def test_native_chain_keeps_the_bf16_boundary_and_stock_k_walk():
    source = (
        ROOT / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_output_chain.metal"
    ).read_text()
    assert "ds4_output_fp_qmm_t_impl<T, 32, 8" in source
    assert "for (int k = 0; k < K_eff; k += BK)" in source
    assert "mma_op.store_result(y, y_stride)" in source
    assert "y_group_offset" in source
    assert "instantiate_ds4_output_oa(bfloat16_t, 32, 32, 32)" in source
    assert "instantiate_ds4_output_oa(bfloat16_t, 64, 32, 32)" in source


def test_native_chain_owns_one_ephemeral_intermediate_then_o_b():
    source = (
        ROOT / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_output_chain.cpp"
    ).read_text()
    allocation = source.index(
        "array o_mid({1, kTokens, kOBInput}, bfloat16, nullptr, {})"
    )
    temporary = source.index("encoder.add_temporary(o_mid)")
    o_a = source.index('"ds4_output_oa_interleaved_bfloat16_t_bm"', temporary)
    o_b = source.index('"ds4_projection_mxfp8_qmm_t_bfloat16_t_bm"', o_a)
    assert allocation < temporary < o_a < o_b


def test_output_chain_symbols_are_built_with_default_off_model_seam():
    cmake = (ROOT / "omlx/custom_kernels/glm_moe_dsa/csrc/CMakeLists.txt").read_text()
    bindings = (ROOT / "omlx/custom_kernels/glm_moe_dsa/csrc/bindings.cpp").read_text()
    wrapper = (ROOT / "omlx/custom_kernels/glm_moe_dsa/fast.py").read_text()
    model = (ROOT / "omlx/patches/deepseek_v4/deepseek_v4_model.py").read_text()
    for filename in ("ds4_output_chain.cpp", "ds4_output_chain.metal"):
        assert filename in cmake
    for symbol in ("ds4_output_oa_interleaved", "ds4_output_projection_chain"):
        assert symbol in bindings
        assert symbol in wrapper
    assert "ds4_output_oa_interleaved" not in model
    assert "ds4_output_projection_chain" in model
    assert '"OMLX_DSV4_OUTPUT_CHAIN_PREFILL", "0"' in model
