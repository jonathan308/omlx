# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.fast_attention prefill fast paths."""

import mlx.core as mx
import pytest

from omlx.utils.fast_attention import (
    blocked_sliding_window_attention,
    mixed_head_dim_sdpa,
)


def _reference_window_attention(q, k, v, scale, window, sinks):
    L, S = q.shape[2], k.shape[2]
    prefix = S - L
    qpos = mx.arange(prefix, S)[:, None]
    kpos = mx.arange(S)[None, :]
    mask = (kpos <= qpos) & (kpos > qpos - window)
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask, sinks=sinks
    )


@pytest.mark.parametrize("prefix", [0, 50, 127, 128, 300])
@pytest.mark.parametrize("dims", [(64, 64), (192, 128)])
@pytest.mark.parametrize("L", [512, 511, 300])
def test_blocked_sliding_window_matches_masked(prefix, dims, L):
    mx.random.seed(prefix + L)
    qk_dim, v_dim = dims
    H, Hk, window = 8, 2, 128
    S = prefix + L
    q = mx.random.normal((1, H, L, qk_dim))
    k = mx.random.normal((1, Hk, S, qk_dim))
    v = mx.random.normal((1, Hk, S, v_dim))
    sinks = mx.random.normal((H,))
    scale = qk_dim**-0.5
    ref = _reference_window_attention(q, k, v, scale, window, sinks)
    out = blocked_sliding_window_attention(
        q, k, v, scale=scale, window=window, sinks=sinks, block=128
    )
    assert out is not None
    assert out.shape == ref.shape
    assert mx.allclose(out, ref, atol=1e-4, rtol=1e-4).item()


@pytest.mark.parametrize("pad", [0, 37])
def test_blocked_sliding_window_honours_padded_bool_mask(pad):
    mx.random.seed(pad)
    H, Hk, L, window, prefix = 4, 2, 384, 128, 100
    S = prefix + L
    q = mx.random.normal((1, H, L, 64))
    k = mx.random.normal((1, Hk, S, 64))
    v = mx.random.normal((1, Hk, S, 64))
    qpos = mx.arange(prefix, S)[:, None]
    kpos = mx.arange(S)[None, :]
    mask = (kpos <= qpos) & (kpos > qpos - window) & (kpos >= pad)
    mask = mask[None, None]
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask)
    out = blocked_sliding_window_attention(
        q, k, v, scale=0.125, window=window, mask=mask, block=128
    )
    assert out is not None
    assert mx.allclose(out, ref, atol=1e-4, rtol=1e-4).item()


def test_blocked_sliding_window_declines_unsupported_layouts():
    q = mx.zeros((2, 4, 512, 64))
    k = mx.zeros((2, 4, 512, 64))
    assert (
        blocked_sliding_window_attention(q, k, k, scale=1.0, window=128) is None
    )  # batched
    q = mx.zeros((1, 4, 200, 64))
    k = mx.zeros((1, 4, 200, 64))
    assert (
        blocked_sliding_window_attention(q, k, k, scale=1.0, window=128) is None
    )  # shorter than two blocks


def test_mixed_head_dim_sdpa_matches_unfused():
    from omlx.utils import fast_attention

    if not fast_attention._nax_available():
        pytest.skip("fused padded-V path is only enabled on NAX GPUs")
    mx.random.seed(0)
    q = mx.random.normal((1, 8, 256, 192)).astype(mx.float16)
    k = mx.random.normal((1, 2, 256, 192)).astype(mx.float16)
    v = mx.random.normal((1, 2, 256, 128)).astype(mx.float16)
    sinks = mx.random.normal((8,)).astype(mx.float16)
    scale = 192**-0.5
    ref = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask="causal", sinks=sinks
    )
    out = mixed_head_dim_sdpa(q, k, v, scale=scale, mask="causal", sinks=sinks)
    assert out is not None
    assert out.shape == ref.shape
    assert mx.allclose(out, ref, atol=2e-2, rtol=2e-2).item()


def _mixed_dims_reference(q, k, v, scale, mask):
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    mx.eval(out)
    return out


@pytest.mark.parametrize("mask_kind", ["causal", "array"])
def test_mixed_head_dim_sdpa_long_context_all_routes(monkeypatch, mask_kind):
    """Native, padded-to-256 and padded-to-qk routes all match the reference."""
    from omlx.utils import fast_attention

    if not fast_attention._nax_available():
        pytest.skip("fast mixed head-dim paths are only enabled on NAX GPUs")
    mx.random.seed(1)
    L, S = 300, 1100
    q = (0.5 * mx.random.normal((1, 8, L, 192))).astype(mx.bfloat16)
    k = (0.5 * mx.random.normal((1, 2, S, 192))).astype(mx.bfloat16)
    v = (0.5 * mx.random.normal((1, 2, S, 128))).astype(mx.bfloat16)
    scale = 192**-0.5
    if mask_kind == "causal":
        mask = "causal"
    else:
        mask = (mx.arange(L)[:, None] + (S - L)) >= mx.arange(S)[None, :]
        mask = mask[None, None]
    ref = _mixed_dims_reference(q, k, v, scale, mask)

    outs = {}
    outs["auto"] = mixed_head_dim_sdpa(q, k, v, scale=scale, mask=mask)
    monkeypatch.setattr(
        fast_attention, "_native_mixed_dims_supported", lambda *a: False
    )
    outs["pad256"] = mixed_head_dim_sdpa(q, k, v, scale=scale, mask=mask)
    for name, out in outs.items():
        assert out is not None, name
        assert out.shape == ref.shape, name
        assert mx.allclose(out, ref, atol=2e-2, rtol=2e-2).item(), name


def test_mixed_head_dim_sdpa_native_probe_is_cached():
    from omlx.utils import fast_attention

    fast_attention._native_mixed_dims_supported.cache_clear()
    first = fast_attention._native_mixed_dims_supported(192, 128)
    assert fast_attention._native_mixed_dims_supported(192, 128) is first
    assert fast_attention._native_mixed_dims_supported.cache_info().hits >= 1
