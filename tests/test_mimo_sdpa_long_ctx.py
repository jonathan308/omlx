# SPDX-License-Identifier: Apache-2.0
"""MiMo decode / MTP-verify attention kernels for long KV caches.

``sdpa_rows``: every row of a short forward in one pass over the KV cache,
bit-identical to MLX's vector kernel (the path it replaces: one call for
rows x GQA <= 32, row chunks beyond).
"""

import zlib

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.mimo_v2 import decode_fast as df
from omlx.patches.mimo_v2 import sdpa_rows as sr

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available() or not df._nax_available(),
    reason="MiMo fused decode kernels are validated and enabled on M5 (NAX) GPUs",
)

BF16 = mx.bfloat16
D, DV = 192, 128


def _mlx_sdpa(q, k, v, cache=None, scale=1.0, mask=None, sinks=None):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask, sinks=sinks)


def _today(q, k, v, scale, mask, sinks):
    """The decode path's attention before these kernels (B, L, H * Dv)."""
    B, H, L, _ = q.shape
    rep = H // k.shape[1]
    if L * rep > 32:
        return df._sdpa_row_chunks(_mlx_sdpa, q, k, v, None, scale, mask, sinks, 32 // rep)
    return _mlx_sdpa(q, k, v, scale=scale, mask=mask, sinks=sinks).swapaxes(1, 2).reshape(B, L, -1)


def _inputs(B, H, Hk, L, S, mask_kind, with_sinks, tag):
    mx.random.seed(zlib.crc32(repr(tag).encode()))
    q = (mx.random.normal((B, L, H, D)) * 3).astype(BF16).swapaxes(1, 2)
    # KV-cache views: the head stride exceeds the key count (step slack).
    k = mx.random.normal((B, Hk, S + 256, D)).astype(BF16)[:, :, :S]
    v = mx.random.normal((B, Hk, S + 256, DV)).astype(BF16)[:, :, :S]
    sinks = mx.random.normal((H,)).astype(BF16) if with_sinks else None
    from mlx_lm.models.base import create_causal_mask

    if mask_kind == "none":
        mask = None
    elif mask_kind == "causal":
        mask = "causal"
    elif mask_kind == "window":
        mask = create_causal_mask(L, offset=S - L, window_size=S // 3)
    elif mask_kind == "padded":
        mask = create_causal_mask(L, offset=S - L, left_padding=mx.array([0, 700][:B]))
    else:  # additive
        allowed = create_causal_mask(L, offset=S - L)
        bias = (mx.random.normal((L, S)) * 0.5).astype(BF16)
        mask = mx.where(allowed, bias, mx.array(-mx.inf, dtype=BF16))
    return q, k, v, mask, sinks


def _bits(x):
    return np.array(x.view(mx.uint16))


CASES = [
    # (B, H, Hk, L, S, mask, sinks)
    (1, 64, 4, 1, 1100, "none", False),
    (1, 64, 4, 1, 1100, "none", True),
    (1, 64, 4, 2, 1100, "causal", False),
    (1, 64, 4, 3, 1100, "causal", False),
    (1, 64, 4, 3, 1100, "causal", True),
    (1, 64, 4, 4, 17000, "causal", False),
    (1, 64, 4, 3, 17000, "window", True),
    (1, 64, 4, 3, 5000, "additive", False),
    (2, 64, 4, 3, 5000, "padded", False),
    (2, 64, 4, 2, 5000, "padded", True),
    (1, 64, 8, 3, 5000, "causal", True),
]


@pytest.mark.parametrize("case", CASES)
def test_sdpa_rows_is_bit_identical_to_mlx(case):
    B, H, Hk, L, S, mask_kind, with_sinks = case
    q, k, v, mask, sinks = _inputs(B, H, Hk, L, S, mask_kind, with_sinks, case)
    scale = D ** -0.5
    out = sr.sdpa_rows(q, k, v, scale, mask, sinks)
    assert out is not None
    ref = _today(q, k, v, scale, mask, sinks)
    assert out.shape == ref.shape == (B, L, H * DV)
    assert (_bits(out) != _bits(ref)).sum() == 0


def test_sdpa_rows_declines_outside_mlx_2pass_contract():
    scale = D ** -0.5
    # MLX's single-pass kernel below 1024 keys (on Ultra / Max GPUs).
    q, k, v, _, _ = _inputs(1, 64, 4, 3, 1000, "causal", False, "short")
    if sr._device_class() in ("d", "s"):
        assert sr.sdpa_rows(q, k, v, scale, "causal", None) is None
    # Rows whose MLX calls would pick different key-block counts (a causal
    # 3-row verify at exactly 16384 / 65536 keys) keep today's path.
    if sr._device_class() == "d":
        assert sr.plan_blocks(3, 64, 4, 16384, True) is None
        assert sr.plan_blocks(3, 64, 4, 65536, True) is None
        assert sr.plan_blocks(3, 64, 4, 65537, True) == 1024
        assert sr.plan_blocks(1, 64, 4, 16384, False) == 512
    # MLX's single-row GQA-8 variant (no mask, no sinks) has its own arithmetic.
    q = mx.zeros((1, 32, 1, 128), BF16)
    k = mx.zeros((1, 4, 9000, 128), BF16)
    assert sr.sdpa_rows(q, k, k, 1.0, None, None) is None


def test_mlx_blocks_heuristic_matches_mlx_0_32():
    assert sr.mlx_blocks(8192, 16, "d") == 128
    assert sr.mlx_blocks(16384, 16, "d") == 512
    assert sr.mlx_blocks(65535, 32, "d") == 512
    assert sr.mlx_blocks(65536, 16, "d") == 1024
    assert sr.mlx_blocks(9000, 2, "d") == 256
    assert sr.mlx_blocks(40000, 8, "s") == 512
    assert sr.mlx_blocks(40000, 2, "g") == 32


def test_sdpa_rows_reads_strided_views():
    """Queries / keys / values whose last axis is not contiguous (read with
    their strides, no copy) give the same result as contiguous copies."""
    B, H, Hk, L, S = 1, 64, 4, 3, 5000
    q, k, v, _, sinks = _inputs(B, H, Hk, L, S, "causal", True, "strided")
    kt = mx.contiguous(k.swapaxes(2, 3)).swapaxes(2, 3)  # (B, Hk, S, D) view, inner stride S
    vt = mx.contiguous(v.swapaxes(2, 3)).swapaxes(2, 3)
    qt = mx.contiguous(q.swapaxes(2, 3)).swapaxes(2, 3)
    fn = sr.sdpa_rows
    scale = D ** -0.5
    a = fn(q, mx.contiguous(k), mx.contiguous(v), scale, "causal", sinks)
    b = fn(qt, kt, vt, scale, "causal", sinks)
    assert a is not None and b is not None
    assert (_bits(a) != _bits(b)).sum() == 0
