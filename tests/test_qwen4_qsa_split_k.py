"""Split-K sparse-GQA must agree with the single-pass kernel at every split."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


compat.apply_mlx_vlm_qwen4_exp_compat_patch()
qsa_fast = importlib.import_module("mlx_vlm.models.qwen4_exp.qsa_fast")

Q_HEADS, KV_HEADS, HEAD_DIM, TOPK, COMPRESS = 24, 2, 256, 512, 4


def _split_available() -> bool:
    return fast.is_native_available() and fast.has_symbol(
        "qwen4_qsa_sparse_gqa_attention_split"
    )


def _inputs(key_len: int, q_len: int):
    mx.random.seed(0)
    queries = mx.random.normal((1, Q_HEADS, q_len, HEAD_DIM)).astype(mx.bfloat16)
    keys = mx.random.normal((1, KV_HEADS, key_len, HEAD_DIM)).astype(mx.bfloat16)
    values = mx.random.normal((1, KV_HEADS, key_len, HEAD_DIM)).astype(mx.bfloat16)
    stride = max(1, key_len // COMPRESS // TOPK)
    blocks = mx.contiguous(
        mx.broadcast_to(
            mx.arange(0, TOPK, dtype=mx.int32)[None, None] * stride,
            (1, q_len, TOPK),
        )
    )
    mx.eval(queries, keys, values, blocks)
    return queries, keys, values, blocks


def _attend(monkeypatch, splits, args, q_offset):
    monkeypatch.setenv("OMLX_QWEN4_QSA_SPLIT_K", str(splits))
    monkeypatch.setattr(qsa_fast, "_NATIVE_QSA_MAIN_DISABLED", False)
    out = qsa_fast._native_sparse_gqa_attention(*args, q_offset=q_offset)
    assert out is not None, "native sparse-GQA seam failed closed"
    mx.eval(out)
    return out


@pytest.mark.skipif(not _split_available(), reason="split-K kernel unavailable")
@pytest.mark.parametrize("splits", [1, 2, 8, 16])
@pytest.mark.parametrize("q_len", [3])
def test_split_k_seam_matches_single_pass(monkeypatch, splits, q_len):
    key_len = 8192
    args = _inputs(key_len, q_len)
    q_offset = key_len - q_len

    single = _attend(monkeypatch, 0, args, q_offset)
    split = _attend(monkeypatch, splits, args, q_offset)

    # The seam returns [B, M, H, D] for both routes.
    assert split.shape == single.shape == (1, q_len, Q_HEADS, HEAD_DIM)
    ref = single.astype(mx.float32)
    rel = float(mx.max(mx.abs(ref - split.astype(mx.float32)))) / float(
        mx.max(mx.abs(ref))
    )
    # bf16 output rounding; a base-e cross-split merge lands at 1e-1..5e-1.
    assert rel < 2e-2, f"splits={splits} q_len={q_len} rel={rel:.3e}"
