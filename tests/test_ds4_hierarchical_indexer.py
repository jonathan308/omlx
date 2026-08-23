"""Exactness and derived-cache tests for the certified DS4 index screen."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches.deepseek_v4 import hierarchical_indexer as hi


pytestmark = pytest.mark.skipif(
    not (
        fast.is_native_available()
        and fast._EXT_MMA_SCORE
        and fast.has_symbol("dsa_topk_indices")
    ),
    reason="native rank-48 and top-k kernels are unavailable",
)


def _fixture(rows=16, pooled=2048, latent=16):
    mx.random.seed(20260825)
    basis_seed = mx.random.normal((128, latent)).astype(mx.float32)
    basis, _ = mx.linalg.qr(basis_seed, stream=mx.cpu)
    keys = (
        mx.random.normal((1, pooled, latent)).astype(mx.float32) @ basis.T
    ).astype(mx.bfloat16)
    # Queries share the same known subspace so the certified path succeeds;
    # random full-rank queries are still allowed to fail closed in production.
    q_rows = keys[0, :rows].astype(mx.float32)
    q = mx.broadcast_to(q_rows[None, None], (1, 64, rows, 128)).astype(
        mx.bfloat16
    )
    weights = mx.full((1, rows, 64), 0.05, dtype=mx.bfloat16)
    mx.eval(keys, q, weights)
    return q, keys, weights


def test_disabled_hierarchical_index_is_a_noop(monkeypatch):
    monkeypatch.setattr(hi, "_ENABLED", False)
    q, keys, weights = _fixture()
    assert (
        hi.hierarchical_topk(
            q,
            keys,
            weights,
            SimpleNamespace(),
            query_offset=8192,
            topk=512,
            ratio=4,
            kernels=fast,
        )
        is None
    )


def test_certified_hierarchical_indices_match_full_scan(monkeypatch):
    monkeypatch.setattr(hi, "_ENABLED", True)
    monkeypatch.setattr(hi, "_MIN_POOL", 1024)
    monkeypatch.setattr(hi, "_REFRESH_POOL", 4096)
    # Keep all but one key so this unit test targets the mapping/tie/certificate
    # contract; real-fixture pruning and rate are covered by the physical gate.
    monkeypatch.setattr(hi, "_CANDIDATE_FRACTION", 0.99)
    q, keys, weights = _fixture()
    offset = 8192
    expected_scores = fast.dsa_indexer_scores_mma(
        q,
        keys[:, None],
        weights,
        mask_ratio=4,
        mask_q_offset=offset,
    )
    expected = mx.sort(
        fast.dsa_topk_indices(expected_scores, 512)[:, 0], axis=-1
    )
    actual = hi.hierarchical_topk(
        q,
        keys,
        weights,
        SimpleNamespace(),
        query_offset=offset,
        topk=512,
        ratio=4,
        kernels=fast,
    )
    assert actual is not None
    mx.eval(expected, actual)
    assert mx.array_equal(expected, actual).item()


def test_derived_key_projection_extends_without_basis_refresh(monkeypatch):
    monkeypatch.setattr(hi, "_REFRESH_POOL", 4096)
    _, keys, _ = _fixture(pooled=1024)
    cache = SimpleNamespace()
    first = hi._state_for_cache(cache, keys[:, :768])
    second = hi._state_for_cache(cache, keys)
    assert first.basis is second.basis
    assert second.basis_pool_length == 768
    assert second.projected_pool_length == 1024
    assert second.key_projection.shape == (1024, 48)
