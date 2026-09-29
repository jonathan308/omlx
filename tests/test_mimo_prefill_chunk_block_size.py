# SPDX-License-Identifier: Apache-2.0
"""The paged cache block size follows MiMo's wider prefill floor.

With the prefix cache on, every prefill chunk is clamped to the next block
boundary; a block smaller than the prefill floor would silently split the
wider chunks back down.
"""

from types import SimpleNamespace

from omlx.scheduler import Scheduler


def _scheduler(*, floor: int, window: int = 128, mimo: bool = True) -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s.config = SimpleNamespace(paged_ssd_cache_dir="/tmp/ssd", paged_cache_block_size=256)
    s._qwen35_prefill_floor = floor
    s._detect_rotating_window_sizes = lambda: {window}
    s._detect_pooling_cache = lambda: False
    s._is_mimo_hybrid = lambda: mimo
    return s


def test_block_size_reaches_the_wide_prefill_floor():
    s = _scheduler(floor=4096)
    s._align_block_size_with_rotating_window()
    assert s.config.paged_cache_block_size == 4096


def test_block_size_keeps_the_pooling_default_without_a_floor():
    s = _scheduler(floor=0)
    s._align_block_size_with_rotating_window()
    assert s.config.paged_cache_block_size == Scheduler._POOLING_ROTATING_BLOCK_SIZE


def test_block_size_ignores_a_floor_off_the_window_grid():
    s = _scheduler(floor=4000)
    s._align_block_size_with_rotating_window()
    assert s.config.paged_cache_block_size == Scheduler._POOLING_ROTATING_BLOCK_SIZE


def test_non_mimo_rotating_models_are_unchanged():
    s = _scheduler(floor=4096, mimo=False)
    s._align_block_size_with_rotating_window()
    # window 128 -> smallest multiple in [512, 1024]
    assert s.config.paged_cache_block_size == 512


def test_wide_mimo_chunk_requires_fused_full_attention(monkeypatch):
    """Without the fused 192/128 attention the wider chunk is not used."""
    import sys
    from types import SimpleNamespace as NS

    import omlx.utils
    from omlx import scheduler

    def use(module):
        # ``from .utils import fast_attention`` reads the package attribute
        # first, then sys.modules; None in sys.modules makes it ImportError.
        monkeypatch.setitem(sys.modules, "omlx.utils.fast_attention", module)
        if module is None:
            monkeypatch.delattr(omlx.utils, "fast_attention", raising=False)
        else:
            monkeypatch.setattr(omlx.utils, "fast_attention", module, raising=False)

    use(None)
    assert scheduler._mimo_fused_full_attention() is False

    use(NS(_native_mixed_dims_supported=lambda qk, v: False, _nax_available=lambda: True))
    assert scheduler._mimo_fused_full_attention() is True

    use(NS(_native_mixed_dims_supported=lambda qk, v: True, _nax_available=lambda: False))
    assert scheduler._mimo_fused_full_attention() is True

    use(NS(_native_mixed_dims_supported=lambda qk, v: False, _nax_available=lambda: False))
    assert scheduler._mimo_fused_full_attention() is False
