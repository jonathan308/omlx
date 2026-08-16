"""Decode fast-path kernels (fused residual+RMS norm, ...) with fallback.

Ports of the user's closed-unmerged mlx core PRs so omlx ships the fusion
without waiting on an mlx release. Every public symbol degrades to the
composed mlx ops when the native extension is absent, ABI-mismatched, or
the shape/dtype is unsupported — callers can use these unconditionally.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)


def _detach_import_error(exc: Exception) -> Exception:
    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None
    return exc


try:
    from . import _ext
except Exception as exc:  # pragma: no cover - depends on local native build
    _ext = None
    _IMPORT_ERROR = _detach_import_error(exc)
    if any(Path(__file__).parent.glob("_ext*.so")):
        logger.warning(
            "%s: native extension is present but failed to load; falling "
            "back to the composed path: %s",
            __name__,
            _IMPORT_ERROR,
        )
else:
    _IMPORT_ERROR = None


def _verify_abi(ext, import_error):
    """Disable native symbols when the extension rejects mlx arrays (#2139)."""
    if ext is None:
        return ext, import_error
    probe = getattr(ext, "abi_probe", None)
    if probe is None:
        return ext, import_error
    try:
        probe(mx.zeros((1,)))
    except TypeError as exc:
        logger.warning(
            "%s: native kernels disabled — nanobind ABI mismatch with the "
            "installed mlx wheel; rebuild against the installed mlx.",
            __name__,
        )
        return None, _detach_import_error(exc)
    return ext, import_error


_ext, _IMPORT_ERROR = _verify_abi(_ext, _IMPORT_ERROR)

NATIVE_AVAILABLE = _ext is not None


def _composed_rms_norm_residual(
    x: mx.array, weight: mx.array, residual: mx.array, eps: float
) -> Tuple[mx.array, mx.array]:
    summed = x + residual
    out = mx.fast.rms_norm(summed, weight, eps)
    return out, summed


def rms_norm_residual(
    x: mx.array,
    weight: mx.array,
    residual: mx.array,
    eps: float,
    *,
    stream: Optional[mx.Stream] = None,
    force_composed: bool = False,
) -> Tuple[mx.array, mx.array]:
    """Return (rms_norm(x + residual) * weight, x + residual).

    Single fused Metal dispatch when the native extension applies; composed
    add + mx.fast.rms_norm otherwise. NOTE: the fused kernel requires dense
    rows (row-contiguous, unit-stride last axis — always true for hidden
    states produced by matmul/attention/add). Layout cannot be vetted on
    lazy arrays, so do not route arbitrary strided views through the native
    path; use force_composed=True for those.
    """
    if not force_composed and _ext is not None and _ext.rms_norm_residual_supported(
        x, weight, residual, stream
    ):
        out, summed = _ext.rms_norm_residual(x, weight, residual, eps, stream)
        return out, summed
    return _composed_rms_norm_residual(x, weight, residual, eps)


__all__ = ["NATIVE_AVAILABLE", "rms_norm_residual"]
