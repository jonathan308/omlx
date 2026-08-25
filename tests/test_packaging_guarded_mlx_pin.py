# SPDX-License-Identifier: Apache-2.0
"""The Fusion DMG must pin guarded MLX and must not admit 0.32.2."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIN = "0.32.1.dev20260825+26421e953"


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


def test_pyproject_pins_exact_guarded_mlx_not_open_range():
    text = _read("pyproject.toml")
    assert "mlx>=0.32.0.dev0,<0.33" not in text
    assert PIN in text
    assert f"mlx=={PIN}" in text
    assert f"mlx-metal=={PIN}" in text
    overrides = text.split("[tool.uv]", 1)[1]
    assert f"mlx=={PIN}" in overrides
    assert "mlx==0.32.0" not in overrides


def test_packaging_build_never_queries_pypi_for_mlx():
    text = _read("packaging/build.py")
    assert "pypi.org/pypi/mlx" not in text
    assert "GUARDED_MLX_VERSION" in text
    assert PIN in text
    assert "0.32.2" in text  # forbidden-version guard


def test_dmg_name_is_macos26_only():
    script = _read("scripts/release_macos_dmg.sh")
    assert "macos15-26" not in script
    assert 'DMG_NAME="oMLX-$APP_VERSION-macos26-arm64.dmg"' in script


def test_checklist_does_not_label_runtime_macos_15_26():
    text = _read("docs/release-public-checklist.md")
    assert "macos15-26" not in text
    assert "macos26-arm64" in text


@pytest.mark.parametrize(
    "rel",
    [
        "packaging/_wheels/guarded/README.md",
        "docs/packaging-guarded-mlx.md",
    ],
)
def test_docs_reject_cp313_rollback_wheels(rel):
    text = _read(rel)
    assert "cp313" in text.lower() or "CPython 3.13" in text
    assert "not drop-in" in text or "not drop-in" in text.lower()
    assert "26421e953" in text
    assert "0.32.2" in text
