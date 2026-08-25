# SPDX-License-Identifier: Apache-2.0
"""The Fusion DMG must pin guarded MLX and must not admit 0.32.0 or 0.32.2."""

import re
import subprocess
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
    overrides = text.split("[tool.uv]", 1)[1].split("[tool.", 1)[0]
    quoted = re.findall(r'"([^"]+)"', overrides)
    assert f"mlx=={PIN}" in quoted
    assert f"mlx-metal=={PIN}" in quoted
    assert not any(q in {"mlx==0.32.0", "mlx==0.32.2"} for q in quoted)
    assert "ops/install_mlx_variant.sh" in text


def test_packaging_build_never_queries_pypi_for_mlx():
    text = _read("packaging/build.py")
    assert "pypi.org/pypi/mlx" not in text
    assert "GUARDED_MLX_VERSION" in text
    assert PIN in text
    assert 'FORBIDDEN_MLX_VERSIONS = ("0.32.0", "0.32.2")' in text


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


def test_live_studio_coordinator_is_off_limits_for_metal_compile():
    docs = " ".join(
        _read(rel)
        for rel in (
            "docs/packaging-guarded-mlx.md",
            "docs/release-public-checklist.md",
            "scripts/build_guarded_mlx_cp311_wheels.sh",
            ".github/workflows/release-macos-dmg.yml",
        )
    )
    assert "serving live DS4" in docs or "serving DS4" in docs
    assert "cluster coordinator" in docs
    assert "OMLX_I_AM_NOT_THE_LIVE_CLUSTER_COORDINATOR" in _read(
        "scripts/build_guarded_mlx_cp311_wheels.sh"
    )
    assert "github-hosted" in _read(".github/workflows/release-macos-dmg.yml")


def test_install_mlx_variant_script_closes_fusion_hole():
    path = ROOT / "ops" / "install_mlx_variant.sh"
    assert path.is_file()
    text = path.read_text()
    assert PIN in text
    assert "--no-deps" in text
    assert "--no-index" in text
    assert "0.32.0" in text
    assert "0.32.2" in text
    assert "ssh " not in text
    assert "does NOT compile Metal" in text
    subprocess.run(["bash", "-n", str(path)], check=True)
    # Wheels are not committed; --check must fail closed instead of PyPI.
    result = subprocess.run(
        ["bash", str(path), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    err = result.stderr.lower()
    assert "0.32.0" in err or "no .whl" in err or "missing" in err
    stock = subprocess.run(
        ["bash", str(path), "stock"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert stock.returncode != 0
    assert "0.32.0" in stock.stderr or "stock" in stock.stderr.lower()
