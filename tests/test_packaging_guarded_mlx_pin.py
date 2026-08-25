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


def test_version_is_064b1():
    assert _read("omlx/_version.py").strip() == '__version__ = "0.6.4b1"'


def test_deployment_targets_are_macos26_nax_stays_262():
    setup = _read("setup.py")
    assert 'DEFAULT_CUSTOM_KERNEL_DEPLOYMENT_TARGET = "26.0"' in setup
    assert 'DEFAULT_CUSTOM_KERNEL_DEPLOYMENT_TARGET = "15.0"' not in setup
    build = _read("apps/omlx-mac/Scripts/build.sh")
    assert "OMLX_CUSTOM_KERNEL_DEPLOYMENT_TARGET:-${MACOSX_DEPLOYMENT_TARGET:-26.0}" in build
    assert "--minimum-deployment-target 26.0" in build
    pbx = _read("apps/omlx-mac/oMLX.xcodeproj/project.pbxproj")
    assert "MACOSX_DEPLOYMENT_TARGET = 26.0;" in pbx
    assert "MACOSX_DEPLOYMENT_TARGET = 15.0;" not in pbx
    nax = _read("omlx/custom_kernels/qwen35_prefill/csrc/CMakeLists.txt")
    assert "-mmacosx-version-min=26.2" in nax


def test_build_sh_includes_bonsai_and_decode_fast():
    text = _read("apps/omlx-mac/Scripts/build.sh")
    assert "omlx/custom_kernels/decode_fast" in text
    assert "omlx/custom_kernels/bonsai" in text
    assert "omlx_decode_fast_kernels.metallib" in text
    assert "omlx_bonsai_kernels.metallib" in text


def test_updater_requires_exact_macos26_not_15_26_range():
    src = _read("apps/omlx-mac/Sources/Updater/ReleasesChecker.swift")
    tests = _read("apps/omlx-mac/Tests/oMLXTests/ReleasesCheckerTests.swift")
    assert "macos15-26" in tests
    assert 'tahoeOnly = "oMLX-0.6.4b1-macos26-arm64.dmg"' in tests
    assert "macOSMajor == 26 && range.lowerBound < 26" in src
    assert "pypi.org/pypi/mlx" not in _read("packaging/build.py")
    build = _read("packaging/build.py")
    assert "pip\", \"download\"" not in build.split("def swap_platform_wheels", 1)[1].split("def _parse_git_requirements", 1)[0]
    assert "https://pypi.org/pypi/mlx/json" not in build
