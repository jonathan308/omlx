#!/usr/bin/env bash
# Install the guarded MLX cp311 pair with --no-deps / --no-index.
#
# Fusion 3a82bd27 referenced this path from pyproject.toml but the file was
# missing, so a fresh uv/pip resolve could still land stock PyPI mlx 0.32.0
# (the old [tool.uv] override) or 0.32.2 (the open 0.32.x range). This script
# is the operator install path: local wheels only, exact pin, never PyPI.
#
# This does NOT compile Metal, does NOT run setup.py build_ext, and does NOT
# SSH to / mutate the live Fusion cluster ranks while the Studio is serving
# DS4. If packaging/_wheels/guarded is empty, fail and wait — do not invent
# wheel bytes and do not fall back to stock.

set -euo pipefail

PIN_VERSION="0.32.1.dev20260825+26421e953"
PIN_COMMIT="26421e953"
PYTHON_BIN="${PYTHON_BIN:-}"
CHECK_ONLY=0

die() {
    echo "ops/install_mlx_variant.sh: error: $*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage: ops/install_mlx_variant.sh [--check] [--python PATH]

Install mlx/mlx-metal == ${PIN_VERSION} (jonathan308/mlx@${PIN_COMMIT})
from packaging/_wheels/guarded into a CPython 3.11 environment.

  --check         Validate local wheels only; do not pip install
  --python PATH   Interpreter (default: PYTHON_BIN, python3.11, or python3)

Refuses: stock / PyPI 0.32.0, stock / PyPI 0.32.2, cp313 wheels, Metal compile,
live-cluster rank installs. Uninstalls mlx+mlx-metal together, then installs
both wheels with --no-deps --no-index (never a single-package force-reinstall).
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --check)
            CHECK_ONLY=1
            shift
            ;;
        --python)
            [ $# -ge 2 ] || die "--python requires a path"
            PYTHON_BIN="$2"
            shift 2
            ;;
        stock|0.32.0|0.32.2)
            die "refusing '$1': that restores stock PyPI MLX. Only ${PIN_VERSION} is allowed"
            ;;
        -*)
            die "unknown option $1"
            ;;
        *)
            die "unknown argument '$1' (no variant labels; wheels live in packaging/_wheels/guarded)"
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
WHEEL_DIR="$REPO_ROOT/packaging/_wheels/guarded"

[ -d "$WHEEL_DIR" ] || die "missing $WHEEL_DIR"

shopt -s nullglob
wheels=("$WHEEL_DIR"/*.whl)
if [ "${#wheels[@]}" -eq 0 ]; then
    die "no .whl files in $WHEEL_DIR. Do not compile Metal on the live DS4 Studio / cluster coordinator. Wait for a later window or a GitHub-hosted macos-26 runner that is not that machine, then run scripts/build_guarded_mlx_cp311_wheels.sh with OMLX_I_AM_NOT_THE_LIVE_CLUSTER_COORDINATOR=1. Do not pip install mlx==0.32.0 or mlx==0.32.2 from PyPI."
fi

found_mlx=0
found_metal=0
for whl in "${wheels[@]}"; do
    base="$(basename "$whl")"
    case "$base" in
        *0.32.2*) die "refusing $base: stock/PyPI 0.32.2 (TP2 JACCL all-reduce loss, rank exit 75)" ;;
        *0.32.0*) die "refusing $base: stock/PyPI 0.32.0" ;;
        *cp313*) die "refusing $base: cp313 is not drop-in for the bundled CPython 3.11 runtime" ;;
    esac
    case "$base" in
        mlx-*"$PIN_VERSION"*cp311*) found_mlx=1 ;;
        mlx_metal-*"$PIN_VERSION"*|mlx-metal-*"$PIN_VERSION"*) found_metal=1 ;;
    esac
done

[ "$found_mlx" -eq 1 ] || die "no mlx-*${PIN_VERSION}*cp311* frontend wheel in $WHEEL_DIR"
[ "$found_metal" -eq 1 ] || die "no mlx_metal-*${PIN_VERSION}* backend wheel in $WHEEL_DIR"

echo "ops/install_mlx_variant.sh: guarded pair ${PIN_VERSION} present in $WHEEL_DIR"

if [ "$CHECK_ONLY" -eq 1 ]; then
    echo "check ok (no install, no compile)"
    exit 0
fi

if [ -z "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3.11 || command -v python3 || true)"
fi
[ -n "$PYTHON_BIN" ] || die "python3.11 not found (pass --python)"

"$PYTHON_BIN" - <<'PY' || die "interpreter is not CPython 3.11; the Fusion DMG / this pin is cp311"
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY

# Uninstall both packages first. Never force-reinstall a single package
# (rpath breakage between mlx and mlx-metal).
"$PYTHON_BIN" -m pip uninstall -y mlx mlx-metal >/dev/null 2>&1 || true

# --no-index: a missing local wheel must fail, not silently pull PyPI 0.32.0/0.32.2.
"$PYTHON_BIN" -m pip install \
    --no-deps \
    --no-index \
    --find-links "$WHEEL_DIR" \
    "mlx==${PIN_VERSION}" \
    "mlx-metal==${PIN_VERSION}"

"$PYTHON_BIN" - <<PY
import importlib.metadata as metadata
import sys

pin = "${PIN_VERSION}"
forbidden = ("0.32.0", "0.32.2")
for name in ("mlx", "mlx-metal"):
    version = metadata.version(name)
    if version != pin:
        print(f"ops/install_mlx_variant.sh: error: {name}=={version}, expected {pin}", file=sys.stderr)
        raise SystemExit(1)
    if version in forbidden:
        print(f"ops/install_mlx_variant.sh: error: {name} resolved to forbidden {version}", file=sys.stderr)
        raise SystemExit(1)
print(f"installed mlx=={pin} mlx-metal=={pin} (no PyPI, no compile)")
PY
