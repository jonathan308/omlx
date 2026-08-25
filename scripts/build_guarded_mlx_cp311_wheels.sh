#!/usr/bin/env bash
# Build a matched CPython 3.11 mlx + mlx-metal wheel pair from
# jonathan308/mlx@26421e953 for the Fusion DMG.
#
# Metal-only. Do not run on Linux. Does not invent wheel bytes.

set -euo pipefail

PIN_VERSION="0.32.1.dev20260825+26421e953"
PIN_COMMIT="26421e953"
PIN_REPO="https://github.com/jonathan308/mlx.git"
NANOBIND_PIN="2.13.0"
MACOS_TARGET="26.0"

die() {
    echo "build_guarded_mlx_cp311_wheels.sh: error: $*" >&2
    exit 1
}

[ "$(uname -s)" = "Darwin" ] || die "Metal wheels must be built on macOS, not $(uname -s)"
[ "$(uname -m)" = "arm64" ] || die "expected Apple Silicon arm64, got $(uname -m)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
OUT_DIR="$REPO_ROOT/packaging/_wheels/guarded"
mkdir -p "$OUT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.11 || command -v python3 || true)}"
[ -n "$PYTHON_BIN" ] || die "python3.11 not found"
"$PYTHON_BIN" - <<'PY' || die "interpreter is not CPython 3.11"
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY

"$PYTHON_BIN" -m pip install -q "nanobind==$NANOBIND_PIN" "cmake>=3.27" \
    "setuptools" "wheel" "pip"

SRC_DIR="${OMLX_MLX_SRC:-}"
cleanup() {
    if [ -n "${TMP_SRC:-}" ]; then
        rm -rf "$TMP_SRC"
    fi
}
trap cleanup EXIT

if [ -z "$SRC_DIR" ]; then
    TMP_SRC="$(mktemp -d "${TMPDIR:-/tmp}/omlx-mlx-src.XXXXXX")"
    git clone --filter=blob:none "$PIN_REPO" "$TMP_SRC"
    git -C "$TMP_SRC" checkout --detach "$PIN_COMMIT"
    SRC_DIR="$TMP_SRC"
fi

actual="$(git -C "$SRC_DIR" rev-parse HEAD)"
[[ "$actual" == "$PIN_COMMIT"* ]] \
    || die "mlx checkout $actual is not $PIN_COMMIT"

export MACOSX_DEPLOYMENT_TARGET="$MACOS_TARGET"
export CMAKE_ARGS="${CMAKE_ARGS:+$CMAKE_ARGS }-DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOS_TARGET}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(sysctl -n hw.ncpu)}"

DIST="$SRC_DIR/dist"
rm -rf "$DIST"
mkdir -p "$DIST"

(
    cd "$SRC_DIR"
    "$PYTHON_BIN" setup.py clean --all
    MLX_BUILD_STAGE=1 "$PYTHON_BIN" setup.py bdist_wheel
    "$PYTHON_BIN" setup.py clean --all
    MLX_BUILD_STAGE=2 "$PYTHON_BIN" setup.py bdist_wheel
)

shopt -s nullglob
wheels=("$DIST"/*.whl)
[ "${#wheels[@]}" -gt 0 ] || die "no wheels produced in $DIST"

found_mlx=0
found_metal=0
for whl in "${wheels[@]}"; do
    base="$(basename "$whl")"
    case "$base" in
        *0.32.2*) die "produced $base contains forbidden 0.32.2" ;;
        *cp313*) die "produced $base is cp313; need cp311" ;;
    esac
    case "$base" in
        mlx-*"$PIN_VERSION"*cp311*) found_mlx=1 ;;
        mlx_metal-*"$PIN_VERSION"*) found_metal=1 ;;
    esac
    cp "$whl" "$OUT_DIR/$base"
    echo "copied $base"
done

if [ "$found_mlx" -ne 1 ] || [ "$found_metal" -ne 1 ]; then
    echo "warning: expected version string $PIN_VERSION was not both frontend and backend." >&2
    echo "mlx setup.py stamps .devYYYYMMDD from the build date. The DMG pin is exact;" >&2
    echo "rebuild so metadata matches $PIN_VERSION or stop and inspect $OUT_DIR." >&2
    ls -1 "$OUT_DIR"
    exit 1
fi

echo "Guarded cp311 pair ready in $OUT_DIR"
echo "Pin: $PIN_VERSION (commit $PIN_COMMIT, nanobind $NANOBIND_PIN, macOS $MACOS_TARGET)"
