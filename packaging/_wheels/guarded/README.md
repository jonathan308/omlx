# Guarded CPython 3.11 MLX wheels

This directory is the only allowed source of `mlx` / `mlx-metal` for the
bundled Fusion DMG runtime. Wheels are **not** committed; drop the matched
pair here before `packaging/build.py --venvstacks-only` or
`ops/install_mlx_variant.sh`. That script installs with `--no-deps --no-index`
and refuses stock 0.32.0 / 0.32.2.

## Pin

| Field | Value |
| --- | --- |
| Version | `0.32.1.dev20260825+26421e953` |
| Commit | `26421e953` on `jonathan308/mlx` branch `feat/jaccl-subgroups` |
| nanobind | `2.13.0` |
| Python ABI | **cp311** (the DMG embeds `cpython-3.11`) |
| Platform | `macosx_26_0_arm64` (Path B: macOS 26 only) |
| Forbidden | PyPI `mlx==0.32.0`, PyPI/stock `mlx==0.32.2` |

TP2 rejected 0.32.2 (JACCL all-reduce loss, rank exit 75, ~92-94 GiB wired
on M5). Do not package it.

## CPython 3.13 wheels are not drop-in

Wheels under `minimax-m3-cluster/runtime_patches/variants/guard-0321-rollback/`
were built for **CPython 3.13**. They cannot be copied into this cp311
venvstacks layer. A matched **frontend + backend** pair must be rebuilt from
`26421e953` with Python 3.11.

## Build the pair later (not the live Studio)

Do **not** compile Metal, run `setup.py build_ext`, or run
`apps/omlx-mac/Scripts/build.sh` on the Fusion Studio that is serving
live DS4, and do not send this to a self-hosted Mac that is the cluster
coordinator. This Linux cloud VM also cannot compile Metal.

When an idle GitHub-hosted macos-26 runner or later window exists:

```bash
scripts/build_guarded_mlx_cp311_wheels.sh
```

Expected filenames (do not invent bytes; produce them on macOS):

```
mlx-0.32.1.dev20260825+26421e953-cp311-cp311-macosx_26_0_arm64.whl
mlx_metal-0.32.1.dev20260825+26421e953-py3-none-macosx_26_0_arm64.whl
```

See `docs/packaging-guarded-mlx.md`.
