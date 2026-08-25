# Guarded MLX pin for the 0.6.4b1 Fusion DMG

Path B of this first cluster beta ships a **macOS 26-only** signed DMG. The
bundled CPython 3.11 environment must not resolve stock PyPI MLX.

## Pin

- `mlx==0.32.1.dev20260825+26421e953`
- `mlx-metal==0.32.1.dev20260825+26421e953`
- Source: `jonathan308/mlx@26421e953` (`feat/jaccl-subgroups`)
- nanobind **2.13.0** (must match the wheel ABI or custom kernels reject
  every `mlx.core.array`)
- Never `0.32.2`. TP2 rejected 0.32.2 (JACCL all-reduce loss, rank exit 75,
  ~92-94 GiB wired on M5).

`pyproject.toml` previously used `mlx>=0.32.0.dev0,<0.33`, which admits
0.32.2, and `[tool.uv] override-dependencies` forced stock `mlx==0.32.0`.
Both are now the exact local version above. A fresh resolve without the
local wheels must fail rather than land PyPI.

## CPython 3.13 rollback wheels are not drop-in

`minimax-m3-cluster/runtime_patches/variants/guard-0321-rollback/` holds
**cp313** wheels. The DMG embeds **CPython 3.11**. Those files cannot be
copied into `packaging/_wheels/guarded/`. Build a matched cp311
frontend (`mlx`) + backend (`mlx-metal`) pair from `26421e953`.

## Native kernels (same ABI)

Rebuild with `OMLX_WITH_CUSTOM_KERNEL=1` against the guarded cp311 pair:

| Extension | Notes |
| --- | --- |
| `glm_moe_dsa` / DS4 | Sparse MLA, DSA indexer, DSpark, DeepSeek V4 |
| `decode_fast` | Fused decode paths |
| `qwen35_prefill` | Prefill + NAX/ANE metallibs on SDK 26.2+ |
| `minimax_m3` | MiniMax M3 MSA |
| `bonsai` | 1-bit / 2-bit decode |

`apps/omlx-mac/Scripts/build.sh release --with-custom-kernel` is the DMG
path. Default kernel deployment target is **26.0**.

## Next local / macOS step (not the live Studio)

This pin does **not** include wheel bytes. **Do not** run
`scripts/build_guarded_mlx_cp311_wheels.sh`, `setup.py build_ext`, or
`apps/omlx-mac/Scripts/build.sh` on the Studio that is serving live DS4,
and do not send this work to a self-hosted Mac worker / private machine
that is the in-use cluster coordinator.

cp311 `mlx` / `mlx-metal` compile waits for a later window or a
**GitHub-hosted** `macos-26` runner that is not that coordinator. When
that idle Mac exists, with Xcode 26.6 and CPython 3.11:

```bash
export OMLX_I_AM_NOT_THE_LIVE_CLUSTER_COORDINATOR=1
scripts/build_guarded_mlx_cp311_wheels.sh
```

Confirm the produced version string is exactly
`0.32.1.dev20260825+26421e953` (mlx's setup stamps `.devYYYYMMDD` from the
build date plus `+` plus `git rev-parse --short HEAD`). Then rebuild the
venvstacks export and the app on that Mac. Do not create or push tag
`v0.6.4b1` until that pair exists and is reviewed.

## Signing environment

Create GitHub Environment `macos-release` (protected, required reviewers).
Do not commit Apple credentials. In the GitHub UI add secrets:

- `APPLE_DEVELOPER_ID_APPLICATION_P12_BASE64`
- `APPLE_DEVELOPER_ID_APPLICATION_P12_PASSWORD`
- `APPLE_NOTARY_API_KEY_P8_BASE64`
- `APPLE_NOTARY_KEY_ID`
- `APPLE_NOTARY_ISSUER_ID`

and environment variable `APPLE_TEAM_ID`.
