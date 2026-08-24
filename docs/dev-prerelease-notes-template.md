# oMLX VERSION dev prerelease

> Development build for testing. This is a GitHub **prerelease**, not a stable
> release. Replace every `TODO` and remove claims without linked evidence before
> publishing.

Source commit: `TODO`

Signed artifact: `oMLX-TODO-macos15-26-arm64.dmg`

SHA-256: `TODO` (also supplied as the attached `.sha256` file)

Signing: Developer ID Application, hardened runtime, secure timestamp

Notarization: Apple accepted and ticket stapled to both app and DMG

## What testers should exercise

- In-flight cancel during prefill and decode; record cancel-to-stop latency.
- Steering/replacement requests and preservation of the valid reusable prefix.
- Concurrent requests while another request is cancelled or steered.
- Hot in-memory prefix/KV reuse and SSD-tier spill, restore, and restart reuse.
- `TODO: other release-specific behavior and rollback gates`.

Please attach the request ID, diagnostics export, exact reproduction steps, and
whether the run was cold/warm, cache miss/hit, and single/concurrent. Do not
attach API keys, private prompts, model credentials, or signing material.

## Benchmark provenance

Optimization reference hardware:

- Apple M3 Ultra with 256 GB unified memory (rank 0);
- Apple M5 Max with 128 GB unified memory (rank 1);
- direct Thunderbolt 5 link using JACCL RDMA;
- DeepSeek-V4-Flash-0731-MXFP4-MLX (DS4 MXFP4);
- signed 3:5 tensor-parallel split (M3 Ultra:M5 Max);
- MTP fixed verification depth 5 when explicitly stated.

These details describe where the linked benchmarks came from, not minimum app
requirements or guaranteed performance on other Macs. A 3:5 split is the
tensor partition, not eight machines. See
[`docs/benchmark-provenance.md`](benchmark-provenance.md) for the required
methodology and cache/concurrency labels.

| Scenario | Prompt / output | Cache state | Concurrency | Median tok/s | Runs / spread | Evidence |
| --- | --- | --- | --- | ---: | --- | --- |
| End-to-end decode | `TODO` | warm-process / prefix-miss | single-stream | TODO | TODO | TODO |
| End-to-end decode | `TODO` | warm-process / hot-prefix-hit | single-stream | TODO | TODO | TODO |
| End-to-end decode | `TODO` | warm-process / SSD-prefix-hit | single-stream | TODO | TODO | TODO |
| Aggregate decode | `TODO` | warm-process / prefix-miss | concurrent-TODO | TODO | TODO | TODO |
| Cold prefill | `TODO` | cold-process / prefix-miss | single-stream | TODO | TODO | TODO |

Do not publish a `1,000-1,300 tok/s` target, a modeled roofline, or a
kernel-only result as measured end-to-end throughput. State baseline and
candidate commits, use identical prompts/settings, and separate per-request
from aggregate throughput.

## Compatibility and known limits

- Apple Silicon, arm64; packaged for supported macOS 15-26 hosts.
- The first install from an upstream-team build is manual. Subsequent fork
  updates require the same Apple signing team as the installed fork build.
- `TODO: model/backend compatibility changes`.
- `TODO: known issues and rollback switches`.

## Verification

After downloading both assets:

```sh
shasum -a 256 -c oMLX-TODO-macos15-26-arm64.dmg.sha256
xcrun stapler validate oMLX-TODO-macos15-26-arm64.dmg
spctl --assess --type open --verbose=2 \
  --context context:primary-signature oMLX-TODO-macos15-26-arm64.dmg
```

The checksum proves byte identity; Gatekeeper/stapler checks confirm the signed
and notarized distribution. Do not bypass Gatekeeper for this build.
