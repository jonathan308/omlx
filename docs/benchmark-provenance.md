# Benchmark hardware and result provenance

Performance numbers published for this fork must identify the source commit,
physical topology, workload, cache state, concurrency, and statistic. A bare
tokens-per-second value is not reproducible and must not be presented as a
general oMLX result.

## DS4 optimization reference topology

The current DeepSeek-V4-Flash optimization campaign uses this heterogeneous
Apple Silicon setup:

| Component | Reference configuration |
| --- | --- |
| Rank 0 | Apple M3 Ultra, 256 GB unified memory |
| Rank 1 | Apple M5 Max, 128 GB unified memory |
| Interconnect | Direct Thunderbolt 5 link |
| Collective backend | JACCL Thunderbolt RDMA |
| Model | DeepSeek-V4-Flash-0731-MXFP4-MLX (DS4 MXFP4) |
| Tensor-parallel split | 3:5 (M3 Ultra:M5 Max) |
| Speculative verification | MTP fixed depth 5, when explicitly enabled |

The 3:5 notation describes the signed tensor-parallel partition, not a machine
count. Results from a single-rank microbenchmark, a modeled JACCL transfer, or
only one of the two machines must say so; those results are not end-to-end
two-host measurements. Record whether the direct TB5/JACCL path was physically
active and measured rather than inferred from a bandwidth model.

## Required run metadata

Every published benchmark table or release note must include:

- exact git commit and whether the worktree was clean;
- oMLX, Python, MLX, MLX-LM, macOS, Xcode/Metal, and JACCL versions;
- model repository/revision plus a stable artifact or manifest hash;
- both device names, unified-memory capacities, power modes, and thermal state;
- physical link, negotiated Thunderbolt topology, RDMA readiness, and backend;
- tensor-parallel split and rank assignment;
- prompt and generated-token counts, sampling settings, and stop conditions;
- DFlash/MTP mode and depth, including acceptance rate when speculation is on;
- per-request decode throughput and aggregate throughput as separate values;
- warmup count, measured repetitions, statistic (median/p50/p95), and raw log;
- active feature/rollback gates and any non-default environment variables.

Avoid embedding local usernames or absolute model paths in published logs.

## Cache, temperature, and concurrency labels

Use all applicable labels. Do not shorten them to simply `cached` or `warm`.

| Dimension | Required label | Meaning |
| --- | --- | --- |
| Process/model | `cold-process` | New process and model load; no prior Metal pipeline or allocator warmup |
| Process/model | `warm-process` | Same process/model after declared warmup passes |
| Prompt reuse | `prefix-miss` | Zero reusable prompt tokens; report `cached_tokens=0` |
| Prompt reuse | `hot-prefix-hit` | Reused from the in-memory prompt/KV tier; report hit tokens and hit rate |
| Prompt reuse | `ssd-prefix-hit` | Restored from the SSD tier; report hit tokens, bytes, and restore time |
| Prompt reuse | `mixed-prefix-hit` | Hot and SSD tiers both contributed; report each tier separately |
| Request load | `single-stream` | Exactly one active generation request |
| Request load | `concurrent-N` | N active requests; report per-request and aggregate tok/s |
| Input identity | `fixed-prompt` | Repeated identical prompt, useful for cache-hit tests |
| Input identity | `independent-prompts` | Distinct prompts, suitable for cache-miss and sustained-load tests |

For tiered-cache claims, a qualifying run includes an initial miss, an
in-memory reuse, an eviction/spill into the SSD tier, a subsequent SSD restore,
and reuse after process restart when persistence is being claimed. Report cache
hit tokens and cache-source telemetry alongside timing. A warm Metal pipeline
with a new KV cache is `warm-process / prefix-miss`, not a fully cold run.

For cancellation or steering measurements, also record time from client action
to stopped token production, whether the request had entered prefill or decode,
whether a replacement/steered request reused the valid prefix, and whether
other concurrent requests continued without interruption.

## Release benchmark matrix

At minimum, a performance-bearing prerelease should link raw evidence for:

1. cold-process, prefix-miss, single-stream;
2. warm-process, prefix-miss, single-stream;
3. warm-process, hot-prefix-hit, single-stream;
4. warm-process, SSD-prefix-hit, single-stream;
5. warm-process, prefix-miss, concurrent-N;
6. warm-process, prefix-hit, concurrent-N;
7. the same declared DS4/MTP depth-5 and 3:5 TP configuration used for the
   headline comparison.

If a cell was not run, mark it `not measured`; do not silently substitute a
projection, kernel-only measurement, or a different topology.
