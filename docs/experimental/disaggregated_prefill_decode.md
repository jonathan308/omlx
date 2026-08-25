# Disaggregated prefill/decode over JACCL RDMA

Status: physically proven prototype, default off, not connected to the public
HTTP scheduler.

## Contract

Two workers load identical full model replicas. The selected prefill worker
processes every prompt token and samples the first output token. It then sends:

1. a bounded manifest containing model identity, prompt length, MLX-LM cache
   class names, tensor tree paths, shapes, dtypes and `meta_state` strings;
2. every cache tensor directly with `mx.distributed.send`; and
3. the first sampled token.

The decode worker verifies the model identity, receives each tensor with
`mx.distributed.recv`, reconstructs the original MLX-LM cache classes through
their `from_state` contract, and continues generation from the first token.
No cache tensor is copied through Python bytes, HTTP, SSD or the CPU.

Readiness and flow control use the existing `RankControlPlane` TCP/system-
Python proxy. JACCL is idle throughout arbitrarily long prefill and is entered
only after both workers are ready to transfer cache tensors. This separation is
load-bearing: an early prototype used periodic RDMA heartbeats and eventually
hit the 30-second progress guard during a 30K prefill.

## Capability rule

The current topology requires the complete model and its admitted cache budget
to fit independently on both workers. Qwen3.8-27B-4bit is about 15 GB and fits
comfortably on both test Macs. DeepSeek-V4-Flash does not fit as a full replica
on the 128 GB M5 Max (its two current tensor shards together measure about
171 GB), so DS4 needs a future form where each logical prefill/decode worker is
itself a shard group.

The cache codec is model-independent. A model is admitted when all cache state
leaves are MLX arrays and every cache class is an installed MLX-LM class with a
`from_state` method. Unknown cache classes, non-array leaves, model-identity
mismatches, malformed shapes/dtypes and unbalanced byte ledgers fail before
decode.

## Physical evidence

Hardware:

- rank 0: Apple M3 Ultra, 256 GB;
- rank 1: Apple M5 Max, 128 GB;
- JACCL over direct Thunderbolt RDMA;
- patched MLX `0.32.2.dev20260825+ceab91938`;
- model: `Qwen3.8-27B-4bit`, identity
  `f4ad23f9019f77fcc8e494ff76423e577cdd48fac923f93fb83c6b5f3872b022`.

| prompt | prefill role | decode role | prefill | cache bytes | handoff | wire rate | decode | parity |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 512 | M3 | M5 | 302.33 tok/s | 187.50 MB | 42.58 ms | 4.40 GB/s | 30.15 tok/s | 32/32 exact |
| 512 | M5 | M3 | 874.77 tok/s | 187.50 MB | 39.28 ms | 4.77 GB/s | 29.37 tok/s | 32/32 exact |
| 4,096 | M5 | M3 | 1,005.74 tok/s | 422.38 MB | 64.79 ms | 6.52 GB/s | 31.21 tok/s | 64/64 exact |
| 30,000 | M5 | M3 | 871.82 tok/s | 2.120 GB | 323.98 ms | 6.54 GB/s | 27.37 tok/s | 64/64 exact |

At 30K, the complete cache handoff costs 0.94% of prefill time. At 4K it costs
about 1.0%. Both cache directions and both role assignments passed.

The 4K+64 measurements imply a steady two-stage request interval of roughly
`max(4.073 s prefill, 2.018 s decode) + 0.065 s handoff = 4.138 s`, versus
about `4.073 + 2.012 = 6.085 s` when the M5 performs both phases serially. That
is a projected 1.47x steady request-throughput improvement after pipeline fill;
the first request pays the handoff and therefore has nearly unchanged/slightly
higher latency. A multi-request physical overlap gate is still required before
claiming the projection as measured aggregate throughput.

## Implemented artifacts

- `omlx/cluster/cache_transfer.py`: bounded universal cache manifest,
  reconstruction and direct point-to-point tensor transport.
- `omlx/cluster/disaggregated_worker.py`: two-rank full-replica parity worker,
  role reversal and control/data-plane separation.
- `benchmarks/bench_disaggregated_prefill_decode.py`: configured-fabric launcher
  and durable JSON report.
- `tests/test_cluster_cache_transfer.py` and
  `tests/test_disaggregated_worker.py`: schema/class/tensor guards and scheduler
  frontier helpers.

## Before serving integration

1. Add a planner capability that proves full-replica fit on both nodes and
   measures both role directions. Choose the orientation that minimizes the
   expected pipeline interval for the configured prompt/decode workload.
2. Add a persistent two-stage request queue. While decode handles request N,
   prefill should process N+1; cache transfer occurs at the stage boundary.
3. Preserve request IDs, cancellation, steering, grammar state, sampler state,
   logit processors, tool-call parsing and per-request telemetry across the
   ownership transition.
4. Add bounded cache-transfer admission, timeout/CRC failure handling, receiver
   teardown, and cache ownership/garbage collection.
5. Gate MTP/speculative state separately; the current prototype proves fixed
   greedy decode only.
6. Run two-request physical overlap, B2/B4 queues, 30K/100K cache transfer,
   cancellation during prefill/handoff/decode, forced rank loss and reload.
7. Expose the mode only when the planner proves it useful. Models that do not
   fit twice, single-request interactive workloads, and slow fabrics should
   remain on tensor/pipeline/single-node execution automatically.
