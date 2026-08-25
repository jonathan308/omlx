# DS4 exact verify-window kernel plan

Status: active performance workstream; no production dispatch yet.

## Objective

Replace depth-5 DS4 target verification's decode-row composition with one
M=2..6 verify-window kernel. The kernel must preserve the existing M=1 decode
arithmetic while amortizing local/pooled KV reads across all verification
positions.

This is a B1 speculative-verification optimization. It reduces one sequence's
target backbone cycle. True B2 agent scaling additionally needs a sequence
batch dimension, per-sequence committed DSpark contexts, independent accepted
prefixes, rollback, cancellation, and cache ownership.

## Current bottleneck

`decode_fast/sparse_attn_decode` fuses local-window + selected-pooled attention
for one query position and supports B<=8. DSpark target verification represents
M positions as the batch dimension. It therefore executes one `(batch,row)`
attention state per query and reloads local/pooled KV for every verify row.

At 99% draft acceptance the current depth-5 physical gate emits about 5.85
tokens per cycle, but target-backbone work remains about 68.8 ms/cycle. The
100 tok/s target needs below 58.5 ms; 130 needs below roughly 45 ms.

## Candidate ABI

Inputs:

- `q`: `[1,H,M,512]`, M=1..6, BF16/FP16;
- physical local KV source: `[1,1,W,512]` plus exact row-specific ring starts
  or index maps;
- final pooled latent/value source: `[1,S,512]`;
- selected pooled indices: `[1,M,Kmax]`, uint32;
- selected lengths: `[M]`, so the 511/512 pooling boundary is represented
  without padding or duplicated KV;
- sinks: `[H]`, FP32;
- scale and exact row offsets.

Output: `[1,H,M,512]` in the same dtype/order as the existing exact path.

## Arithmetic contract

For every `(head,verify_position)`:

1. preserve current local-row order;
2. preserve selected-pooled top-k order;
3. dot products accumulate in the same order and round to storage dtype at the
   same boundary as `sparse_attn_decode`;
4. local and pooled logsumexp values retain their existing dtype roundings;
5. sink logaddexp remains FP32;
6. local and pooled value partial sums remain separate until the existing final
   rounding point;
7. ragged K uses explicit lengths, never semantic padding;
8. every M>1 row must be byte-identical to invoking the current deployed M=1
   fused decode kernel on that row. The existing fused kernel is accuracy-gated
   against the FP32/composed reference rather than bitwise-equal to it.

The implementation may stage a KV tile once and update M independent score,
normalizer, and value states. It may not change reduction order merely to gain
throughput.

## Kernel sequence

1. Prototype local-window score/value reuse for M=2..6 with a shared physical
   ring source.
2. Add pooled selected-index processing with explicit per-row K lengths.
3. Fuse sinks/normalizer/value passes.
4. Add M=1 dispatch compatibility.
5. Route only exact DS4 verify shapes behind a default-off environment gate.
6. Promote only after full TP2 parity and whole-model speed gates.

## Correctness gates

- BF16 and FP16; M=1,2,3,4,5,6.
- H=24/32/40/64 tensor-rank shapes where supported.
- local W near 0, 1, 127, 128 and full ring wrap.
- pooled K 1, 511, 512 and ragged `[511,512,...]`.
- random, adversarial equal-score/tie, large-magnitude, and sink-dominant data.
- exact row equality with the current M=1 fused decode kernel, plus the existing
  FP32 error bound against the composed reference.
- identical target tokens, acceptance decisions, rollback lengths, cache
  offsets, and completion hashes on single node and TP2.
- cancellation at every verify boundary.

## Performance gates

- report kernel-only M=1..6 median/p95 on M3 Ultra and M5 Max;
- compare current fused-B rows, exact composed reference, and verify-window;
- profile bytes and command buffers, not just wall time;
- require a material whole-model B1 win (initial promotion floor: >=10% target
  backbone reduction and >=5% API decode gain) with zero parity drift;
- retain current path as one-flag rollback.

## Follow-on inventions

1. Absorbed MLA / tighter latent KV for 1M context, only if profiling proves
   expanded cache/materialization traffic is causal and the latent
   representation is lossless.
2. Fused indexer-score to deterministic top-k, preserving FP32 ordering/ties
   while eliminating dense score tensor writes.
3. True sequence-batched DSpark MTP (`N x M`) after B1 verify-window economics
   are proven.

## First cross-chip baseline

Command:

```sh
~/omlx-v2-build/bin/python benchmarks/bench_ds4_verify_window.py \
  --iterations 20 --warmup 3
```

BF16, H=32, local=128, pooled=512. All M=1..6 batch rows were bitwise
identical to independent M=1 fused-kernel calls.

| Chip | M=1 median | M=6 median | Logical KV/pass M=1→M=6 |
| --- | ---: | ---: | ---: |
| M3 Ultra | ~0.419 ms | ~0.280 ms | 0.655→3.932 MB |
| M5 Max | ~0.519 ms | ~0.255 ms | 0.655→3.932 MB |

The current kernel is already one dispatch and gains occupancy at M=4..6,
despite logical KV traffic scaling with M. Therefore a shared-window kernel is
not automatically the dominant latency win; it needs real Metal counters and
whole-backbone attribution. The baseline supports continuing the prototype for
bandwidth/energy and several-millisecond cycle savings, but not claiming it
alone reaches 100–130 tok/s.
