# DS4 TP prefill MoE tail8 probe

This is an isolated, unbuilt strict-lossless probe for the next routed-MoE
prefill experiment. The public TP2 model currently owns the M3 GPU, so no new
GPU timing was taken and nothing is wired into CMake, bindings, model dispatch,
the server, or the remote node.

## Candidate: dynamic BM8 route microtiles inside one BM32 block

The M=1024 fixture has 6,144 routes and exactly 24 rows per expert. Current
BM32 Steel kernels multiply 32 rows, including eight zero-padded rows. Simply
changing the block builder to BM24 would be wrong for real skewed routing:
experts with 25–32 rows would gain a second block and reread their weights.

The prototype keeps the BM32 block list and stages every MXFP4 weight tile
once. It represents the route dimension as four BM8 accumulator microtiles:

- rows 0–7 always execute;
- rows 8–15 execute only when `rows > 8`;
- rows 16–23 execute only when `rows > 16`;
- rows 24–31 execute only when `rows > 24`.

All conditions are uniform across the threadgroup. The weight and activation
loaders advance once per BK32 step; valid microtiles keep the same K sequence,
FP32 MMA accumulation, FP16 projection/down store, stable MLX `Sigmoid`, and
LimitedSwiGLU ordering. The pair kernel still shares X and avoids the 24 MiB
pair temporary. A matching down prototype applies the same route-row cull.

The unbuilt source is
`benchmarks/prototypes/ds4_tp_prefill_moe_tail8.metal`. Future isolated symbols
are frozen as:

```text
deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8(...)
deepseek_mxfp4_gather_qmm_blocks_tail8(...)
```

Neither symbol exists in the current native extension.

The standalone source compiled cleanly to a temporary AIR object with the
shipping Metal toolchain and the same strict flags as the native target. It
was not linked, installed, or executed on the GPU.

## Why this is the remaining plausible ds4-metal transfer

The latest ds4-metal resident-prefill bundle combines a compact expert map,
pair/down tail-SIMDgroup culls, and exact MXFP4 dequantization. It measured
10.7% faster full-model M3 prefill at 1,024 tokens, with bit-identical logits.
oMLX already rejected its first-pass half-LUT port (-5.5% warmed prefill) and
the shared-X phase-A kernel alone (8.209 ms versus 8.017 ms pair-concat). The
tail cull is structurally different: it changes no dequantization, block count,
weight reads, or materialized dtype, and removes only padded MMA instructions.

For the uniform fixture:

- current padded MMA rows: 8,192;
- tail8 MMA rows: 6,144;
- row-MMA reduction: 25%;
- MMA-only ceiling: 1.333x;
- weight-read amplification: 1.0x.

For a Poisson sensitivity model with mean 24 routes/expert, the expected MMA
row reduction is 17.9% and the MMA-only ceiling is 1.218x. This model is not a
substitute for capturing real router counts.

## Measured break-even from the rejected phase A

The existing M3 measurements are 8.209271 ms for shared-X BM32 and 8.017229 ms
for pair-concat. On the uniform 24-row fixture, tail8 only needs route-row MMA
to represent 9.36% of shared-X time to break even. Passing the 1.05x promotion
gate requires a 27.97% MMA share. Under Poisson mean-24 padding, those
thresholds rise to about 13.1% and 39.1%.

Sensitivity using the uniform fixture:

| Route-row MMA share | Projected tail8 time | Speedup vs pair-concat |
|---:|---:|---:|
| 10% | 8.004 ms | 1.002x |
| 20% | 7.799 ms | 1.028x |
| 30% | 7.594 ms | 1.056x |
| 40% | 7.388 ms | 1.085x |
| 50% | 7.183 ms | 1.116x |

These are projections, not measurements. M5 should continue using its faster
stock NAX path; the tail8 probe is an M3-family candidate only.

## Why a single no-mid fused kernel is blocked

The M3 Ultra reports `maxThreadgroupMemoryLength = 32768` bytes. A persistent
kernel must retain `BM * 1024 * 2` bytes of FP16 activated mid plus at least
the X and two MXFP4 weight staging tiles:

| BM | Mid | Minimum total with pair staging | Fits 32 KiB? |
|---:|---:|---:|:---:|
| 8 | 16 KiB | 21.6 KiB | yes |
| 16 | 32 KiB | 38.3 KiB | no |
| 24 | 48 KiB | 55.5 KiB | no |
| 32 | 64 KiB | 72.8 KiB | no |

BM8 fits, but 24 routes require three complete expert passes, rereading gate,
up, and down weights 3x. Keeping 24 rows in one threadgroup needs 48 KiB for
mid before any staging. Therefore a strict-lossless gate/up→SwiGLU→down
persistent kernel cannot simultaneously avoid mid and preserve one weight
read on this GPU. Recomputing gate/up per output supertile is worse.

## Next safe gate

When the public model is deliberately unloaded, a native agent can wire the
two isolated symbols and run:

```bash
python benchmarks/bench_ds4_tp_prefill_moe_tail8.py \
  --model /path/to/DS4-Flash --rank 0 --strict
```

The harness gates the pair, down, and composed gate/up/activation/down
projection boundaries separately. Promotion requires `mx.array_equal` at all
three and at least 1.05x for the composed projection versus the current M3
path. The separate phase-B deterministic reduction must then preserve local
routed output, followed by a real route-count capture and cold-prefill
full-model gate. Until then this remains an evidence-backed prototype, not a
speed claim.
