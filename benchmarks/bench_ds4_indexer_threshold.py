#!/usr/bin/env python3
"""Measure the DS4 ratio-4 prefill-indexer slope around pooled N=4096."""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx

from omlx.custom_kernels.glm_moe_dsa import fast


def _measure(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append((time.perf_counter() - started) * 1e3)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-tokens", type=int, default=512)
    parser.add_argument(
        "--pooled-tokens",
        default="3584,3840,4096,4352,4608,5120,6144,8192",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--layers", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not fast.has_symbol("dsa_indexer_scores") or not fast.has_symbol(
        "dsa_topk_indices"
    ):
        raise SystemExit("native DS4 indexer kernels are unavailable")
    pooled_lengths = tuple(int(value) for value in args.pooled_tokens.split(","))
    if args.query_tokens < 2 or any(value <= 512 for value in pooled_lengths):
        raise SystemExit("query tokens must be >=2 and pooled lengths >512")

    mx.random.seed(args.seed)
    query_tokens = args.query_tokens
    q = mx.random.uniform(-0.5, 0.5, (1, 64, query_tokens, 128)).astype(mx.bfloat16)
    weights = mx.random.uniform(-0.5, 0.5, (1, query_tokens, 64)).astype(mx.bfloat16)
    mx.eval(q, weights)

    layer_queries = [
        mx.random.uniform(-0.5, 0.5, q.shape).astype(mx.bfloat16)
        for _ in range(args.layers)
    ]
    layer_weights = [
        mx.random.uniform(-0.5, 0.5, weights.shape).astype(mx.bfloat16)
        for _ in range(args.layers)
    ]
    mx.eval(*layer_queries, *layer_weights)

    print(
        "N\tscore_ms\ttopk_ms\tsort_ms\tcombined_ms\t"
        "parallel_layers_ms\tparallel_per_layer_ms"
    )
    for pooled_tokens in pooled_lengths:
        keys = mx.random.uniform(-0.5, 0.5, (1, 1, pooled_tokens, 128)).astype(
            mx.bfloat16
        )
        mx.eval(keys)
        query_offset = pooled_tokens * 4 - query_tokens * 2

        def score(keys=keys, query_offset=query_offset):
            return fast.dsa_indexer_scores(
                q,
                keys,
                weights,
                causal=False,
                mask_ratio=4,
                mask_q_offset=query_offset,
                use_nax=False,
            )

        resident_scores = score()
        mx.eval(resident_scores)

        def topk(resident_scores=resident_scores):
            return fast.dsa_topk_indices(resident_scores, 512, bucketed=False)

        resident_indices = topk()
        mx.eval(resident_indices)
        score_ms = _measure(score, warmup=args.warmup, repeats=args.repeats)
        topk_ms = _measure(topk, warmup=args.warmup, repeats=args.repeats)
        sort_ms = _measure(
            lambda resident_indices=resident_indices: mx.sort(
                resident_indices, axis=-1
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        combined_ms = _measure(
            lambda: mx.sort(
                fast.dsa_topk_indices(score(), 512, bucketed=False),
                axis=-1,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        layer_keys = [
            mx.random.uniform(-0.5, 0.5, keys.shape).astype(mx.bfloat16)
            for _ in range(args.layers)
        ]
        mx.eval(*layer_keys)

        def parallel_graph(layer_keys=layer_keys, query_offset=query_offset):
            outputs = []
            for layer_q, layer_k, layer_w in zip(
                layer_queries, layer_keys, layer_weights
            ):
                layer_scores = fast.dsa_indexer_scores(
                    layer_q,
                    layer_k,
                    layer_w,
                    causal=False,
                    mask_ratio=4,
                    mask_q_offset=query_offset,
                    use_nax=False,
                )
                outputs.append(
                    mx.sort(
                        fast.dsa_topk_indices(layer_scores, 512, bucketed=False),
                        axis=-1,
                    )
                )
            return outputs

        parallel_ms = _measure(
            parallel_graph,
            warmup=max(1, args.warmup // 2),
            repeats=max(3, args.repeats // 2),
        )
        print(
            f"{pooled_tokens}\t{score_ms:.3f}\t{topk_ms:.3f}\t"
            f"{sort_ms:.3f}\t{combined_ms:.3f}\t{parallel_ms:.3f}\t"
            f"{parallel_ms / args.layers:.3f}"
        )


if __name__ == "__main__":
    main()
