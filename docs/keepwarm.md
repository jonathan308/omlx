# Latent keepwarm

oMLX Fusion includes an opt-in keepwarm path derived from ThunderMLX. It
addresses two independent idle penalties:

- a bounded Metal matmul keeps the GPU command path and clocks responsive;
- distributed ranks additionally exchange a tiny message through the actual
  MLX/JACCL point-to-point data path, keeping the RDMA queue pairs exercised.

The implementation does not run a competing Metal or collective thread.
Single-node touches execute on the engine's serialized MLX executor.
Distributed touches are carried in MLX-LM's rank-symmetric request-sharing
envelope and execute on every rank's generation stream. A data-plane failure
terminates the affected rank so the existing supervisor replaces the complete
communicator before another model collective.

## Enable

Keepwarm is disabled by default:

```bash
export OMLX_KEEPWARM=1
```

Restart oMLX and reload resident models after changing the setting. Cluster
launches propagate one coordinator-owned value to every rank.

Defaults when enabled:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `OMLX_KEEPWARM_INTERVAL_SECONDS` | `10` | Periodic idle touch cadence |
| `OMLX_KEEPWARM_IDLE_AFTER_SECONDS` | `2` | Idle gap before periodic warming |
| `OMLX_KEEPWARM_MATRIX_SIZE` | `1` | Idle fp16 matmul width |
| `OMLX_KEEPWARM_REQUEST_START` | `1` | Warm immediately before a request after idle |
| `OMLX_KEEPWARM_REQUEST_START_IDLE_SECONDS` | `2` | Request-start idle threshold |
| `OMLX_KEEPWARM_REQUEST_START_MATRIX_SIZE` | `128` | Request-start matmul width |
| `OMLX_KEEPWARM_POST_RESPONSE` | `1` | Delayed follow-up-turn warm |
| `OMLX_KEEPWARM_POST_RESPONSE_DELAY_SECONDS` | `5` | Follow-up warm delay |
| `OMLX_KEEPWARM_POST_RESPONSE_MATRIX_SIZE` | `128` | Follow-up matmul width |
| `OMLX_KEEPWARM_LARGE_CACHE_TOKENS` | `8192` | Large-cache cadence threshold |
| `OMLX_KEEPWARM_LARGE_CACHE_INTERVAL_SECONDS` | `60` | Large-cache periodic cadence |
| `OMLX_KEEPWARM_SLOW_THRESHOLD_SECONDS` | `1` | Slow-touch alarm threshold |
| `OMLX_KEEPWARM_SLOW_BACKOFF_SECONDS` | `60` | Backoff after a slow touch |
| `OMLX_CLUSTER_KEEPWARM_DATAPLANE_PING` | `1` | Exercise JACCL when distributed |

Matrix sizes and repeats are bounded by the parser. Keepwarm actions are not
user inference requests and do not appear in request totals or throughput.
Engine and rank metrics expose the policy, count, skips, failures, slow count,
last action, elapsed time, and whether a data-plane ping ran.

## Qualification

Compare back-to-back TTFT with requests sent after 5, 15, and 60 seconds of
idle, first with keepwarm disabled and then enabled. Record cold-prefix
prefill/decode separately from cache hits. Also check power, thermals, desktop
responsiveness, concurrent admission, cancellation, and rank recovery after a
forced ping failure. Do not enable it by default for a public build until those
gates pass on both single-node and clustered hardware.
