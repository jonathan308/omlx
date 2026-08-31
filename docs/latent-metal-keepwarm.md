# Latent Metal keepwarm (experimental)

Latent Metal keepwarm is an opt-in latency feature for cached follow-up turns
on a loaded local engine. It keeps the Metal command path ready without reading
or mutating model weights, KV state, prompt tokens, prefix-cache entries, or SSD
artifacts.

The mechanism is adapted from the Apache-2.0
[ThunderMLX](https://github.com/jonathan308/ThunderMLX) keepwarm design, with
oMLX-specific continuous-batching, cache-clear, model-load, and live-settings
gates.

## How it works

After useful request activity, idle or post-response work on the engine's
existing one-worker MLX executor lazily creates a dedicated, thread-bound MLX
stream and compiles one safe `mx.fast.metal_kernel`. The kernel dispatches a
single fp32 element with a one-thread grid. Its serving path uses
`mx.async_eval`: it submits without a host read, `mx.eval`, scalar access, or
synchronization barrier, and retains only the latest four-byte output plus its
four-byte input.

A request-start action reuses an already prepared pulse. If idle preparation
has not happened yet, request start performs zero Metal work: no allocation,
stream creation, compilation, or fallback matmul. Active requests and queued
admissions on that engine always win.

The local cadence defaults are physically qualified for the asynchronous pulse:

- periodic idle pulse every 2 seconds;
- post-response pulse after 1 second;
- the same 2-second cadence for a large resident cache.

Explicit environment overrides remain authoritative. On the development M3
Ultra, hot submission after preparation measured 0.12–0.13 ms of host-side
submission time. This is deliberately reported as submission latency—not GPU
completion latency or a claim about end-to-end TTFT.

## Enable and observe

Enable **Settings → Advanced → Performance → Latent Metal keepwarm**. The switch
applies to loaded Batched and VLM engines immediately and is saved for engines
loaded later. It is disabled by default because keeping the GPU command path
active can use slightly more idle power.

Engine telemetry records `execution_mode` as:

- `async_prepared` when an off-path idle action created the stream, compiled
  the kernel, and submitted its first pulse;
- `async_submitted` when a prepared pulse was reused;
- a skip event when request-start preparation was unavailable or request state
  changed.

Elapsed time is submission/preparation time. A failure or a submission exceeding
the slow threshold enters bounded backoff and never makes inference unavailable.

## Safety and lifecycle gates

- no pulse runs before a real request completes or resident cache is observed;
- an idle pulse rechecks queued admissions and scheduler work on the MLX lane;
- stream creation, pulse use, and close share the engine's FIFO executor thread;
- close drains only the dedicated pulse stream before scheduler/model teardown;
- action width and repeats are validated even though the kernel always dispatches
  one element and one thread;
- the serving path retains exactly one four-byte output, not an accumulating
  output list;
- clearing the in-memory cache disarms warming until the next real request;
- unload drops all synthetic arrays, the compiled callable, and the pulse stream;
- cache reuse, cache serialization, SSD writes/restores, sampling, tool output,
  MTP acceptance, and model math are untouched.

## Environment controls

| Variable | Default | Purpose |
| --- | ---: | --- |
| `OMLX_KEEPWARM` | `0` | Master switch |
| `OMLX_KEEPWARM_INTERVAL_SECONDS` | `2` | Periodic idle cadence |
| `OMLX_KEEPWARM_IDLE_AFTER_SECONDS` | `2` | Idle time before periodic pulse |
| `OMLX_KEEPWARM_MATRIX_SIZE` | `1` | Bounded compatibility/action metadata |
| `OMLX_KEEPWARM_REPEATS` | `1` | Pulses per selected action |
| `OMLX_KEEPWARM_REQUEST_START` | `1` | Reuse a prepared pulse at request start |
| `OMLX_KEEPWARM_REQUEST_START_IDLE_SECONDS` | `2` | Request-start idle gate |
| `OMLX_KEEPWARM_REQUEST_START_MATRIX_SIZE` | `128` | Bounded compatibility metadata |
| `OMLX_KEEPWARM_POST_RESPONSE` | `1` | Enable post-response pulse |
| `OMLX_KEEPWARM_POST_RESPONSE_DELAY_SECONDS` | `1` | Post-response delay |
| `OMLX_KEEPWARM_POST_RESPONSE_MATRIX_SIZE` | `128` | Bounded compatibility metadata |
| `OMLX_KEEPWARM_LARGE_CACHE_TOKENS` | `8192` | Long-cache cadence threshold |
| `OMLX_KEEPWARM_LARGE_CACHE_INTERVAL_SECONDS` | `2` | Long-cache cadence |
| `OMLX_KEEPWARM_SLOW_THRESHOLD_SECONDS` | `1` | Slow-submission threshold |
| `OMLX_KEEPWARM_SLOW_BACKOFF_SECONDS` | `60` | Backoff after failure/slow work |

## Release gate

Compare cached-turn TTFT after 0, 5, 15, and 60 seconds of idle with the toggle
off and on. Test first, second, and later turns plus independent sessions.
Also gate B1/B2/B4/B6 throughput, concurrent cache hits, cancellation,
cross-session restores, SSD writes, explicit cache clear, repeated load/unload,
and process footprint over repeated pulses. Cold-prefill rate, generated tokens,
tool calls, cache-hit lengths, and MTP acceptance must remain equivalent:
keepwarm changes readiness only, never model math.

## Physical qualification reference

The final implementation was qualified on an M3 Ultra with Qwen3.8 Flash Next
oQ4e MTP and a 220K deterministic agent/tool conversation. The executable
resident cache used by that test is separate Fusion work and is not included in
this PR; the OFF/ON comparison changed only the persisted keepwarm switch.

- OFF model TTFT after 5/15/60-second gaps: 1.84/1.76/1.76 seconds.
- Repeated ON model-TTFT median: about 1.45 seconds; the 60-second result was
  0.85 seconds model TTFT and 1.83 seconds visible TTFT.
- Every measured turn emitted the exact expected structured tool call.
- Live pulse submissions were about 0.3-1.5 ms after a 3-4 ms first prepare.
- Seven memory samples over 60 seconds had exactly flat RSS and cache bytes.
- The full 1,187-file cache manifest hash was unchanged across the pulse soak.
- Same-binary B1 decode median was 55.56 tok/s OFF and 58.79 tok/s ON; cold
  prefill was effectively unchanged at 887.01 versus 885.31 tok/s.
- B1/B2/B4/B6 completed without request errors, and targeted cancellation
  stopped one stream while all three survivors completed.

These figures characterize one hardware/model stack, not a universal speed
promise. They demonstrate that the asynchronous pulse can improve cached-turn
readiness without changing inference outputs, cache persistence, active-work
throughput, or memory growth.
