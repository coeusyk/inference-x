# Proposal: expose-per-request-engine-timing

## Summary

Expose per-request engine-internal timing — queue wait, prefill duration,
decode duration, and total engine execution time — as an additive,
optional field on `ChatCompletionResponse` and the terminal
`ChatStreamChunk`, sourced directly from `vllm.v1.metrics.stats.RequestStateStats`
(the same object vLLM's own `AsyncLLM` already attaches to every finished
`RequestOutput`).

## Why

`GET /metrics` (B2, `expose-native-engine-metrics`) exposes vLLM's
*process-wide, scrape-time* engine state — aggregate gauges and histograms a
Prometheus scraper reads independently of any single request. It answers
"how is the engine doing right now," not "how was *my* request served."

`GET /v1/metrics` (`observability/`, `add-observability-pipeline`) answers a
different question again: HTTP-boundary latency, TTFT, and tokens/sec,
measured from outside the engine, aggregated across all requests.

Neither surface lets a client attribute a single response's latency to
queueing versus prefill versus decode. That decomposition is the signal a
client cannot compute itself — it doesn't know when the scheduler picked up
its request, or how the engine spent the time between admission and its
first token. `ChatStreamChunk`'s own docstring already reserves this:
"Per-request timings are Phase B3 and are not carried here"
(`schemas/chat.py`, `ChatStreamChunk`) — this proposal is that reservation
being filled.

This is the next logical step after B2: B2 proved vLLM's engine-internal
state is available and mountable with near-zero reimplementation; this
change proves the same is true one level down, per request, without
inventing a new metrics pipeline.

## What Changes

- Add `EngineTiming` to `schemas/chat.py`: `queue_time_ms`, `prefill_time_ms`,
  `decode_time_ms`, `inference_time_ms` — four `float` fields, all derived
  from vLLM's own per-request `RequestStateStats` monotonic timestamps
  (`queued_ts`, `scheduled_ts`, `first_token_ts`, `last_token_ts`; see
  `design.md` for the empirical verification of exactly what vLLM 0.22.1
  populates and why these four fields, not others).
- Add `timing: Optional[EngineTiming] = None` to `ChatCompletionResponse`
  and to `ChatStreamChunk`'s terminal event — mirroring exactly how `usage`
  was added under DEC-049/OS-2: additive, defaults to `None`, absent rather
  than estimated when the engine cannot supply it.
- Extend `derive_terminal_metadata()` (`engines/vllm_engine.py`) to also
  derive `EngineTiming` from the same finished `RequestOutput` it already
  reads `usage` and `finish_reason` from. No new call site, no second read
  of engine state — preserves the DEC-050 single-source guarantee that
  streaming and non-streaming report identical values for the same request.
- No change to `GET /metrics` or `GET /v1/metrics` — this is response-body
  data for a single request, not a new aggregate or scrape surface.

## In scope

- The four timing fields listed above, computed exclusively from
  engine-core-process monotonic timestamps (see `design.md` for the
  clock-domain verification that makes this safe).
- Threading `EngineTiming` through both `generate()` and `generate_stream()`
  via the existing single derivation call site.
- Graceful, typed absence (`timing: None`) when the engine does not supply
  `RequestStateStats` — never an estimate, never a zero standing in for
  "unknown."

## Out of scope

- A distinct "scheduler delay" separate from "queue wait." vLLM 0.22.1
  tracks exactly one queue-related interval per request
  (`queued_ts` → `scheduled_ts`); there is no empirical basis for a second,
  finer-grained breakdown, and none is invented here (see `design.md`
  verification log).
- `RequestStateStats.arrival_time` and `.first_token_latency`. Both are
  frontend wall-clock (`time.time()`) values; `first_token_latency` in
  particular duplicates the TTFT `GET /v1/metrics` already reports
  (DEC-049) rather than adding new signal. Excluded to keep engine timing
  and HTTP-boundary timing clearly distinct, per this change's design
  requirements.
- `RequestStateStats.is_corrupted` (NaN-logit detection) — orthogonal to
  timing; not a timing field, not added here.
- Any change to `GET /metrics`, `GET /v1/metrics`, admission control
  (B4), request queueing (B5), or multi-model process split (B6).
- Preemption-aware timing. vLLM's own scheduler comment
  (`stats.py`: `if req_stats.scheduled_ts == 0.0: # ignore preemptions`)
  means a preempted-then-rescheduled request's `queued_ts`→`scheduled_ts`
  interval reflects only the first scheduling attempt. This change surfaces
  vLLM's number as-is and does not attempt to reconstruct preemption
  history.
- Support for a non-vLLM `BaseEngine` implementation that cannot supply
  timing — the contract is `timing: Optional[EngineTiming]`, `None` is a
  valid and expected value for such a backend, same posture as `usage`.

## Capabilities

### New Capabilities

- `platform`: per-request engine timing attribution (`EngineTiming` on
  `ChatCompletionResponse` / `ChatStreamChunk`).

### Modified Capabilities

- None. `GET /v1/metrics`, `GET /metrics`, and the existing
  `ChatCompletionResponse`/`ChatStreamChunk` fields are unaffected; this is
  a strictly additive field.

## Impact

- `src/inference_x/schemas/chat.py` — add `EngineTiming`; add
  `timing: Optional[EngineTiming] = None` to `ChatCompletionResponse` and
  `ChatStreamChunk`.
- `src/inference_x/engines/vllm_engine.py` — extend
  `derive_terminal_metadata()` to also return `EngineTiming | None`,
  derived from `output.metrics` (`RequestStateStats`) on the same finished
  `RequestOutput` already in scope at that call site.
- `src/inference_x/services/chat_service.py` — thread the derived `timing`
  onto the wire in `stream_response()`: bundled onto the existing
  `include_usage`-gated terminal usage event alongside `usage`, not a new
  event or flag (see `design.md` Decision 6).
- No impact to `src/inference_x/api/`, `src/inference_x/routing/`,
  `src/inference_x/observability/`, or `GET /metrics`/`GET /v1/metrics`.
