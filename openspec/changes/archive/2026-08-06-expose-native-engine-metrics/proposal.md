# Proposal: expose-native-engine-metrics

## Summary

Mount vLLM's own Prometheus stat logger as a `GET /metrics` endpoint on the
InferenceX FastAPI app, so vLLM's execution-internal metrics (KV cache
utilization, running/waiting queue depth, per-phase timing) become scrapeable
without InferenceX reimplementing them.

## Why

`GET /v1/metrics` (`observability/` subsystem, added by
`add-observability-pipeline`) only sees the HTTP boundary — request latency,
status code, TTFT, tokens/sec derived from the SSE stream. It has no
visibility into vLLM's own scheduler and KV cache state, because that state
never crosses `BaseEngine`. vLLM's `AsyncLLM` already constructs a
production-grade `PrometheusStatLogger` internally (`vllm.v1.metrics.loggers`)
unless explicitly disabled. Exposing it is a mount, not a reimplementation —
by far the highest-leverage way to get engine-internal observability, and the
next item on the Phase B roadmap (`docs/PHASE-A-ARCHITECTURE.md` §10, "B2").

## What Changes

- Mount `GET /metrics` (Prometheus text exposition format) on the FastAPI app
  in `src/inference_x/api/main.py`, backed by vLLM's own `PrometheusStatLogger`
  registry.
- Exclude `/metrics` from `ObservabilityMiddleware`'s recording path, so
  scrape traffic does not enter `InMemoryStorage` or skew `GET /v1/metrics`
  aggregates (`avg_latency_ms`, `total_requests`, etc.).
- Declare `prometheus-client` as a direct `pyproject.toml` dependency (today
  present only transitively via `vllm`). This is required, not optional:
  `api/main.py` calls `prometheus_client.make_asgi_app(...)` directly under
  either branch of `design.md` Decision 1 — InferenceX's own code imports the
  `prometheus_client` public API, it does not merely benefit from vLLM's
  transitive dependency graph, and relying on the latter would silently break
  if vLLM ever re-pins or drops `prometheus-client`.
- Log a one-line startup `WARNING` when `pool_size > 1`, naming the
  metric-label collision risk of multiple `AsyncLLM` instances' default
  `PrometheusStatLogger`s sharing one process-wide registry — recorded, not
  guarded against (no new construction-time validation).
- No change to `BaseEngine`, `ChatService`, `AdmissionController`, any route
  handler, or the `GET /v1/metrics` schema/computation.

## In scope

- Mounting the `/metrics` endpoint.
- Excluding `/metrics` from HTTP-boundary metrics recording.
- Resolving, at implementation time, which Prometheus registry vLLM's default
  stat logger populates in the pinned vLLM version, and mounting against that
  registry (see `design.md` Decision 1 — this is a verification task, not a
  design choice made here).
- The `pool_size > 1` startup warning.

## Out of scope

- Any change to `GET /v1/metrics`'s response schema or computation.
- A unified view combining the two metric systems.
- OpenTelemetry tracing or distributed spans.
- Per-request decomposed queue/prefill/decode timing in the response body
  (Phase B3).
- Admission rescope, batch queueing, or multi-model process split (B4–B6).
- `engines/registry.py` — unrelated to this change.

## Capabilities

### New Capabilities

- `platform`: native engine metrics exposition (`GET /metrics`).

### Modified Capabilities

- None. `GET /v1/metrics` is unaffected; this is a strictly additive endpoint.

## Impact

- `src/inference_x/api/main.py` — mount the `/metrics` route.
- `src/inference_x/observability/middleware.py` — exclude `/metrics` from
  `dispatch()`'s recording path.
- `pyproject.toml` — add `prometheus-client` as a direct dependency (required
  because `api/main.py` imports its public API directly — see `design.md`
  Decision 1).
- `src/inference_x/engines/vllm_engine.py` — one-line startup `WARNING` for
  `pool_size > 1`. Branch A was confirmed (see `design.md` Decision 1); no
  registry-accessor import was needed.
- No impact to `src/inference_x/services/`, `src/inference_x/routing/`, or
  any existing test in `tests/` beyond new additions.
