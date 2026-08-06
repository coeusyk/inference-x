# Tasks: expose-native-engine-metrics

## 1. Verify registry identity (blocks all other tasks) — COMPLETE
- [x] 1.1 In an environment with `vllm` installed (project `.venv`, vLLM
      0.22.1, via `uv run`), inspected
      `vllm.v1.metrics.loggers.PrometheusStatLogger.__init__` and
      `vllm.entrypoints.serve.instrumentator.metrics.get_prometheus_registry`,
      and constructed a real `PrometheusStatLogger` against a real
      `VllmConfig` to confirm empirically which registry the default logger
      populates. **Result: Branch A** — registers directly onto
      `prometheus_client.REGISTRY` (confirmed by observing 125 new `vllm:`
      collector names appear in `REGISTRY._names_to_collectors` after
      construction, not merely inferred from source).
- [x] 1.2 Updated `design.md` Decision 1 with the resolved branch (Branch A)
      before starting task 2.

## 2. Mount `GET /metrics`
- [x] 2.1 Add the route in `src/inference_x/api/main.py`, using
      `prometheus_client.make_asgi_app()` with no `registry` argument
      (Branch A, resolved in task 1 — see `design.md` Decision 1).
- [x] 2.2 Confirm the route is reachable before `lifespan`'s engine
      construction completes returns a well-formed (if empty) exposition
      body, not a 500 — Prometheus scrapers should get a valid response even
      before the first engine is constructed.

## 3. Exclude `/metrics` from HTTP-boundary metrics recording
- [x] 3.1 Add a path check at the top of
      `ObservabilityMiddleware.dispatch()` in
      `src/inference_x/observability/middleware.py` that skips timing/
      recording entirely for `/metrics` (pass the request straight to
      `call_next` and return).
- [x] 3.2 Confirm `/health` needs no equivalent change (already outside the
      middleware's extraction scope — verify, don't assume).

## 4. Dependency declaration
- [x] 4.1 Add `prometheus-client` as a direct dependency in `pyproject.toml`
      (currently transitive via `vllm` only). Required regardless of which
      Decision 1 branch resolves true — `api/main.py` calls
      `prometheus_client.make_asgi_app(...)` directly in both cases (see
      `design.md` Decision 1, "Dependency consequence").

## 5. `pool_size > 1` startup warning
- [x] 5.1 Log a one-line `WARNING` in `VLLMEngine.__init__` (near existing
      engine-construction logging) when `pool_size > 1`, naming the
      metric-label collision risk from multiple `AsyncLLM` instances'
      default `PrometheusStatLogger`s. No new validation or rejection logic.

## 6. Tests
- [x] 6.1 `GET /metrics` returns a 200 with `Content-Type` matching
      Prometheus text exposition format and includes at least one `vllm:`
      prefixed series.
- [x] 6.2 A `GET /metrics` scrape does not change
      `MetricsService.summary().total_requests`, `avg_latency_ms`, or
      `p95_latency_ms` from their pre-scrape values.
- [x] 6.3 Existing `GET /v1/metrics` tests remain green, unmodified.
- [x] 6.4 If `pool_size > 1` is exercised in a test, assert the new
      `WARNING` is logged (no assertion on Prometheus registry state itself,
      which is vLLM-owned).

## 7. Docs
- [x] 7.1 Add a `CHANGELOG.md` `[Unreleased]` entry describing the new
      `GET /metrics` endpoint.
- [x] 7.2 Mark B2 complete in `docs/PHASE-A-ARCHITECTURE.md` §10, following
      the pattern of the existing "B1 status: complete" writeup.

## 8. Validation
- [x] 8.1 `ruff check .` green.
- [x] 8.2 `mypy src/` green, with no new baseline entries beyond what task
      1's resolved branch requires.
- [x] 8.3 `pytest tests/unit` green.
- [x] 8.4 `openspec validate expose-native-engine-metrics --strict` passes.
