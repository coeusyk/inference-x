# Design: expose-native-engine-metrics

## Context

`docs/PHASE-A-ARCHITECTURE.md` §10 lists "expose vLLM's native Prometheus
stats" as Phase B2, immediately after the completed B1 (AsyncLLM migration).
The Phase-A audit's OWN-B5 sibling note flagged a future overlap between the
existing HTTP-boundary observability middleware
(`src/inference_x/observability/middleware.py`) and vLLM's native stat
loggers once B2 lands.

Concretely verified during this proposal's research (read-only, no `vllm`
package installed locally to execute against):

- `ObservabilityMiddleware.dispatch()` wraps every request — a future
  `/metrics` scrape would be recorded into `InMemoryStorage` exactly like a
  chat completion, skewing `GET /v1/metrics`'s `avg_latency_ms` and
  `total_requests` with scrape noise. vLLM's own reference
  `prometheus-fastapi-instrumentator` mounting code excludes `/metrics`,
  `/health`, `/load`, `/ping`, `/version`, `/server_info` from instrumentation
  for exactly this reason. The overlap is **scrape traffic polluting the
  existing HTTP-boundary ring buffer**, not the two metric namespaces
  colliding (`vllm:*` vs. `avg_latency_ms` etc. are disjoint by name).
- `prometheus-client` 0.25.0 is already resolved in `uv.lock` (transitive via
  `vllm`); `prometheus-fastapi-instrumentator` is also present transitively
  but unused by any InferenceX code today.
- vLLM 0.22.1 (the version `uv.lock` currently resolves for `vllm>=0.6.0`)
  constructs a default `vllm.v1.metrics.loggers.PrometheusStatLogger` inside
  `AsyncLLM` via `StatLoggerManager`, unless `log_stats=False`
  (`disable_log_stats` on `AsyncEngineArgs`) and no custom logger is supplied.
  `VLLMEngine.__init__` (`vllm_engine.py:414-415`) already passes `**kwargs`
  into `AsyncEngineArgs(**kwargs)`, so `disable_log_stats` is already a live
  knob with zero code change required to toggle it.
- `PrometheusStatLogger.__init__` calls `unregister_vllm_metrics()` and
  registers `vllm:*` series carrying `engine` and `model_name` labels. A
  second `AsyncLLM` instance (`pool_size > 1`) constructing its own default
  `PrometheusStatLogger` is therefore a genuine, if narrow, label-collision
  risk against B1 Decision 3 ("`pool_size > 1` is not guaranteed and not
  forbidden").

## Goals / Non-Goals

**Goals**

- Mount vLLM's native Prometheus exposition with zero new abstraction layer
  — a passthrough, not a reimplementation.
- Keep `GET /v1/metrics` uncontaminated by `/metrics` scrape traffic.
- Keep `BaseEngine`'s declared contract, `ChatService`, and
  `AdmissionController` completely untouched.

**Non-Goals**

- Unifying the two metrics systems into one view.
- Solving the `pool_size > 1` label-collision risk — this change records it
  (a startup warning), it does not guard against it. Multi-engine serving
  redesign is B6's scope.
- OpenTelemetry tracing or distributed spans.
- Per-request decomposed timing in the response body (B3).

## Decisions

### Decision 1 — Registry identity: verified as Branch A (default global registry)

**Status: resolved (task 1 of `tasks.md` complete).** Verified empirically
against the installed vLLM 0.22.1 (`.venv`, via `uv run`), not inferred:

1. `vllm.v1.metrics.loggers.Gauge`/`Counter`/`Histogram` are, by object
   identity, `prometheus_client.metrics.Gauge`/`Counter`/`Histogram` — the
   exact same classes, not vLLM subclasses or wrappers.
2. `prometheus_client.Gauge.__init__`'s `registry` parameter defaults to the
   module-level `prometheus_client.registry.REGISTRY` singleton.
3. `PrometheusStatLogger.__init__`'s source contains zero occurrences of
   `registry=` — every metric it constructs falls through to that library
   default, i.e. the global `REGISTRY`.
4. `unregister_vllm_metrics()` (called at the top of every
   `PrometheusStatLogger.__init__`, to make re-construction idempotent)
   explicitly operates on `REGISTRY` — vLLM's own code treats the global
   registry as canonical for its default stat logger.
5. **Direct execution, not just source reading:** constructed a real
   `VllmConfig` (`facebook/opt-125m`) and a real `PrometheusStatLogger`
   against it, and observed `prometheus_client.REGISTRY._names_to_collectors`
   gain 125 new `vllm:`-prefixed collector names as a direct result — e.g.
   `vllm:e2e_request_latency_seconds`, `vllm:num_requests_running`.

**Conclusion: Branch A.** `src/inference_x/api/main.py` mounts
`prometheus_client.make_asgi_app()` with no `registry` argument. Zero
vLLM-specific imports land in `api/`; the Engine Boundary (DEC-047) stays
fully clean — `api/` never imports from `vllm` directly for this endpoint.

**Caveat found during verification (not a Branch B, a separate axis):**
`vllm.entrypoints.serve.instrumentator.metrics.get_prometheus_registry()` —
part of vLLM's own `serve` CLI mounting code, not something InferenceX
calls — returns the same global `REGISTRY` *unless* the environment variable
`PROMETHEUS_MULTIPROC_DIR` is set, in which case it returns a fresh
`CollectorRegistry()` wrapped in `prometheus_client.multiprocess
.MultiProcessCollector`. This is `prometheus_client`'s standard
multi-worker-process pattern (used by any Gunicorn/uvicorn multi-worker
deployment), not a vLLM-specific concern, and does not apply to InferenceX's
current single-process `uvicorn` deployment. It is recorded here as a known
limit of this decision's scope: if InferenceX is ever run under
`PROMETHEUS_MULTIPROC_DIR` (multi-worker mode), `/metrics` mounting must
switch to the multiprocess `CollectorRegistry` pattern — out of scope for
this change, tracked as a future note rather than solved now.

**Dependency consequence:** `api/main.py` calls
`prometheus_client.make_asgi_app()` directly — this is InferenceX's own
import of `prometheus_client`'s public API, not a byproduct of `vllm`
merely depending on `prometheus-client` transitively, so `prometheus-client`
must be declared as a direct `pyproject.toml` dependency. This conclusion
held under both branches while the fork was open and is unaffected by its
resolution.

**Resolution procedure (completed):** constructed a real `VllmConfig` and
`PrometheusStatLogger` via `uv run python` against the project's own `.venv`
(vLLM 0.22.1 installed there) and confirmed registration onto
`prometheus_client.REGISTRY` directly, per the evidence above. Task 2 may
now begin.

### Decision 2 — `/metrics` is excluded from `ObservabilityMiddleware`, not merely uncounted after the fact

The exclusion happens at `dispatch()` entry (a path check before any timing
or extraction work begins), matching vLLM's own
`prometheus-fastapi-instrumentator` exclusion pattern
(`excluded_handlers=["/metrics", "/health", ...]`). Only `/metrics` is added
to InferenceX's exclusion. `/health` is recorded by `ObservabilityMiddleware`
today — the `is_chat` gate only controls chat-request metadata extraction,
not the unconditional `recorder.record()` call at the end of `dispatch()` —
and changing that is out of scope for this change.

### Decision 3 — No new settings/config toggle

`disable_log_stats` already reaches `AsyncEngineArgs` via the existing
`**kwargs` passthrough in `VLLMEngine.__init__`. Since the knob already
exists one layer up with no repo code required to use it, this change adds no
new `INFERENCE_X_*` environment variable to control it — introducing one
would be speculative configuration for a value nothing today needs to flip
per-deployment.

### Decision 4 — `pool_size > 1` label collision is recorded, not guarded

Consistent with B1 Decision 3's stance ("`pool_size > 1` is not guaranteed
and not forbidden... no new construction-time validation is added for it in
either direction"), this change logs a one-line `WARNING` at startup when
`pool_size > 1`, naming the metric-label collision risk, and adds no
validation, rejection, or label-disambiguation logic. Redesigning
multi-engine serving is B6's scope.

## Compatibility Invariants

1. `GET /v1/metrics`'s response schema (`MetricsResponse`) and every field's
   meaning are unchanged — no `vllm:*` data feeds it, preserving the DEC-050
   no-discontinuous-metric boundary for the existing series.
2. The `vllm:*` Prometheus series' names, labels, and stability are vLLM's
   own contract, not one InferenceX makes or guarantees — this endpoint is
   additive and backend-defined, not a wrapped/re-typed InferenceX metric.
3. `BaseEngine`'s declared contract is unchanged — no new capability method,
   no new `getattr` discovery pattern added to admission or any caller.
4. `pool_size > 1` remains not-guaranteed-and-not-forbidden (B1 Decision 3);
   this change adds no new construction-time validation in either direction.
5. `/metrics` scrape requests never enter `InMemoryStorage` and never affect
   any `GET /v1/metrics` aggregate (`total_requests`, `avg_latency_ms`,
   `p95_latency_ms`, `avg_ttft_ms`, `avg_tokens_per_sec`).

## Risks / Trade-offs

- **Registry identity is unverified at proposal time.** Mitigated by Decision
  1's explicit two-branch task — implementation cannot proceed past task 1
  without resolving it, so no code is written against a guessed assumption.
- **`pool_size > 1` label collision is real but narrow** (only manifests with
  multi-instance serving, which is not yet a supported configuration per B1
  Decision 3). Recorded via startup warning rather than solved, consistent
  with this change's scope.
- **A second Prometheus-format endpoint alongside the existing JSON one** is
  two metrics surfaces to explain to operators. Accepted: they answer
  different questions (HTTP-boundary vs. engine-internal) and merging them
  would require reshaping vLLM's native series into InferenceX's schema,
  which is explicitly out of scope.

## Validation properties (required)

- `ruff check .`, `mypy src/`, `pytest tests/unit` all green.
- No new `mypy` baseline entry: Branch A resolved with no registry-accessor
  import, so no additional baseline entry was required.
- New regression test: a `GET /metrics` request does not increment
  `MetricsService.summary().total_requests`, and does not move
  `avg_latency_ms`/`p95_latency_ms` from their pre-scrape values.
- New test: `GET /metrics` returns `Content-Type` matching Prometheus text
  exposition format and includes at least one `vllm:` prefixed series name.
