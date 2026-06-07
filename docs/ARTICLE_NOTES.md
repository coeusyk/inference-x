# Article Notes

This file is maintained by the agent and by hand throughout the project.
Capture reasoning, tradeoffs, benchmarks, and implementation details here so the project can be turned into a technical article without reconstructing history.

## How to use

- Add a note after every meaningful milestone, architectural decision, benchmark, or noteworthy WSL2/vLLM finding.
- Keep notes short and factual. Use bullets, not polished prose.
- Capture exact commands, config values, error messages, and numbers when useful.
- Do not write final article language here — just raw observations.

## Note format

### [Date] — [Milestone or topic]
- **Change**: what was done
- **Why**: motivation
- **Tradeoff**: what was accepted or rejected
- **Validation**: how it was tested
- **Useful quote / command / number**: anything worth citing in the article
- **Screenshot / artifact**: optional

---

<!-- Add notes below this line -->

### 2026-06-07 — Phase 1 core inference slice complete (add-core-vllm-engine)

- **Change**: First runnable vLLM-backed inference path — OpenAI-compatible `POST /v1/chat/completions` and `GET /health` through layered modules (schemas → engine → service → routes).
- **Why**: Prove the architecture before routing, observability, or UI; establish a stable public contract for Phase 2+.
- **Tradeoff**:
  - Engine health is a boolean set at init success (`is_healthy()`); no live GPU ping on every `/health` poll — avoids latency and GPU churn.
  - Engine loads lazily on first request via FastAPI `Depends` — server starts fast, first `/health` or chat call pays model-load cost (~2–4 min cold, faster with cached weights).
  - `stream=true` rejected with HTTP 400 — Phase 1 is non-streaming only; silent ignore would break client expectations.
  - vllm stays in main `[project.dependencies]`; dev deps in `[dependency-groups] dev`.
  - Must run server with `uv run` or `./scripts/dev.sh serve` — bare pyenv `uvicorn`/`python` bypasses `.venv`.
  - Only one vLLM server per GPU — second instance fails KV cache alloc (`Available KV cache memory: -0.04 GiB`).
  - Legacy `core/engine.py` and `core/schemas.py` stubs remain; cleanup deferred.
- **Validation**:
  - `uv run python -m pytest tests/unit/ -v` → 32/32 passed (stub engine + dependency overrides; no GPU in unit tests).
  - Live smoke on WSL2 + CUDA: `qwen2.5-0.5b`, first health ~90s, chat reply `'Hello.'`, 38 tokens.
  - Commit: `fc1ee1d` on `develop`.
- **Useful quote / command / number**:
  - Start: `INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b ./scripts/dev.sh serve`
  - Smoke: `uv run python scripts/smoke_test.py`
  - Sync: `uv sync`
  - vLLM 0.22.1 on Python 3.13; WSL forces `spawn` multiprocessing and `pin_memory=False`.
  - Health codes: 200 healthy, 503 engine flag false, 500 engine init/inference failure.
  - Error shape: `{"error": {"message": "...", "type": "internal_error"|"invalid_request_error"}}`; validation errors use FastAPI 422 `detail` array.

---

### 2026-06-07 — Phase 2 model registry and routing layer (add-model-registry-routing)

- **Change**: Added `ModelRegistry`, `BaseRouter`, `TaskRouter`, `ExplicitModelPolicy`, `DefaultModelPolicy`, `GET /v1/models`. Wired routing into `ChatService`. All config-driven via `config/routing.yaml` + `config/models.yaml`.
- **Why**: Phase 2 contract — platform should select models by policy, not hardcode one engine reference in the service layer.
- **Tradeoff**:
  - Still single-engine (one GPU). Router resolves a model name; if it doesn't match the loaded engine, HTTP 400 is returned with a restart instruction. Multi-engine fanout deferred to Phase 3+.
  - `DefaultModelPolicy` validates the default at construction time, not per-request — misconfiguration fails at startup, not mid-request.
  - `ExplicitModelPolicy` returns `None` for unregistered models (falls through to default) rather than rejecting — clients can pass any `model` field and still get a valid response via fallback.
  - `ModelRegistry.from_config()` owns YAML parsing and Pydantic validation; `AppSettings.get_model_config()` retained for backward compat only.
  - `GET /v1/models` included (was optional in proposal) — removes need for out-of-band config knowledge; one thin handler.
- **Validation**:
  - `uv run pytest tests/unit/ -v` → 59/59 passed. 27 new tests: `test_model_registry.py` (11 tests), `test_router.py` (9 tests), extended `test_routes.py` (+4 for `/v1/models`) and `test_chat_service.py` (+2 routing tests).
  - `POST /v1/chat/completions` contract: unchanged. `GET /health` contract: unchanged.
- **Useful quote / command / number**:
  - Route selection order: `ExplicitModelPolicy` → `DefaultModelPolicy` (two policies, chain-of-responsibility).
  - Config: `config/routing.yaml` `default_model` overridden by `INFERENCE_X_DEFAULT_MODEL` env var.
  - `GET /v1/models` returns `{"object": "list", "data": [{"id": "...", "object": "model", "owned_by": "inferencex"}]}`.
  - Routing adds zero HTTP round-trips — selection is in-process, synchronous, before engine dispatch.

---

### 2026-06-07 — Phase 2 exit criteria verification (add-model-registry-routing)

- **Change**: Added FastAPI lifespan eager init (`initialize_app()` in `deps.py`, `lifespan` in `main.py`). Added `tests/conftest.py` to skip init in unit tests. Added `tests/unit/test_startup.py` (3 tests) and `/v1/models` config reflection test.
- **Why**: Phase 2 exit review — config/model-load failures must surface at startup, not on first request.
- **Tradeoff**:
  - Eager init loads vLLM at process start (~90s–4min cold) before accepting traffic — ops trade latency for predictability.
  - Unit tests patch `deps.initialize_app` to noop via autouse conftest — production path unchanged.
  - Unregistered `model` in chat request falls back to default (HTTP 200 if default loaded), not HTTP 400 — intentional per DEC-011; documented fallback, not error.
  - Registered but not-loaded model (e.g. request `tinyllama-chat` when `qwen2.5-0.5b` loaded) → HTTP 400 with restart instruction.
  - Missing `model` field → FastAPI/Pydantic 422 (field required).
  - Invalid/missing default model → `ValueError` at router construction → startup failure (now caught by lifespan).
  - `config/routing.yaml` is declarative documentation; runtime default comes from `INFERENCE_X_DEFAULT_MODEL` env var (fallback `qwen2.5-0.5b`), not YAML parse at startup.
- **Validation**:
  - `uv run pytest tests/unit/ -v` → 63/63 passed (59 original + 3 startup + 1 models-yaml reflection).
  - Runtime routing check: `unknown-model` → routes to `qwen2.5-0.5b`; invalid default → `ValueError` at construction.
  - `GET /v1/models` returns 5 models matching `config/models.yaml` names.
  - Phase 1 contracts unchanged: health 200/503/500, chat completions schema and error shapes preserved.
- **Useful quote / command / number**:
  - Startup log sequence: registry names → router ready → engine healthy.
  - Startup failure: CRITICAL log + process exit (lifespan re-raises).
  - Serve: `INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b ./scripts/dev.sh serve` — model loads before first request.

---

### 2026-06-07 — Phase 3 observability pipeline (add-observability-pipeline)

- **Change**: Added `observability/` package: `storage.py` (InMemoryStorage), `recorder.py` (MetricsRecorder), `exporters.py` (NullExporter + JsonLineExporter), `middleware.py` (ObservabilityMiddleware). Added `services/metrics_service.py`. Wired middleware into `main.py` via `app.add_middleware`. Recorder singleton in `deps.py`.
- **Why**: A serving platform without observability cannot be debugged or compared. Phase 3 exit criteria require latency, token, and error recording without touching the API contract.
- **Tradeoff**:
  - Middleware is the only injection point — route handlers and services have zero observability code.
  - In-memory ring buffer (1000 records, `collections.deque`) — no I/O in the request path; data lost on restart. Phase 5 can swap in a durable adapter.
  - Response body buffered only for `POST /v1/chat/completions` status 200 — needed to extract token counts from the JSON body. All other paths are untouched. One extra allocation per chat request.
  - `BaseHTTPMiddleware` body-buffer swap technique: consume `body_iterator`, create new `Response(content=bytes, ...)` — Starlette re-calculates `content-length`. Confirmed identical payloads via `test_response_body_unchanged_after_middleware`.
  - File exporter (`JsonLineExporter`) is opt-in via `INFERENCE_X_METRICS_FILE` env var — off by default. Opens in append mode per write; no persistent file handle.
  - Middleware recorder is a true process singleton — wired at module load before FastAPI DI is available. Tests isolate via `storage.clear()` in fixture (not recorder injection).
- **Validation**:
  - `uv run pytest tests/unit/ -v` → 93/93 passed (63 prior + 30 new observability tests).
  - Middleware integration: health, chat, models endpoints all recorded; response body byte-for-byte identical after buffering; error flag set on 500.
  - Recorder error safety: storage and exporter exceptions both swallowed (mock tests confirm).
  - Phase 1 + 2 contracts: all prior 63 tests pass unchanged.
- **Useful quote / command / number**:
  - Enable file export: `INFERENCE_X_METRICS_FILE=/tmp/inferencex-metrics.jsonl ./scripts/dev.sh serve`
  - Record shape: `{"request_id": "...", "path": "/v1/chat/completions", "method": "POST", "status_code": 200, "latency_ms": 142.5, "model": "qwen2.5-0.5b", "prompt_tokens": 12, "completion_tokens": 38, "total_tokens": 50, "error": false}`
  - `MetricsService.summary()` returns: total_requests, error_count, avg_latency_ms, p95_latency_ms.
