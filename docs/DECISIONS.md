# InferenceX Decisions

Use this document to capture non-obvious design decisions as the project evolves.

## Decision template

### DEC-000
- Date:
- Status: proposed | accepted | superseded
- Context:
- Decision:
- Consequences:

## Initial decisions

### DEC-001
- Date: 2026-06-07
- Status: accepted
- Context: The project needs a clear first milestone and must remain incremental.
- Decision: Start with one vLLM-backed OpenAI-compatible chat completions API before adding routing, observability, or UI.
- Consequences: Early progress stays measurable and later phases can build on a stable contract.

### DEC-002
- Date: 2026-06-07
- Status: accepted
- Context: Development is happening on Windows, while vLLM requires Linux tooling.
- Decision: Standardize local development on WSL2 Ubuntu and treat it as the primary runtime environment.
- Consequences: Commands, setup steps, and scripts should target Linux inside WSL2 first.

### DEC-003
- Date: 2026-06-07
- Status: accepted
- Context: The repo must support agent-assisted development in Cursor.
- Decision: Use `.cursor/rules/` for persistent repository guidance and keep the rules split by concern.
- Consequences: Repo-level instructions remain concise and easier for the editor to apply consistently.

### DEC-004
- Date: 2026-06-07
- Status: accepted
- Context: The vLLM engine must be importable without a GPU for local dev and testing.
- Decision: Guard `from vllm import ...` inside a try/except in `vllm_engine.py`; raise a clear `RuntimeError` at instantiation time if vllm is absent.
- Consequences: Tests and schema validation work without vLLM. The engine fails fast with a clear message when vLLM is missing at runtime.

### DEC-005
- Date: 2026-06-07
- Status: superseded
- Context: `pyproject.toml` listed `requires-python = ">=3.13"` but WSL2 runtime is Python 3.12.
- Decision: Change to `>=3.11` and separate `vllm` into an optional extra (`[vllm]`) so dev installs don't require GPU wheels.
- Consequences: Dev setup is lighter; production installs must use `pip install inferencex[vllm]`.
- Superseded by: DEC-007 (2026-06-07) — optional vllm broke `uv sync` and Phase 1 smoke tests.

### DEC-007
- Date: 2026-06-07
- Status: accepted
- Context: Moving vllm to `[project.optional-dependencies]` caused `uv sync` to uninstall vllm; smoke test returned HTTP 500 "vllm is not installed".
- Decision: Keep vllm in main `[project.dependencies]` (Phase 1 is vLLM-backed). Use `[dependency-groups] dev` for test deps (uv includes dev group by default on sync). Keep `requires-python = ">=3.13"` to match the uv-managed `.venv`.
- Consequences: `uv sync` restores the full runtime stack including vllm. CI without GPU must use a separate strategy (mock engine / skip integration), not drop vllm from default deps.

### DEC-006
- Date: 2026-06-07
- Status: accepted
- Context: `core/engine.py` and `core/schemas.py` contained implementation code in the wrong layer.
- Decision: Migrate all engine logic to `engines/` and schema models to `schemas/` per the architecture spec. Legacy stub files left in place; remove in a cleanup change.
- Consequences: Dependency flow API → Services → Interfaces → Implementations is now enforced by module layout.

### DEC-008
- Date: 2026-06-07
- Status: accepted
- Context: Phase 1 exit review found `/health` returned HTTP 200 with `status: degraded` when the engine health flag was false.
- Decision: Return HTTP 503 when `is_healthy()` is false; return HTTP 500 when engine initialization raises `RuntimeError`; return HTTP 200 only when healthy.
- Consequences: Load balancers and smoke tests can distinguish unavailable from ready. Body still includes `status` and `engine` fields.

### DEC-009
- Date: 2026-06-07
- Status: accepted
- Context: Phase 1 needs a health signal without running inference on every poll.
- Decision: Set `_healthy = True` only after successful vLLM `LLM()` init; `is_healthy()` returns that flag with no live generation.
- Consequences: Health reflects init-time readiness, not runtime degradation after load. Acceptable for Phase 1; runtime probes can be added in observability phase.

### DEC-010
- Date: 2026-06-07
- Status: superseded
- Context: Request schema accepts `stream: bool` but Phase 1 has no SSE implementation.
- Decision: Reject `stream=true` in `ChatService.complete()` with `ValueError` → HTTP 400 and structured error body.
- Consequences: Clients get an explicit error instead of a misleading non-streaming 200 response.
- Superseded by: DEC-023 (2026-06-07) — Phase 5 enables OpenAI-compatible SSE streaming.

---

## Phase 2 decisions

### DEC-011
- Date: 2026-06-07
- Status: accepted
- Context: Phase 2 adds a registry and router but the hardware is still one GPU / one loaded engine.
- Decision: `ChatService` resolves a model name via `TaskRouter` then validates it against `loaded_model`. If the routed model differs from the loaded engine, a `ValueError` → HTTP 400 is returned with a clear restart instruction.
- Consequences: Multi-model serving is explicitly deferred. Clients are told exactly what model is loaded and how to switch. No silent wrong-model responses.

### DEC-012
- Date: 2026-06-07
- Status: accepted
- Context: Choosing where to validate the `default_model` setting (startup vs. first request).
- Decision: Validate at `TaskRouter` and `DefaultModelPolicy` construction — not at `ChatService.complete()`. If the default model is absent from the registry, the app fails to start with a clear `ValueError`.
- Consequences: Misconfiguration surfaces immediately on startup, not mid-request. Operationally safer; no request can succeed against a missing default model.

### DEC-013
- Date: 2026-06-07
- Status: accepted
- Context: `GET /v1/models` was optional in the proposal.
- Decision: Include it. It costs one thin route handler and two Pydantic models. It lets clients enumerate registered models without reading `models.yaml` directly and removes the need for out-of-band documentation.
- Consequences: Adds one endpoint to the public contract. Follows OpenAI API shape (`object: "list"`, `data: [...]`).

### DEC-014
- Date: 2026-06-07
- Status: accepted
- Context: `ModelRegistry` could be derived from `AppSettings.get_model_config()` or owned separately.
- Decision: Give `ModelRegistry` its own `from_config(config_dir)` factory that reads `models.yaml` and validates all entries via Pydantic. `AppSettings` retains `get_model_config` for backward compat but is no longer the canonical registry.
- Consequences: Registry is independently testable with a `tmp_path` fixture. Settings retains minimal config surface for env vars.

### DEC-015
- Date: 2026-06-07
- Status: accepted
- Context: Phase 2 exit review found registry, router, and engine were built lazily on first request via `lru_cache` in `deps.py`. Config errors and model-load failures only surfaced when a client hit an endpoint.
- Decision: Add a FastAPI `lifespan` handler in `main.py` that calls `deps.initialize_app()` at startup. `initialize_app()` eagerly builds registry, router, and engine; logs each step; re-raises on failure so uvicorn never enters a ready state. Unit tests skip eager init via `tests/conftest.py` autouse patch on `deps.initialize_app`.
- Consequences: Misconfiguration and engine init failures fail fast at process start with CRITICAL logs. First request no longer pays cold-start init cost. Test suite remains GPU-less via conftest noop patch.

---

## Phase 3 decisions

### DEC-016
- Date: 2026-06-07
- Status: accepted
- Context: Observability must not touch route handlers or services, and must not block the request path.
- Decision: Use Starlette `BaseHTTPMiddleware` as the sole injection point. The middleware times the full request, extracts model and token info from bodies for `/v1/chat/completions` only, then calls `MetricsRecorder.record()` synchronously after response dispatch. All errors in extraction are caught and discarded.
- Consequences: Route handlers and services are completely unaware of observability. Any future observability extension only touches `middleware.py` and downstream storage/export. The middleware adds negligible overhead for non-chat routes (no body parsing).

### DEC-017
- Date: 2026-06-07
- Status: accepted
- Context: Choosing a storage backend for metrics that is non-blocking and has no external dependency.
- Decision: `InMemoryStorage` — a thread-safe `collections.deque` capped at 1000 records. No file I/O, no network, no external DB. The optional `JsonLineExporter` (NDJSON append to file) is the only out-of-process path, enabled via `INFERENCE_X_METRICS_FILE` env var.
- Consequences: Storage is trivially testable. Data does not survive process restarts. A Phase 5 hardening change can swap in a SQLite or file-backed adapter by implementing the same `append/all/recent` interface without changing recorder or middleware.

### DEC-018
- Date: 2026-06-07
- Status: accepted
- Context: `BaseHTTPMiddleware` buffers the response body when the middleware needs to read it (for token extraction). This changes how the response is streamed.
- Decision: Only buffer the response body for `POST /v1/chat/completions` with status 200. For all other paths (health, models, errors) the response body_iterator is never touched. After buffering, a new `Response` is constructed with the same status, media type, and headers (excluding `content-length`, which Starlette recalculates). This preserves the exact payload the client receives.
- Consequences: One additional in-memory allocation for chat completion responses (small for non-streaming). Phase 1 and Phase 2 API contracts are unaffected — confirmed by 93/93 passing tests including `test_response_body_unchanged_after_middleware`.

### DEC-019
- Date: 2026-06-07
- Status: accepted
- Context: The middleware singleton recorder must be wired at app creation (`add_middleware`), before FastAPI's dependency injection is available.
- Decision: `deps._build_recorder()` is an `lru_cache` function that builds the `InMemoryStorage` + configured exporter + `MetricsRecorder`. `deps.get_recorder()` is a plain function (not FastAPI Depends) that returns this singleton. `main.py` calls `deps.get_recorder()` at module load when registering the middleware. Integration tests clear `recorder.storage` in a per-test fixture rather than injecting a different recorder.
- Consequences: The recorder is a true process singleton shared between middleware and any future `MetricsService` Depends usage. Test isolation is via `storage.clear()` before each test — simple and reliable.

### DEC-021
- Date: 2026-06-07
- Status: accepted
- Context: Phase 4 compare flow worked (sequential / dual-server), but comparing two models on a single server always failed because the server loaded exactly one engine per process (DEC-011).
- Decision: Replace the single-engine architecture with `EnginePool` (`engines/pool.py`) — a dict of `{model_name: BaseEngine}` instances. `ChatService` dispatches by routed model name instead of validating a single loaded model. `INFERENCE_X_LOADED_MODELS` (comma-separated env var) controls which models are loaded at startup; default is `INFERENCE_X_DEFAULT_MODEL` for backward compatibility. The error message when a model is not in the pool preserves the "loaded model is '…'" pattern so the playground preflight check still works.
- Consequences: A single server can serve multiple models simultaneously (subject to GPU VRAM). Playground `--compare` on a single URL works when both models are listed in `INFERENCE_X_LOADED_MODELS`. Startup is slower when multiple large models are loaded. DEC-011 is superseded for code architecture; the operator constraint (one model fits in memory) remains a deployment concern.

---

## Phase 4 decisions

### DEC-020
- Date: 2026-06-07
- Status: accepted
- Context: Phase 4 requires a playground and comparison flow. Options: (a) React/Next.js SPA, (b) single HTML+JS file, (c) Python CLI script.
- Decision: Python CLI script (`playground/client.py`) using stdlib only (`argparse`, `urllib.request`, `json`). No npm, no node_modules, no frontend build pipeline. The script talks directly to the existing API over HTTP.
- Consequences: Runs immediately in WSL2 with `python3` — zero setup beyond a live server. Output is terminal text designed for paste/screenshot. Side-by-side compare uses columnar layout. Testable with `unittest.mock`. A future HTML client can be added without touching the Python script.

### DEC-022
- Date: 2026-06-07
- Status: accepted
- Context: The playground CLI output was plain text (`print()` + string formatting). Good enough for smoke tests but not demos or article screenshots. The Phase 4 compare output specifically relied on a fixed-width `SEPARATOR` string with manual column math.
- Decision: Add `rich>=13.0` as a project dependency. Introduce `playground/console.py` with per-call Console factories (`stdout_console()`, `stderr_console()`) and top-level helpers (`print_header`, `print_health`, `print_models_table`, `print_error`). Add `print_single` and `print_compare` to `client.py` and wire them into `run_single`, `run_compare`, and `run_compare_sequential`. All public function signatures (`format_single`, `format_compare`, `extract_text`, `format_usage`, `wrap_column`, `Prompt`, etc.) and the `SEPARATOR` constant are kept unchanged so the test suite requires no edits. Console instances are created fresh at call time (`Console(file=sys.stdout, ...)`) so tests that `mock.patch("sys.stdout")` capture rich output correctly without any special test fixtures.
- Consequences: Model names are bold and prominent; latency/tokens are muted; errors are red-bordered panels to stderr; compare mode shows two Panels side-by-side via `Columns` + a stats `Table`; sequential compare shows a spinner during the server-reload wait. 158 unit tests pass unchanged.

---

## Phase 5 decisions

### DEC-023
- Date: 2026-06-07
- Status: accepted
- Context: DEC-010 intentionally rejected `stream=true` while Phase 1 had no SSE implementation. Phase 5 now needs streaming for OpenAI-compatible clients and a live playground demo.
- Decision: Enable `stream=true` on `POST /v1/chat/completions`. Engines expose `generate_stream()` as raw text chunks, `ChatService.stream_response()` formats OpenAI-compatible SSE `data: {...}` events, and the route returns `text/event-stream`. Observability skips response-body buffering for streaming responses.
- Consequences: Non-streaming JSON responses remain unchanged. Streaming requests are recorded by middleware without backend token extraction, so streamed token counts are client-side estimates until a later usage event is added.

### DEC-024
- Date: 2026-06-07
- Status: accepted
- Context: The rich CLI improved batch output, but Phase 5 needs an interactive demo where token streaming, latency, health, and model comparison are visible while the request is running.
- Decision: Add a Textual TUI in `playground/app.py` with `httpx` async streaming. Keep `playground/client.py` intact for batch compare use. Use Textual CSS for a dark terminal layout and expose pure helper functions for unit tests instead of testing the full app event loop.
- Consequences: The playground now has a demo-ready live interface without adding backend UI endpoints or npm tooling. `textual` and `httpx` are runtime dependencies, and token usage shown after streams is estimated client-side.

### DEC-025
- Date: 2026-06-07
- Status: accepted
- Context: A post–Phase 5 security audit identified missing input bounds, verbose error leakage, permissive default bind address, unstructured logging, and playground SSRF risk to internal networks.
- Decision: Hardening pass — (1) `ChatCompletionRequest` schema caps: `content` max 32k chars, ≤50 messages, `max_tokens` ≤4096 with per-model cap `min(4096, max_model_len)` in `ChatService`; (2) `ValueError`/`RuntimeError` HTTP handlers return generic `"Request could not be processed."` while logging full detail at ERROR; (3) default bind `127.0.0.1` in `config/server.yaml` and `scripts/dev.sh` (`INFERENCE_X_HOST` override); (4) load `config/logging.yaml` via `dictConfig` in `main.py` (rotating file + console); (5) `playground/url_validation.py` blocks private/link-local/loopback hosts unless `--allow-internal`.
- Consequences: Playground `preflight_compare` can no longer parse model-mismatch hints from sanitized API errors — use `/health` `loaded_models` or `--sequential` instead. Local dev requires `--allow-internal` for `localhost` URLs. Makefile `playground` targets pass that flag automatically.
- Deferred to Phase 6 (with reasons):
  - **API authentication** — needs operator identity model and key storage; out of scope for a single-GPU dev server.
  - **Rate limiting / inference concurrency caps** — requires queue design and 503 contract; deferred until multi-tenant or public exposure.
  - **Streaming / request wall-clock timeouts** — needs engine cancellation semantics vLLM does not expose cleanly on the sync `llm_engine` path.
  - **Request body size limits at middleware** — depends on Starlette/FastAPI global limit policy coordinated with observability body peek.

### DEC-SEC-01
- Date: 2026-06-08
- Status: accepted
- Context: Security audit of all transitive dependencies run with `uv run pip-audit`.
- Decision: Accept the single finding. Full output:
  ```
  Found 1 known vulnerability in 1 package
  Name      Version  ID              Fix Versions
  --------- -------  --------------  ------------
  diskcache  5.6.3   CVE-2025-69872  (none listed)
  ```
  `diskcache` is a transitive dependency of `vllm` and is not directly imported or called by InferenceX code. No public fix version is available at audit time. The risk surface is limited to local-only execution (server binds to 127.0.0.1). Re-audit when vllm ships an updated transitive dep.
- Consequences: Known CVE accepted under local-only deployment assumption. Must re-evaluate before any public network exposure.

### DEC-GIT-01
- Date: 2026-06-08
- Status: accepted
- Context: `logs/` directory is created by the server at runtime but was absent from `.gitignore`, risking the log file (`logs/inference_x.log`) being tracked by git.
- Decision: Added `logs/` to `.gitignore`. If the log file was already committed before this fix, run `git rm --cached logs/inference_x.log` manually to stop tracking it without deleting the file on disk.
- Consequences: Future server runs will not accidentally commit log files. The manual `git rm --cached` step is a one-time cleanup for repos that tracked the file before this fix.

### DEC-DEFER-01
- Date: 2026-06-08
- Status: accepted (deferred)
- Context: API authentication — no identity model or key storage exists.
- Decision: Not implemented. Server binds to `127.0.0.1` (loopback-only), providing network-level isolation for single-user local use. Authentication must be designed and implemented before any public network exposure.
- Consequences: Any process that can reach localhost 8000 can call the API without credentials. Acceptable for local-only dev; not acceptable for shared or public deployments.

### DEC-DEFER-02
- Date: 2026-06-08
- Status: accepted (deferred)
- Context: Rate limiting and inference concurrency caps — no queue or 503 backpressure exists.
- Decision: Not implemented. Single-user local use case with one request at a time. A queue design with 503 contract is required before any multi-tenant or public exposure scenario.
- Consequences: A burst of concurrent requests will all enter the vLLM engine simultaneously; GPU memory and latency will degrade. Acceptable for single-user dev.

### DEC-DEFER-03
- Date: 2026-06-08
- Status: accepted
- Context: Streaming timeout — needed to prevent stalled engine threads from holding SSE connections open indefinitely.
- Decision: Implemented a per-token asyncio timeout in `ChatService.stream_response()` controlled by `INFERENCE_X_STREAM_TIMEOUT_S` (default 120s). Uses `asyncio.wait_for` on `gen.__anext__()` inside the service layer. On timeout, emits an error SSE event and terminates the stream with `data: [DONE]`. The sync vLLM thread continues to run until the OS reclaims it (daemon thread), but the client connection is released cleanly.
- Consequences: Clients receive a clear error event and the HTTP connection closes within `stream_timeout_s` seconds. The background worker thread is not explicitly cancelled (vLLM sync engine has no cancellation hook), but daemon thread semantics ensure it does not prevent process exit.

### DEC-027
- Date: 2026-06-08
- Status: accepted
- Context: Phase 6 benchmark advisor needs a scoring function to rank models by hardware fit.
- Decision: Weighted composite score (0–100) across four dimensions:
  - **40% Throughput** — most user-visible metric; determines how fast the chat feels
  - **30% TTFT (inverted)** — time-to-first-token drives perceived interactivity more than raw throughput
  - **20% VRAM headroom** — remaining free VRAM after model load; a hard gate (score=0) when VRAM exceeded
  - **10% Quantization fit** — placeholder weight reserved for INT8/FP8 quantization scoring in a future phase; currently always 1.0
  All dimension scores are normalized against the best result in the batch (0–1) before weighting. The hard VRAM gate cannot be overridden by other scores.
- Consequences: Models that technically exceed available VRAM are always ranked last (score=0, viable=False) regardless of throughput numbers. The quantization weight is intentionally reserved to avoid a weight-sum change when implemented.

### DEC-028
- Date: 2026-06-08
- Status: accepted
- Context: Phase 6 hardware profiler must work on multiple environments: WSL2 with CUDA, bare Linux with CUDA, and CPU-only machines.
- Decision: Three-level fallback chain:
  1. **pynvml** Python bindings — fastest, most accurate, works reliably on CUDA systems with NVML driver
  2. **nvidia-smi subprocess** — subprocess parse of `--query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits`; works when pynvml is absent but NVIDIA drivers are present
  3. **CPU-only profile** — fallback when neither GPU path works; `has_gpu=False`, `vram_*=0.0`; VRAM-dependent models will all be hard-gated to non-viable by the advisor
  CPU cores and RAM always come from `psutil` when available, with `os.cpu_count()` and 16.0 GB as final fallbacks. Both `pynvml` and `psutil` are optional extras (`[project.optional-dependencies] hardware`) — the profiler degrades without them.
- Consequences: Hardware profiler is always safe to call; never raises. On pure CPU machines the advisor marks all GPU-bound models non-viable, which is correct behavior.

### DEC-026
- Date: 2026-06-08
- Status: accepted
- Context: The playground needed a better first-run experience: the operator had to know a model name before launching. The UI also lacked polish for model switching, response rendering, and quick-glance stats.
- Decision: Add `playground/startup_screen.py` — a `ModalScreen[str]` that fetches `/v1/models` on mount and presents a `RadioSet` for selection before the main TUI renders. On error or empty list, falls back to a manual `Input`. Six UI changes: (1) startup model-select screen pushed via `push_screen_wait` in `on_mount`; (2) response area already uses `Markdown` — confirmed correct; (3) header replaced with `Horizontal` containing title / URL / health `Label` widgets and a `Rule` separator; (4) model selector in prompt bar replaced with Textual `Select` widget; (5) status bar extended with per-request token-count and latency columns; (6) empty-state message in each `ResponsePanel` hidden when first token arrives, shown on clear. `--model` flag retained as pre-selection hint for the startup screen. Tab key removed from `cycle_model` binding to avoid focus-navigation conflict.
- Consequences: Launching without `--compare` now requires an interactive model selection step; `--compare` skips the screen. No env-var reading for model selection in the playground (the `--model` flag and server-provided model list are the only sources). `test_app.py` unchanged — the startup screen only runs inside `app.run()`.

### DEC-029
- Date: 2026-06-24
- Status: accepted
- Context: The Claude-style chat CLI uses `Markdown.get_stream()` / `MarkdownStream` for efficient token rendering. The prior `textual>=0.60` constraint predates Textual 1.0 versioning and would allow 0.x releases that lack this API.
- Decision: Pin `textual>=8.2,<9` in `pyproject.toml`.
- Consequences: Playground TUIs require Textual 8.x. Lockfile resolves to 8.2.7 in the current environment.

### DEC-030
- Date: 2026-06-24
- Status: accepted
- Context: `make playground` and `make chat` start uvicorn in the background on the same TTY as the Textual alternate-screen UI. Server INFO/WARNING logs paint over the TUI after launch.
- Decision: Redirect background uvicorn stdout/stderr to `logs/playground-server.log` in Makefile targets only (`playground`, `playground-compare`, `chat`). No server-side env hook or logging.yaml changes — `./scripts/dev.sh serve` keeps console logging.
- Consequences: Operators tail `logs/playground-server.log` for server diagnostics during TUI sessions. Foreground dev server behavior unchanged.

### DEC-031
- Date: 2026-06-25
- Status: accepted
- Context: Saved benchmark results lacked a `hardware` field, so `make advise` could score
  desktop VRAM deltas against a laptop's current free VRAM. Legacy JSON in `benchmarks/results/`
  (formerly `docs/benchmarks/`) had no provenance.
- Decision: Add optional `hardware: HardwareProfile` to `BenchmarkResult` (default `None` for
  legacy files). Persist `hardware_before` from `BenchmarkRunner.run()`. Return `AdvisorReport`
  (`ranked` + `warnings`) from `ModelAdvisor.rank()`: skip results when saved hardware
  mismatches current GPU name or VRAM total (>0.5 GB tolerance); soft-warn when `hardware`
  is null but still rank. GPU name matching was substring-based initially; **DEC-036** changed
  it to exact canonical name match. Expose `warnings` on `GET /v1/benchmark/advise` and print
  `WARNING:` lines in `scripts/advise.py`.
- Consequences: Cross-machine result files are omitted from rankings with a clear warning.
  Operators should re-run `make benchmark MODEL=…` after hardware changes. Scoring weights
  unchanged.

### DEC-032
- Date: 2026-06-25
- Status: accepted
- Context: `peak_vram_delta_gb` used `max(0, free_before − free_after)`. When the server
  already held model weights, free VRAM barely changed during the run → reported 0.00 GB
  despite ~3 GB in use. Advisor then treated `vram_used = 0` and passed the VRAM gate.
- Decision: Compute footprint as `vram_total − min(free_before, free_after)` via
  `_peak_vram_footprint_gb()` in `runner.py`. Keep the JSON field name `peak_vram_delta_gb`
  for backward compatibility with stored results and the advisor API.
- Consequences: Pre-loaded model workflow reports realistic VRAM usage (~3 GB for
  qwen2.5-0.5b on 6 GB GPU). Cold-load runs still capture footprint when free VRAM drops.

### DEC-033
- Date: 2026-06-25
- Status: accepted
- Context: Playground loading screen showed a generic "Model failed to load — see log lines
  above" while the real error (vLLM OOM, insufficient VRAM) lived only in
  `logs/playground-server.log`. `extract_error_summary()` matched too few line patterns.
- Decision: Broaden `playground/log_feed.py` `extract_error_summary()` (120-line tail, vLLM
  ERROR patterns, timeout heuristic for stuck weight loads). Fallback text:
  `Startup failed — see logs/playground-server.log for details`. Remove duplicate error
  line in `LoadingScreen.set_error()` (log tailer already surfaces it).
- Consequences: Failure banner and RichLog show actionable summaries when parseable;
  operators still tail the full log file for stack traces.

### DEC-034
- Date: 2026-06-25
- Status: superseded by DEC-036
- Context: After DEC-032, `peak_vram_delta_gb` reflects warm/steady-state VRAM footprint
  (model already loaded on the server). The advisor viability gate compared that footprint
  directly to current free VRAM. Cold vLLM startup allocates weights, KV cache, and CUDA
  overhead in one burst — typically 15–25% above steady state — so models that fit the warm
  footprint could still OOM on cold load.
- Decision: Apply `COLD_START_MARGIN = 1.20` in `advisor.py`:
  `effective_required = peak_vram_delta_gb × margin`; viable only when
  `effective_required < vram_free_gb`. Override via `INFERENCEX_COLD_START_MARGIN`.
  Recommendation strings show `footprint × margin = required`. VRAM headroom scoring uses
  `effective_required` for consistency with the gate.
- Consequences: Superseded — multiplying an absolute footprint inflated requirements and
  comparing against runtime free VRAM conflated server state with static capacity.

### DEC-036
- Date: 2026-06-30
- Status: accepted
- Context: Three advisor scoring bugs surfaced on real benchmark data. (1) TTFT used
  `prompt_results[0]`, which is always a cold-start outlier (CUDA graph miss), unfairly
  penalizing rankings. (2) The DEC-034 gate multiplied absolute `peak_vram_delta_gb` by 1.20
  and compared to `vram_free_gb`, marking models non-viable that had run successfully
  (e.g. 5.38 GB footprint on a 6 GB card). (3) GPU hardware matching used substring
  containment, so an RTX 3060 benchmark could validate on an RTX 3060 Ti.
- Decision:
  - **Warm TTFT:** mean TTFT over `prompt_results[1:]`; fall back to the sole prompt when
    only one result exists.
  - **VRAM gate:** `vram_required = peak_vram_delta_gb + 0.5 GB`; viable when
    `vram_required ≤ vram_total_gb`. VRAM headroom score uses the same denominator
    (`vram_total_gb`). Remove `COLD_START_MARGIN` and `INFERENCEX_COLD_START_MARGIN`.
  - **Hardware match:** exact GPU name equality (case-insensitive, stripped) plus
    ±0.5 GB `vram_total_gb` tolerance.
  - **Static fallback:** when `benchmarks/results/` is empty, `make advise` prints a
    config-based VRAM fit table from `models.yaml` instead of exiting with an error.
- Consequences: Rankings reflect steady-state latency. Models that fit total VRAM are
  correctly marked viable regardless of current free VRAM. Cross-GPU-variant mismatches are
  skipped. New users get actionable guidance before their first benchmark run.

### DEC-035
- Date: 2026-06-25
- Status: accepted
- Context: `gpu_memory_utilization=0.90` against total VRAM requires 7.2 GB on an 8 GB card;
  machines with ~1 GB overhead (driver + desktop) fail immediately even when 6.93 GB is
  free. vLLM's default `max_model_len` of 32768 pre-allocates ~2 GB KV cache per small
  model, blocking dual-model use on 6–8 GB GPUs.
- Decision: `gpu_memory_utilization: auto` computes `(vram_free - buffer) / vram_total` at
  startup (buffer default 0.4 GB via `INFERENCEX_VRAM_SAFETY_BUFFER_GB`, clamped
  [0.50, 0.95]). All models default to `auto`. `max_model_len: 8192` for sub-2B models
  (covers local chat; drops KV cache from ~2 GB to ~0.5 GB). Default registry targets
  8 GiB WSL2 (dense models up to ~1.5B); 3B+, hybrid, and 8B models are omitted unless
  added manually with quantization or more VRAM.
- Why free-VRAM basis: vLLM's utilization is a fraction of total VRAM — deriving the
  fraction from free VRAM is the only way to stay within actually available memory.
  Sizing uses `torch.cuda.mem_get_info` when CUDA is active (same allocator view as
  vLLM); nvidia-smi is used only as fallback before torch init.
- Why 8192: ~6000 words of context; sufficient for playground and benchmark use cases;
  raise per-model in `models.yaml` when longer context is needed.
- Why 0.4 GB buffer: covers CUDA context growth during inference.
- Deferred: 8B+ and 3B+ model entries for 8 GiB GPUs (quantization or manual registry
  entries); multi-model sequential VRAM profiling.
- Consequences: Startup logs show resolved utilization with `(free − buffer) / total`
  breakdown. `BenchmarkResult` stores `max_model_len`; advisor warns on config drift.

### DEC-037
- Date: 2026-07-01
- Status: accepted
- Context: DEC-035's sizing was load-time only and not quant-aware
  (`_BYTES_PER_PARAM = 2` hardcoded), so a 4-bit model would be sized as if it needed
  full bf16 VRAM and rejected or badly under-utilized. There was also no explicit,
  documented VRAM-capacity contract per GPU class, and the observability HTTP surface
  was stubbed: `api/routes/metrics.py` and `schemas/metrics.py` were 0-byte files, and
  streaming (SSE) chat responses recorded no TTFT or tokens/sec because the middleware
  only reads a buffered JSON body for non-streaming responses.
- Decision (Phase 1 of the VRAM-aware architecture plan; Phase 2 admission control and
  Phase 3 CPU offload are deferred):
  - **Quant-aware sizing:** `utils/vllm_pool_config.py` resolves GPU bytes/param from
    `ModelEntry.quantization` (bf16/None → 2, int8/fp8 → 1.0, awq/gptq/int4 → ~0.55)
    instead of a hardcoded constant; threaded through `estimate_weight_gib()` and every
    caller (`estimate_engine_footprint_gib`, `validate_pool_fits`,
    `_weight_scaled_utilization`, `_apply_sequential_vram_caps`,
    `_single_engine_utilization`). Added `qwen2.5-7b-awq` (`quantization: awq`) to
    `config/models.yaml` as the first exercised 4-bit variant — sized for the 12GB tier,
    not loaded by default alongside the 6GB dev pool.
  - **VRAM tiers:** `config/vram_tiers.yaml` (new) declares 6gb/12gb/24gb tiers
    (`gpu_memory_utilization_ceiling`, `max_model_len_cap`, `max_num_seqs`, `block_size`,
    `kv_cache_dtype`) as a documented capacity contract. `utils/vram_tiers.py` resolves
    the highest tier whose `min_vram_gb` floor the probed GPU clears, falling back to the
    lowest tier (with a warning) when probing fails or reports something below every
    floor — never guesses a higher/less-safe tier. `AppSettings.get_vram_tier()` wires it
    to `profile_hardware()`; `deps.initialize_app()` logs the resolved tier at startup.
    Tiers are **resolved and logged only** in this phase — enforcement (admission control
    against `max_num_seqs`/`max_model_len_cap`) is explicitly Phase 2, not built here.
  - **Observability wiring:** filled `schemas/metrics.py` (`MetricsResponse`,
    `VramSummary`, `ModelVramBreakdown`) and `api/routes/metrics.py`
    (`GET /v1/metrics`), registered in `api/main.py`. Response combines request metrics
    (`MetricsService.summary()`) with a live per-model VRAM breakdown (weights estimate,
    real post-load `kv_capacity_tokens` from `VLLMEngine.kv_capacity_tokens`, free/total
    GiB from `probe_gpu_memory_gib()`).
  - **Streaming TTFT/tokens-per-sec:** `ObservabilityMiddleware` wraps
    `response.body_iterator` for SSE chat responses (instead of skipping token
    extraction) to time-to-first-chunk and approximate completion tokens via whitespace
    word count over each chunk's `delta.content` (same approximation as the playground
    and `benchmarks/runner.py`, since streaming carries no final `usage` block — see
    DEC-023). Records exactly one `RequestRecord` when the stream ends (normal
    completion or client disconnect via `GeneratorExit`); the unconditional
    `recorder.record()` call is skipped for this branch to avoid double-counting.
  - **`GET /v1/models` metadata:** additive `quantization`, `max_model_len`,
    `estimated_weights_gib` fields on `ModelObject` so clients can pick a variant that
    fits their tier without a separate call.
- Known limitation: an engine failure *after* SSE headers are sent is not observable as
  `error=True` — Starlette's `BaseHTTPMiddleware` surfaces the inner task's exception to
  the outer ASGI call only after `dispatch()` has finished sending the response, so a
  mid-stream engine failure looks like a normal end-of-stream to the wrapper. Partial
  TTFT/token count is still recorded; only the error flag is unreliable for this case.
- Consequences: `qwen2.5-7b-awq` loads and serves standalone on 12GB+ hardware (not yet
  available to validate live; unit-tested against the quant-aware estimator). On the 6GB
  dev box, `GET /v1/metrics` returns a live weights/KV/free breakdown and streaming TTFT
  (~255ms) / tokens-per-sec (~29) verified against a running server. 352/352 unit tests
  pass. Phase 2 (`routing/admission.py`, `max_context_tokens`/`precision`/`priority`
  request fields, engine knob surfacing, non-streaming batching fix) and Phase 3 (CPU
  offload, prefix caching) remain future work.

### DEC-038
- Date: 2026-07-01
- Status: accepted (admission control) / reverted-with-findings (batching fix)
- Context: Phase 2 of the VRAM-aware plan (DEC-037). Two independent goals: (1) enforce
  context-length and KV-pool limits before dispatch instead of letting oversized requests
  reach vLLM and fail unpredictably; (2) give non-streaming completions the same
  continuous-batching concurrency `generate_stream` already had, since `_run_completion`
  held a lock across the entire blocking `.chat()`/`.generate()` call.
- Decision (admission control, shipped):
  - New `routing/admission.py`: `AdmissionController.admit(routed_model, request, engine)`
    runs after `TaskRouter.select()`, before engine dispatch. Two gates:
    1. **Context length** — `prompt_tokens + requested_output` must fit
      `min(ModelEntry.max_model_len, tier.max_model_len_cap, request.max_context_tokens)`.
      Prompt tokens come from `engine.count_prompt_tokens()` (new `VLLMEngine` method,
      real tokenizer-based) via `getattr`, same optional-attribute pattern already used
      for `kv_capacity_tokens` in `api/routes/metrics.py` — not added to `BaseEngine`'s
      abstract contract, so no test-stub engine needed updating. Falls back to a chars/4
      estimate when the engine has no tokenizer.
    2. **KV pressure** — an in-memory per-model `_KVReservationTracker` (token counts, not
      GiB — simpler and more accurate than converting through `kv_gib_for_tokens` since
      `kv_capacity_tokens` is already real, post-load tokens) compares in-flight
      reservations against `engine.kv_capacity_tokens * 0.9`.
    - `priority: "interactive" | "batch"` (new request field) decides the response to
      either gate: interactive clamps `max_tokens` down to what fits (rejecting only if
      the clamped room is below 16 tokens — a "clamp" to a handful of tokens is really a
      rejection in disguise); batch always rejects instead of silently truncating.
    - Both gates fail open (don't block) when the underlying number is unavailable (no
      tokenizer, no reported KV capacity) — matches this codebase's existing posture for
      advisory signals (e.g. VRAM tier resolution failure in `api/deps.py`).
    - New request fields: `max_context_tokens`, `max_output_tokens` (alias for
      `max_tokens`, additive/back-compat), `priority`. `precision` from the original plan
      was deliberately **not** added — it's meaningless without variant sets (Component 1
      of the plan, "group models into variant sets by quantization"), which don't exist
      yet; adding an inert field would be dead API surface.
    - `ContextTooLongError(ValueError)` reuses the existing sanitized 400 handler.
      `EngineSaturatedError` is new — mapped to 429 + `Retry-After` via a new handler in
      `api/errors.py`, registered in `api/main.py`.
    - `ChatService` gained an optional `admission` constructor param defaulting to
      `AdmissionController(registry)` (no tier, 4096-token cap) so every existing direct
      `ChatService(...)` test call site keeps working unchanged; `api/deps.py` wires the
      real controller with the resolved VRAM tier via a new `_build_admission_controller`.
  - Live-verified: a prompt over `max_context_tokens` returns 400; normal chat and
    streaming still work end-to-end against the running 6GB dev server.
- Decision (non-streaming batching fix, reverted): refactoring `_run_completion` to drive
  `add_request`/`step()` directly (mirroring `generate_stream`, releasing the lock between
  steps) was implemented, unit-tested with mocks, and passed — but a **live concurrency
  test with 2 real concurrent non-streaming requests reproducibly failed one of them** with
  "vLLM produced no output". Root cause: `generate_stream` tolerates a request's own
  `step()`-call "turn" being preempted by another thread, because `output.outputs[0].text`
  is cumulative — whichever call next happens to surface your `request_id` lets you catch
  up. A one-shot `finished=True` handoff has no such recovery: if another thread's `step()`
  call is the one that returns your request's terminal output, it discards it (request_id
  mismatch) and `has_unfinished_requests()` can go globally False before your own thread
  ever sees it — no retry, no error signal until the "no output" `RuntimeError`. The
  original plan called this "low-risk, self-contained"; live testing showed otherwise.
  Reverted to the original blocking `.chat()`/`.generate()` call (proven safe, one lock
  held for the whole generation). A correct fix needs a single shared per-engine driver
  thread that multiplexes `step()` output to per-request queues instead of N independent
  request-owned loops — left as explicit follow-up, not shipped half-verified.
- Consequences: Admission control is live and tested (17 new unit tests: 15 in
  `test_admission.py`, 2 route-level 400/429 integration tests). Non-streaming requests
  still serialize behind one lock per model, same as before this change — no regression,
  no improvement. `engine_knob surfacing` (block_size, max_num_batched_tokens,
  kv_cache_dtype, enable_prefix_caching) from the original Phase 2 scope was not started
  this round; deferred alongside the batching-fix follow-up.

### DEC-039
- Date: 2026-07-01
- Status: accepted
- Context: DEC-038 named the correct fix for non-streaming continuous batching but did not
  ship it: a single shared per-engine driver thread that is the only caller of
  `add_request`/`step()`, demultiplexing every output to its per-request destination by
  `request_id`. This closes that follow-up (`openspec/changes/add-engine-driver-thread`).
- Decision:
  - New `engines/driver.py`: `EngineDriver` owns one vLLM sync `llm_engine` exclusively.
    Callers submit a `(prompt, sampling)` pair — `submit_stream()` returns a `queue.Queue`
    of text deltas terminated by `None`, `submit_complete()` returns a
    `concurrent.futures.Future` resolved with the final `RequestOutput`. The driver thread
    drains newly-submitted requests (calling `add_request` itself, from that same thread),
    then calls `step()` and dispatches each output to its registered channel by
    `request_id` — since exactly one thread ever calls `add_request`/`step()`, there is no
    "wrong" thread left to discard a terminal output the way DEC-038's per-request loops
    could.
  - `VLLMEngine._run_completion` and `generate_stream` both now build a prompt via
    `_stream_prompt()` and submit through `self._driver` instead of running their own
    step loop (`generate_stream`) or a single blocking `.chat()`/`.generate()` call
    (`_run_completion`). This is the "unification": both are now thin adapters over one
    driver, differing only in channel shape.
  - `_POOL_STEP_LOCK` (cross-engine serialization when `pool_size > 1`) is unchanged in
    meaning — each `EngineDriver` is constructed with it as its `step_lock` when
    `pool_size > 1`, a private per-engine lock otherwise — only the caller moved from
    "each request's own thread" to "the one driver thread."
  - Driver failure (`step()` raising) is broadcast to every currently-pending
    channel — completion futures get `set_exception`, stream queues receive the
    exception object itself (callers must check `isinstance(chunk, BaseException)`
    before treating a queue item as a text chunk) — and flips `EngineDriver.is_dead`,
    which `VLLMEngine.is_healthy()` now also checks.
  - Idle-wait uses a bounded `queue.get(timeout=0.05)` rather than a busy loop or a fixed
    sleep, so shutdown stays responsive without spinning when no requests are in flight.
  - `BaseEngine`'s abstract contract, `ChatService`, and all request/response schemas are
    unchanged — this is entirely internal to `VLLMEngine`.
- Live-verified: with a running `opt-125m` server, 2 and then 4 concurrent non-streaming
  `POST /v1/chat/completions` requests all returned complete, correctly-attributed,
  non-truncated responses (finish_reason `stop`/`length`, distinct content per request).
  A follow-up determinism check (`temperature=0`, unique per-request tokens) showed the
  concurrent run's output was byte-identical to the same prompts run fully sequentially,
  confirming no cross-request state leakage. Streaming chat completions were unaffected.
- Consequences: Non-streaming completions get the continuous-batching concurrency
  streaming already had, closing the last open item from DEC-038's admission-control
  round. 379/379 unit tests pass (371 prior + 8 new `test_engine_driver.py` cases plus a
  new `test_generate_completes_via_driver` case). Engine knob surfacing and model variant
  routing (the other two DEC-038/Phase-2-adjacent deferrals) remain separately proposed
  under `openspec/changes/add-engine-knob-surfacing` and
  `openspec/changes/add-model-variant-routing`, not implemented this round.

### DEC-040
- Date: 2026-07-01
- Status: accepted
- Context: `config/vram_tiers.yaml` has declared `block_size`/`kv_cache_dtype`/`max_num_seqs`
  per tier since DEC-037, with a comment noting they weren't yet consumed by the engine.
  `_build_engine_pool` (`api/deps.py`) never resolved a VRAM tier at all — only
  `initialize_app()` did, purely for a log line. `AdmissionController`'s KV-pressure gate
  had no visibility into `max_num_seqs`: a saturated sequence-concurrency ceiling just
  queued silently inside vLLM's scheduler instead of surfacing as an explicit signal.
  This closes `openspec/changes/add-engine-knob-surfacing`.
- Decision:
  - `VramTier` (`utils/vram_tiers.py`) and `config/vram_tiers.yaml` gain two additive
    fields per tier: `max_num_batched_tokens` (bounds the transient prefill-burst compute
    spike, distinct from steady-state KV usage) and `enable_prefix_caching`.
    `.get()`-defaulted (`None`/`False`) so old tier files keep parsing unchanged. 6gb:
    2048 / `false` (tight KV budget, prefix caching would hold blocks longer than
    affordable); 12gb/24gb: 4096/8192 / `true`.
  - `ModelEntry.max_num_batched_tokens: Optional[int]` — same override shape as the
    existing `max_num_seqs`. `enable_prefix_caching` is deliberately tier-only, no
    per-model override — a single engine-startup flag with no meaningful per-model
    variance in a one-model-per-process engine.
  - New `apply_tier_knobs(config, tier)` (`utils/vllm_pool_config.py`): resolves
    `max_num_seqs`/`max_num_batched_tokens` as `min(model override or tier value, tier
    value)` — same composition `AdmissionController._context_ceiling` already uses for
    `max_model_len` — and passes `block_size`/`kv_cache_dtype`/`enable_prefix_caching`
    straight from the tier. A `None` tier leaves the config dict untouched (fail-open).
  - `VLLMEngine.__init__` now forwards `max_num_batched_tokens`, `block_size`,
    `kv_cache_dtype`, `enable_prefix_caching` to `LLM(**kwargs)` alongside the existing
    `max_num_seqs` — confirmed all five are real `EngineArgs` fields on the installed
    vLLM 0.22.1 (`LLM.__init__`'s `**kwargs` forwards to `EngineArgs`).
  - `_build_engine_pool` (`api/deps.py`) gained `_resolve_vram_tier_for_pool` (same
    fail-open-with-a-warning try/except pattern as `_build_admission_controller`) and
    now calls `apply_tier_knobs` on each model's config before constructing its
    `VLLMEngine` — previously this function never resolved a tier at all.
  - `AdmissionController` (`routing/admission.py`) gained a third gate,
    `_InFlightSeqTracker`: a per-model in-flight *request count* (not tokens), checked
    first in `admit()` against the resolved `max_num_seqs` ceiling. Unlike the KV-token
    gate, there is no clamp path — both `interactive` and `batch` priority get
    `EngineSaturatedError` (429 + `Retry-After`) when saturated, since a sequence slot
    can't be partially granted. The increment happens only at the very end of `admit()`
    (alongside the KV token reservation) so an earlier-gate rejection never leaks a slot
    that `release()` would never be called to free.
- Live-verified: with a running `opt-125m` server on the 6gb tier, vLLM's own startup log
  shows `max_num_batched_tokens=2048` and `enable_prefix_caching=False` — the resolved
  tier values, not vLLM's defaults — reaching the real engine config, and the server
  served a normal chat completion afterward.
- Consequences: 401/401 unit tests pass (17 new: `apply_tier_knobs` composition cases,
  `VLLMEngine` kwarg-forwarding cases, `_build_engine_pool` tier-resolution wiring cases,
  6 new `AdmissionController` sequence-gate cases, 1 new route-level 429 test). Model
  variant routing (`openspec/changes/add-model-variant-routing`) remains separately
  proposed, not implemented this round.

### DEC-041
- Date: 2026-07-01
- Status: accepted
- Context: DEC-038 deliberately did not add a `precision` request field, calling it
  "meaningless without model variant sets... which don't exist yet." This closes that
  prerequisite: `openspec/changes/add-model-variant-routing`.
- Decision:
  - `ModelEntry.family: Optional[str]` (`schemas/model.py`) — additive; entries with no
    `family` behave exactly as before (a "family of one," keyed by their own `name`).
  - `ModelRegistry.variants(family)` (`services/model_service.py`) groups entries by
    `family`, falling back to `name` for ungrouped entries — so `variants("some-name")`
    still returns that single entry unaffected by this method's existence.
  - New `routing/variant_selector.py`: `select_variant(family, registry, tier,
    available_vram_gib)` sorts a family's variants by descending
    `_bytes_per_param(quantization)` — reusing `utils/vllm_pool_config`'s existing
    quant-aware table as the *only* source of precision ordering, deliberately not a
    second rank field that could drift out of sync with it — and returns the first
    variant whose `estimate_weight_gib()` fits `available_vram_gib *
    tier.gpu_memory_utilization_ceiling`. Raises `NoVariantFitsError` (naming every
    variant's estimated size) when none fit, or when the family name matches no
    registered entry at all.
  - Selection is **load-time only**: `_build_engine_pool` (`api/deps.py`) resolves each
    name in `INFERENCE_X_LOADED_MODELS` via a new `_resolve_loaded_model_names` — a name
    matching a registered `ModelEntry.name` exactly is used as-is (bypasses the
    selector, unchanged from before); otherwise it's treated as a family name and
    resolved via `select_variant`. A name that is neither a concrete entry nor has any
    registered variants (`registry.variants(name)` empty), or arrives when no VRAM tier
    resolved, falls through to `registry.get(name)`'s existing "not registered" error
    unchanged — one consistent error path for a genuinely unknown name regardless of
    tier availability.
  - `TaskRouter`/`AdmissionController`/`ChatService` are untouched — they only ever see
    the concrete, already-resolved name from `EnginePool`, exactly as before this change.
  - Known scope boundary: `INFERENCE_X_DEFAULT_MODEL` is **not** resolved through the
    selector — `DefaultModelPolicy` still requires an exact registered name. An operator
    loading a family should still set the default model to one of that family's concrete
    variant names, not the family name itself; resolving the router's default through
    variant selection would require threading tier/VRAM state into `_build_router` too,
    which DEC-038's original scoping for this component didn't call for.
- Live-verified: a temporary config (`vram_tiers.yaml` copied from the real one, a
  two-entry `models.yaml` grouping `facebook/opt-125m` and
  `Qwen/Qwen2.5-0.5B-Instruct` under `family: tiny`) started with
  `INFERENCE_X_LOADED_MODELS=tiny` against a real vLLM engine — the server resolved the
  family to the first-listed variant (`tiny-a`), loaded it, reported healthy, and served
  a normal chat completion. `config/models.yaml` also gained a real worked example
  (`qwen2.5-7b-bf16`/`qwen2.5-7b-awq`, both genuine HuggingFace repos, grouped under
  `family: qwen2.5-7b`) not live-verified on this 6-8GB dev box (needs 12GB+/24GB+).
- Consequences: 419/419 unit tests pass (18 new: `ModelEntry.family`/`variants()`,
  `variant_selector` selection-order/fallback/no-fit cases, `_build_engine_pool`
  family-resolution wiring). All three Phase 3 sub-changes (driver thread, engine
  knobs, variant routing) are now shipped.
