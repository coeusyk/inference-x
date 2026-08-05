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

### DEC-042
- Date: 2026-07-01
- Status: accepted
- Context: DEC-041 (70ca241) shipped load-time variant selection for
  `INFERENCE_X_LOADED_MODELS` but explicitly left `INFERENCE_X_DEFAULT_MODEL` out of
  scope: `_build_router`'s `TaskRouter(registry, default_model)` → `DefaultModelPolicy`
  required an exact registered `ModelEntry.name`, so a family name there either matched
  nothing (immediate startup failure) or, if it happened to collide with a real entry
  name, silently used the wrong model. `variant_selector.select_variant()` was never
  consulted for this path. This closes that scope boundary.
- Decision:
  - New `_resolve_default_model(registry, default_model, tier, available_vram_gib)` in
    `api/deps.py`: a name already in the registry passes through unchanged (zero
    behavior change for existing operators). A name matching a `ModelEntry.family` —
    only when a VRAM tier is resolved — is resolved via `select_variant()`, the exact
    same function `_resolve_loaded_model_names` already calls for
    `INFERENCE_X_LOADED_MODELS`. Any other value (unrecognized name, or a real family
    name when no tier is available to size it) is returned **unchanged**, deliberately
    deferring to `TaskRouter`'s existing `DefaultModelPolicy` constructor, which raises
    its own pre-existing `"Default model '{x}' is not in the registry"` `ValueError` —
    this preserves that exact error path/wording for the genuinely-unresolvable case
    instead of introducing a second, differently-worded error message for what was
    already a handled failure mode.
  - `_build_router` now resolves the VRAM tier (`_resolve_vram_tier_for_pool`, already
    used by `_build_engine_pool`) and probes free/total VRAM before constructing
    `TaskRouter`, threading both into `_resolve_default_model`.
  - Detection uses the model registry directly (`ModelEntry.family` membership), not a
    separate `INFERENCE_X_DEFAULT_MODEL_IS_FAMILY` flag — unambiguous from data already
    present, no extra operator-facing config.
  - Hard-error, no silent fallback: when a recognized family has no variant fitting the
    current VRAM budget, `select_variant()`'s existing `NoVariantFitsError` (already a
    `RuntimeError` subclass, naming the family, every variant's estimated size, and the
    tier's budget) propagates unmodified — no new exception type needed, and no
    fallback to "just pick the first/smallest variant anyway," which would silently
    contradict the tier the operator's hardware actually resolved to.
  - Resolution happens once at startup (`_build_router`, `@lru_cache`d), not per
    request — `INFERENCE_X_DEFAULT_MODEL` is operator config, and the loaded model set
    is already fixed for the life of the process; a per-request VRAM check belongs to
    `AdmissionController`, not the default-model lookup.
  - New INFO log on actual resolution: `Default model resolved: {family} → {variant}
    (tier: {tier_name})` — logged only when resolution changes the value, not for the
    concrete-passthrough case.
- Live-verified: a temporary two-model `family: tiny` config (reused from the DEC-041
  live check) started with `INFERENCE_X_DEFAULT_MODEL=tiny` and no
  `INFERENCE_X_LOADED_MODELS` override. Startup log showed `Default model resolved:
  tiny → tiny-a (tier: 6gb)`; a request with an unregistered `model` value (falling
  through `ExplicitModelPolicy` to the resolved default) returned
  `"model":"tiny-a"` in the response.
- Consequences: 425/425 unit tests pass (6 new in
  `test_default_model_resolution.py`). `INFERENCE_X_LOADED_MODELS` resolution logic
  (`_resolve_loaded_model_names`) and `variant_selector.select_variant()` internals are
  both untouched by this change. This was the last item on 3C's known scope-boundary
  list — that list is now empty.

### DEC-043
- Date: 2026-07-01
- Status: accepted
- Context: while validating DEC-042, `tests/unit/test_engine_driver.py::
  test_step_exception_is_broadcast_to_pending_completion_futures` was found flaky
  (~40% failure rate under repeated runs) on unmodified `develop` — confirmed via
  `git stash`, unrelated to any Phase 12 change. `EngineDriver._broadcast_exception`
  (`engines/driver.py`, DEC-038/DEC-039) sets `self._dead = True`, fails every
  request in `_pending`, drains and fails anything already sitting in `_submit_q`,
  then returns — the driver thread exits for good. `submit_stream`/`submit_complete`
  checked `self._dead` and called `_submit_q.put(...)` as two separate, unsynchronized
  steps. A request submitted in the gap between the dead-check reading `False` and
  its own `_submit_q.put()` landing — if `_broadcast_exception`'s drain had already
  run and found the queue empty in between — was enqueued with no thread left to
  ever read it, dispatch it, or fail it: the caller blocked for the full 300s
  completion timeout (or forever, for a stream) instead of receiving the engine's
  actual failure immediately.
- Decision:
  - `EngineDriver` gains one `threading.Lock` (`self._dead_lock`) plus
    `self._dead_exception: BaseException | None`, guarding `_dead`/`_dead_exception`
    together with the decision of whether a given `_submit_q.put()` may happen at
    all — not just the flag read.
  - `_submit` (the shared helper both `submit_stream` and `submit_complete` call)
    now does the dead-check and the enqueue as one atomic critical section: acquire
    `_dead_lock`; if `_dead`, raise `EngineDriverDeadError` immediately (chaining
    `_dead_exception` as the cause) without touching `_submit_q`; otherwise put the
    request onto `_submit_q` before releasing.
  - `_broadcast_exception` does its own atomic critical section under the same
    lock: set `_dead = True` and `_dead_exception = exc`, then drain and fail
    anything currently in `_submit_q`, all before releasing. Broadcasting to
    `_pending` stays outside the lock — `_pending` is only ever touched by the
    driver thread itself.
  - Sharing one lock between both critical sections is the actual fix, not just
    locking the flag read: a design that checks `dead` under a lock, releases it,
    and only then calls `_submit_q.put()` reopens the identical race one level up
    (`_broadcast_exception` could run its own set-and-drain in the gap between
    `submit`'s release and its put). Making "check dead, else enqueue" and "set
    dead, then drain" share one lock means the two can no longer interleave: either
    the enqueue lands before the drain (and is caught by it) or the flag is already
    set before the enqueue is attempted (and it never happens) — no third case.
  - The lock is held only across flag read/write and queue drain/put, never across
    `self._llm_engine.step()` — holding it there would serialize every submission
    against every step() call for no reason.
  - `EngineDriverDeadError` was already raised synchronously by `submit_stream`/
    `submit_complete` for the already-dead case before this change; that contract is
    unchanged for callers (`vllm_engine.py`'s catch sites are untouched) — only the
    race in the transition into the dead state is fixed.
  - `test_step_exception_is_broadcast_to_pending_completion_futures` was rewritten
    to gate the fake engine's `step()` on a registration count
    (`fake.fail_after_registered`) instead of relying on submission-ordering luck,
    making the "both requests registered before failure" scenario deterministic. A
    new `test_submit_after_driver_death_raises_immediately` regression-tests the
    fix directly: a submission after `driver.is_dead` is `True` raises
    `EngineDriverDeadError` in well under a second, not after a timeout.
- Consequences: 426/426 unit tests pass, including 160/160 runs of
  `test_engine_driver.py` under `pytest-repeat --count=20` (added as a dev
  dependency). No change to the driver thread's restart policy, the streaming
  path's chunking/delta logic, `EngineDriver`'s public method signatures, or any
  other module.

### DEC-044
- Date: 2026-07-02
- Status: accepted
- Context: a live `make benchmark MODEL=opt-125m` run reported implausible numbers for a
  125M-param model: 16.7 tok/s, 11030ms p50 latency, 7.06 GB peak VRAM delta. Two
  independent, unrelated bugs were found and fixed together because both were surfaced by
  the same benchmark run:
  1. **VRAM over-allocation.** `scale_model_config_for_pool()` (`utils/vllm_pool_config.py`)
     special-cased `gpu_memory_utilization: "auto"` with an early-return branch that
     computed `(vram_free - buffer) / vram_total` (DEC-035's original formula, with a 0.50
     floor for single-model) regardless of the model's actual weight/KV footprint. This
     meant a 125M-param model and a 7B model requesting `auto` got sized identically —
     whatever fraction of free VRAM happened to be available, not what the model needed.
     `resolve_gpu_memory_utilization()`, the function implementing this branch, became
     fully dead code once removed (confirmed via grep: zero other callers).
  2. **Throughput/latency regression.** `EngineDriver._run()` (`engines/driver.py`,
     introduced in DEC-039/a88b135 to fix a streaming race) called
     `self._submit_q.get(timeout=_IDLE_POLL_S)` — a blocking 0.05s wait — on *every*
     iteration of its loop, including while requests were already in flight. This capped
     every `step()` call to at most 20 Hz regardless of model speed, a ceiling the old
     unthrottled per-request loop DEC-039 replaced never had.
- Decision:
  - `scale_model_config_for_pool()` no longer special-cases `"auto"`. Both `"auto"` and an
    explicit float now flow through the same footprint-aware `_single_engine_utilization()`
    (pool_size ≤ 1) / `_weight_scaled_utilization()` + `_apply_sequential_vram_caps()`
    (pool_size > 1) path — `"auto"` means "no user-set ceiling" (see `_user_util_cap()`),
    not "ignore the model's footprint and grab a flat fraction of free VRAM."
    `resolve_gpu_memory_utilization()` deleted as dead code.
  - `EngineDriver._run()` split submission-draining into `_drain_submissions_blocking()`
    (only called when `self._pending` is empty — genuinely idle, bounds `shutdown()`
    responsiveness) and `_drain_submissions_nowait()` (called whenever requests are
    already in flight — never blocks, goes straight to the next `step()`).
- Live-verified on the RTX 4060 8 GiB dev box, opt-125m: peak VRAM delta 7.06 GB → 1.65 GB;
  mean throughput 16.7 → 329.3 tok/s (19.7×); p50 latency 11030ms → 518ms (21×).
- Consequences: two existing tests had asserted the buggy behavior directly
  (`test_auto_resolves_at_startup` expected utilization `0.82`;
  `test_auto_multi_model_uses_fresh_suggest_not_div_n` expected `qwen_scaled >= 0.42`) —
  both rewritten to assert the corrected, footprint-based values
  (`test_auto_matches_explicit_footprint_sizing`,
  `test_auto_multi_model_uses_weight_scaled_sizing`). No change to compare-mode splitting
  logic itself, `AdmissionController`, or the streaming/non-streaming dispatch contract
  `EngineDriver` exposes — only when it's allowed to block.

### DEC-045
- Date: 2026-07-02
- Status: accepted
- Context: after DEC-044's fix, `qwen2.5-7b-awq` (added in DEC-037 as the first exercised
  4-bit model, but per PHASES.md's Phase 7 post-phase note never validated live) was
  benchmarked directly on the 8 GiB dev box for the first time. It FAILED to load:
  `RuntimeError: GPU memory insufficient for KV cache` — `Model loading took 5.29 GiB
  memory` against an estimated ~3.6 GiB. Root cause: `estimate_weight_gib()`
  (`utils/vllm_pool_config.py`) applies the quantization bytes/param ratio (e.g. 0.55 for
  AWQ) uniformly to every parameter. AWQ/GPTQ-style weight-only quantization does not
  touch embedding lookups or (when untied) the separate lm_head projection — those stay at
  full precision. `Qwen/Qwen2.5-7B-Instruct-AWQ` has `tie_word_embeddings=False` and a
  152k vocab, so those two layers alone are ~1.1B unquantized params (~2 GiB) the estimate
  missed. Separately, `minicpm5-1b` (added in an earlier session) showed a real ~2.76 GiB
  gap between its estimated footprint (3.79 GiB) and measured peak VRAM (6.55 GiB), with no
  root cause found — the model happened to still fit, but the sizing was materially wrong.
- Decision:
  - `estimate_weight_gib()` now splits embedding + (untied) lm_head params out via a new
    `_embedding_param_count()` helper (uses HF config `hidden_size`/`vocab_size`/
    `tie_word_embeddings`) and prices them at bf16 (2 bytes/param) separately from the rest,
    which get the quantization ratio. Applies to any scheme with `bytes_per_param < 2`
    (AWQ/GPTQ/int4/int8/fp8), not just AWQ specifically — conservative in all cases, since
    it only ever adds weight back, never removes it.
  - Added `_architecture_overhead_gib()` + a small `_ARCH_OVERHEAD_GIB` lookup table
    (same shape as the existing quant-ratio table) for architectures with measured overhead
    the generic weights+KV+runtime estimate doesn't model. One entry: `minicpm: 2.8` GiB,
    documented explicitly as a fitted correction, not a diagnosed root cause. First attempt
    keyed this off HF config `model_type` — wrong, because `openbmb/MiniCPM5-1B`'s config
    self-reports `model_type="llama"` / `architectures=["LlamaForCausalLM"]` for tooling
    compatibility, so `model_type` cannot distinguish it from a real Llama model. Fixed to
    key off the model_path/repo id substring instead, confirmed live.
  - `qwen1.5-1.8b` (`Qwen/Qwen1.5-1.8B-Chat`, ungated) added to `config/models.yaml` as the
    first validated ~2B-class entry on 8 GiB WSL2 — `google/gemma-2-2b-it` was tried first
    but is `gated=manual` and this deployment's `HF_TOKEN` isn't approved for it.
- Live-verified on the RTX 4060 8 GiB dev box:
  - `qwen1.5-1.8b`: 73.6 tok/s, 5.71 GiB peak VRAM delta — comfortable headroom.
  - `qwen2.5-7b-awq`: after the fix, sizing rose from `utilization=0.74` to `0.849`,
    weight loading matched the estimate (5.29 GiB), KV cache went from negative to +0.63
    GiB, and it now loads and serves (4.7 tok/s — small KV cache limits batching, but no
    crash). `peak_vram_delta_gb` 7.44 GB, `vram_budget_exceeded: False`.
  - `minicpm5-1b`: after the `model_type` → model_path fix took effect, sizing rose from
    `utilization=0.474` to `0.824`, KV cache available rose from 1.52 GiB to 4.32 GiB.
    Re-benchmarked: 117.8 tok/s (unchanged within noise), peak VRAM delta 7.09 GB —
    `vram_budget_exceeded: True` (measured landed right at the 2.8 GiB margin's edge,
    correctly flagging this model as tight rather than falsely reporting comfortable
    headroom).
- Consequences: 427/427 unit tests pass. One existing real-config test
  (`test_real_config_has_a_quantized_awq_variant` in `test_model_registry.py`) asserted
  `quantized_gib < bf16_gib * 0.35` against the real (cached) HF config — the corrected
  ratio for this untied/large-vocab model is ~0.39, so the threshold was implicitly
  encoding the underestimate bug; raised to 0.45 with a comment explaining why. The
  `minicpm: 2.8` GiB constant is a fitted margin from one measurement, not a mechanism —
  if the real cause scales with `max_model_len` or concurrency rather than being fixed,
  it will need revisiting. No change to compare-mode splitting, `gpu_memory_utilization`
  auto logic's call sites, or the benchmark result schema.

### DEC-046
- Date: 2026-07-03
- Status: accepted
- Context: `make playground` with a real compare pair (qwen2.5-0.5b + qwen2.5-1.5b, both
  ungated, dense bf16, comfortably under 8 GiB combined) failed startup with "Models
  [qwen2.5-0.5b, qwen2.5-1.5b] cannot load sequentially on a 8 GiB GPU: Model
  qwen2.5-0.5b cannot fit in the remaining GPU memory for this pool. Need
  gpu_memory_utilization >= 0.116 but capped at -0.032." Two further, unrelated problems
  surfaced from the same report: (1) the failure banner's message was truncated
  mid-sentence ("...capped at" with nothing after), hiding the actionable
  "Load fewer models..." clause; (2) neither ctrl+c nor any other key could dismiss the
  failed loading screen or quit the app.
- Root causes:
  1. **Sequential VRAM cap double-reservation.** `validate_pool_fits()`'s own two
     pool-level aggregate checks (sum of footprints vs. `_POOL_GPU_HEADROOM`, both using
     an 8% total-VRAM safety margin) already confirmed this pair fits (needs ~7.29 GiB of
     a 7.36 GiB budget). But its third check, `_apply_sequential_vram_caps()`, separately
     reserved `_multi_engine_overhead_gib()` — `min(3.35, total_vram_gib * 0.42)`, a flat
     ~3.35 GiB (42% of an 8 GiB card) for any GPU ≳8 GiB regardless of pool size — *on
     top of* the next engine's full footprint (which already includes its own 0.8 GiB
     runtime/CUDA-graph overhead). 3.35 + 4.91 GiB (qwen2.5-1.5b's own footprint) alone
     exceeds 8 GiB, before the first engine claims anything, producing a negative allowed
     utilization. This constant was never validated against a real model pair on real
     hardware — the only unit test exercising this branch used mocked footprints small
     enough to never trigger it.
  2. **ctrl+c shadowed by Textual's own screen-level binding.** Textual 8.x's
     `Screen`/`ModalScreen` base classes claim plain `ctrl+c` for `copy_text` (and the
     base `App` class separately binds it to `action_help_quit`, a "press ctrl+q instead"
     notification) — both non-priority. `InferenceXApp`'s own `("ctrl+c", "quit", "Quit")`
     binding was also non-priority, so whenever any screen was pushed (e.g. `LoadingScreen`
     for a normal load, not just a failure), the screen-level binding won the focus-chain
     walk before the app's own binding was ever consulted. Confirmed via `App.run_test()`
     (Pilot): simulating `ctrl+c` left `app.is_running is True`.
  3. **Error truncation cut mid-sentence.** `playground/log_feed.py`'s
     `extract_error_summary()` had ~9 separate `[:200]` hard slices; the real
     `validate_pool_fits` message here is 255 characters, so the slice landed inside
     "...capped at -0.032. Load fewer models...", cutting off exactly the useful
     trailing sentence.
- Decision:
  1. Recalibrated `_multi_engine_overhead_gib()` from `min(3.35, total_vram_gib * 0.42)`
     to a flat `0.6` GiB constant. The old signature took `total_vram_gib` and its
     docstring claimed the reservation "scales with GPU size," but the `* 0.42` branch
     only wins below ~8 GiB — every real GPU (6/8/12/24 GiB tiers) got the identical flat
     3.35 GiB regardless, so the "scaling" was dead weight; simplified to what the
     function actually computed. Live-verified on an 8 GiB dev box: both qwen2.5-0.5b and
     qwen2.5-1.5b load and serve real chat completions together, ~0.74 GiB still free.
     Only verified for a 2-engine pool. Two existing unit
     tests that had encoded the old, unvalidated constant's behavior were corrected:
     `test_two_model_pool_uses_weight_aware_share` (expected utilization recalculated,
     0.175 → 0.2625) and `test_validate_pool_fits_rejects_dual_model_on_6gb` (swapped its
     qwen+tinyllama pair — which the corrected math now also allows — for the existing
     qwen+llama3-8b "genuinely too big" pair, preserving the test's intent of rejecting
     infeasible pools on a small GPU). Added
     `test_validate_pool_fits_allows_real_compare_pair_on_8gib`, a direct regression test
     with real Qwen2.5-0.5B/1.5B HF config values (mocked, no network dependency).
  2. `InferenceXApp.BINDINGS` and `ChatApp.BINDINGS` (`playground/app.py`,
     `playground/chat.py`) both mark their `ctrl+c` → quit binding `priority=True` —
     Textual checks priority bindings app-wide before the focus-chain walk, which is also
     why `ctrl+q` already worked reliably. Verified with a Pilot-driven regression test
     (`test_ctrl_c_quits_even_with_failed_loading_screen_on_top`) that fails without the
     fix (confirmed by reverting it and re-running) and passes with it.
  3. Added `_truncate_gracefully()` to `log_feed.py`: truncates at the last word boundary
     before the limit (raised 200 → 280) with an ellipsis, instead of an arbitrary
     mid-word/mid-sentence cut. All ~9 call sites in `extract_error_summary()` route
     through it. Added `test_extract_error_summary_does_not_cut_off_mid_sentence`.
- Consequences: 430/430 unit tests pass. No change to compare-mode splitting logic itself,
  the benchmark result schema, or `AdmissionController`. The `0.6` GiB multi-engine
  overhead figure, like DEC-045's `minicpm: 2.8` GiB, is empirically fitted from one
  verified hardware scenario, not derived from a documented mechanism — revisit if a
  3+-engine compare pool is added.

### DEC-047
- Date: 2026-08-04
- Status: accepted
- Title: Engine Boundary and backend plurality
- Context: Engine Boundary and backend plurality. Reviews of a proposed thick
  Execution Contract (`inference_x/execution/`) and two subsequent architecture
  reviews established a thin Engine Boundary direction, but that consensus was
  not yet recorded in-repo. Current state of the repository:
  - `engines/base.py`: `BaseEngine` is an ABC whose docstring asserts that adding
    a second engine must not require changes here; methods accept and return
    `schemas.chat` wire types; no capability methods are declared on the contract.
  - `api/deps.py`: the composition root constructs `VLLMEngine` directly
    (~line 108). There is no factory dispatch on `ModelEntry.engine`.
  - `engines/registry.py`: empty (0 bytes), despite being the natural home for
    construction.
  - `schemas/model.py`: `engine: Literal["vllm"]` structurally forbids naming a
    second backend in config.
  - `routing/admission.py`: discovers `count_prompt_tokens` and
    `kv_capacity_tokens` via `getattr`, with documented fail-open when absent —
    so a future second backend that omits those attributes would silently skip
    admission gates.
  - Governance conflict: `CONTRIBUTING.md` lists "Non-vLLM inference backends at
    this time" under what does not fit; `AGENTS.md` anti-scope forbids "multiple
    engine implementations" before the relevant phase; DEC-007 keeps `vllm` as a
    required dependency after optional-`vllm` broke `uv sync` / smoke; meanwhile
    `docs/REVIEW-2026-08-03-architecture.md` Phase D5 names an engine factory,
    optional-`vllm`, and a `llama-server` proxy as the consumer-hardware unlock.
  This decision is needed now so implementation and governance stop oscillating
  between "vLLM application forever" and "build a multi-backend framework early,"
  and so Engine Boundary hygiene cannot accidentally gate Phase B (AsyncLLM).
- Problem:
  1. Decorative engine boundary — docs claim a stable interface while the
     composition root hardcodes the concrete backend.
  2. Duck-typed capability discovery — admission and metrics reach past
     `BaseEngine` via `getattr` and fail open silently.
  3. Governance vs vision conflict — the product thesis implies backends become
     implementation details over the project's lifetime, but policy docs forbid
     non-vLLM work without stating plurality as a long-term architectural
     objective (distinct from a delivery commitment).
  4. Premature thick abstraction risk — a dual wire/execution type system under
     `inference_x/execution/` was proposed before a second concrete
     implementation exists, which would freeze internal DTOs ahead of AsyncLLM
     and ahead of any second backend.
- Decision:
  1. **Backend plurality (architectural principle).** Inference-X is
     architecturally designed to support multiple inference backends over its
     lifetime. At the time of this decision, vLLM remains the sole supported
     backend. No second backend is scheduled or committed for a specific
     release. Architectural changes should avoid unnecessarily coupling new
     runtime components to vLLM internals, but backend-neutral abstractions must
     not be introduced until justified by a second concrete implementation.
  2. **Thin Engine Boundary (accepted hygiene).**
     - Populate `engines/registry.py` with `create_engine(...)` as the sole
       construction path used by the app composition root (`api/deps.py`).
     - Declare durable capability methods on `BaseEngine` with default
       `None` / unsupported semantics; at minimum `count_prompt_tokens(...)`.
     - Admission (and metrics) call declared methods; no new silent `getattr`
       discovery for those capabilities.
     - Keep wire schemas (`schemas.chat`) as the engine I/O types until a
       second concrete backend forces extraction.
  3. **Capability durability split.**
     - **Durable:** tokenizer / prompt-token counting (backend-agnostic gate
       input).
     - **Provisional:** `kv_capacity_tokens` and in-process KV reservation
       semantics — may be rescoped or deleted under REVIEW Phase B4
       (admission rescope); must not be frozen as a cross-backend contract by
       this decision.
  4. **Admission fail policy (until B4).** Preserve current
     fail-open-when-unavailable behavior, but make degradation typed and
     observable (structured log when a gate is skipped because a capability is
     `None`). Do not silently tighten to fail-closed in this decision.
  5. **Sequencing (nonblocking).**
     - Phase A (truthful metrics / seed / CI) remains first for product truth.
     - Phase B (AsyncLLM) remains the next high-leverage runtime milestone and
       **must not** be gated on Engine Boundary hygiene.
     - Factory + durable capabilities are additive, parallelizable hygiene (or
       foldable into later D5) — never a blocking program milestone before
       AsyncLLM.
     - Phase C (manifest) and Phase D5 (second-backend vertical slice) remain
       as described in `docs/REVIEW-2026-08-03-architecture.md`; D5 requires a
       future ADR / change that authorizes a concrete second implementation.
  6. **DEC-007 relationship.** `vllm` remains a required dependency.
     Optional-`vllm` is **not** authorized by this decision. Revisit DEC-007
     only when a second backend vertical slice is accepted.
  7. **Backend Abstraction Principle.** Backend-neutral abstractions must not
     be introduced until they are justified by at least two concrete backend
     implementations. Until that point: build only what the repository requires
     today; avoid speculative backend-neutral packages or contracts; avoid
     knowingly hard-coding new vLLM-specific assumptions into architectural
     boundaries such as the composition root, the `BaseEngine` contract, or
     admission capability discovery. This principle applies to architectural
     boundaries, not to backend implementation details.
- Architectural principle: Inference-X is an inference runtime, not a vLLM
  application. The runtime owns architectural policy. Backends own inference
  execution. Architectural decisions should preserve the ability for additional
  backends to exist in the future without requiring current subsystems to be
  rewritten. However, backend-neutral abstractions must only be introduced when
  justified by multiple concrete implementations. This decision establishes
  backend plurality as a long-term architectural direction, not as an
  implementation commitment or delivery roadmap.
- Explicit non-decisions (deferred by this ADR):
  - `inference_x/execution/` package and dual wire↔execution DTOs
  - Backend-neutral Execution Contract freeze
  - Optional-extra `vllm`
  - `engines/backends/vllm/` relocation
  - AST import-boundary enforcement as a gate
  - Second backend implementation (llama.cpp / llama-server or otherwise)
  - AsyncLLM / `EngineDriver` deletion (Phase B)
  - Memory Manager, Planner, Scheduler, or Inference OS packaging
  - `vram_tiers.yaml` neutral / backend-native split
  - Widening `ModelEntry.engine` beyond validated registry keys without a
    second registered backend
  - Changing the public OpenAI HTTP surface
- Ownership:
  - **BaseEngine** — Owns: generation façade and declared capabilities. Knows:
    request/response types currently in use; capability `None` semantics. Must
    never know: HTTP/FastAPI; other backends; admission policy; model selection.
  - **engines/registry** — Owns: instantiation dispatch by engine type. Knows:
    registered backend constructors; model config dicts needed to construct.
    Must never know: admission; routing policy; OpenAI wire minting.
  - **Admission (`routing/`)** — Owns: whether a request may run; gate policy.
    Knows: declared engine capabilities; registry model limits; priority
    semantics. Must never know: vLLM internals; CUDA; Driver/`step()` loop.
  - **API / services** — Owns: wire validation; composition-root wiring; OpenAI
    response minting. Knows: settings, pool, router, admission. Must never
    know: backend-native serving loops.
  - **Concrete backend (`VLLMEngine` + driver)** — Owns: inference execution for
    vLLM. Knows: vLLM APIs, local runtime state, translating current request
    types to native calls. Must never know: HTTP schemas as *owned* long-term
    vocabulary (today borrowed); other backends; global scheduling policy.
- Alternatives considered:
  - **Keep as-is.** Rejected: leaves the decorative boundary and the governance
    contradiction in place; a second backend later becomes an architectural
    migration rather than an implementation project.
  - **Thick Execution Contract** (`inference_x/execution/`, dual wire/execution
    types, optional-`vllm`, `backends/vllm/` move in one milestone). Rejected:
    premature abstraction before a second implementation; high churn against
    Phase B (AsyncLLM); contradicts DEC-007 and current anti-scope if bundled.
  - **Thin Engine Boundary + plurality without timeline (this decision).**
    Accepted.
  - **vLLM-only for ≥12 months (with or without a research note).** Rejected:
    reframes the project identity as a vLLM application and biases future
    subsystems (planner, scheduler, memory) toward vLLM-shaped boundaries;
    forces reopening this ADR when plurality becomes unavoidable.
- Consequences:
  - **Positive:** composition root becomes architecturally honest; engine
    capabilities become explicit; backend plurality is established as a
    long-term architectural direction without requiring premature abstractions;
    AsyncLLM remains unblocked; DEC-007 remains unchanged.
  - **Negative:** engines still import wire schemas (accepted smell until a
    second backend forces extraction); provisional KV capability remains until
    B4; contributors must distinguish "plurality as direction" from "build
    abstractions now."
  - **Deferred work:** sync `CONTRIBUTING.md`, `AGENTS.md`, and a short note in
    `docs/ARCHITECTURE.md` once this decision is accepted; OpenSpec change when
    implementing factory / capabilities; concrete second backend only via a
    future accepted change.
  - **Technical debt accepted:** wire-as-engine-API; import-time CUDA/vLLM
    patches in `api/main.py` (known coupling; not cured here).
  - **Future opportunities:** when a second backend lands, extract execution
    types under the Backend Abstraction Principle; revisit optional-`vllm`
    (DEC-007) only with that vertical slice.
- Risks:
  - **Architectural:** plurality language misread as permission to introduce
    `execution/` or other backend-neutral packages early — mitigated by Explicit
    non-decisions and the Backend Abstraction Principle.
  - **Migration:** factory and durable capabilities touch `deps`, admission, and
    tests — low risk if scoped as hygiene.
  - **Governance:** accepting this ADR without updating `CONTRIBUTING.md` /
    `AGENTS.md` leaves the prior conflict in force — listed under Exit criteria.
  - **Future backend support:** promoting provisional KV admission onto a stable
    cross-backend contract before Phase B4 — mitigated by the durability split.
- Sequencing relative to REVIEW phases:
  - Phase A → Phase B (AsyncLLM) → B4 admission rescope → Phase C (manifest)
    remain the product/runtime spine.
  - DEC-047 factory + durable capabilities are nonblocking relative to that
    spine (dashed dependency only).
  - A future ADR authorizing a concrete second backend precedes D5; execution-
    type extraction, if needed, follows a second implementation (rule of two).
  - Provisional KV capability policy is decided with or after B4, not frozen
    here.
- Governance (required once this decision is accepted):
  - `docs/DECISIONS.md` — this entry; status moves from `proposed` to
    `accepted`.
  - `CONTRIBUTING.md` — replace "Non-vLLM inference backends at this time" with
    language that: vLLM is the sole supported backend today; plurality is a
    long-term architectural objective; no second backend is scheduled; do not
    add backends without a dedicated accepted change.
  - `AGENTS.md` — align anti-scope: forbid *implementing* multiple engines or
    `inference_x/execution/` until justified by a second concrete
    implementation; allow thin factory / durable capabilities; state the
    plurality objective.
  - `docs/ARCHITECTURE.md` — short engine-boundary note: factory ownership; no
    `execution/` package yet; hygiene must not gate AsyncLLM.
  - OpenSpec — required only when implementing factory / capabilities code; not
    required to accept this ADR text.
- Exit criteria (architectural; DEC-047 is fully implemented when all hold):
  1. This entry is `accepted` in `docs/DECISIONS.md`.
  2. `CONTRIBUTING.md` and `AGENTS.md` no longer contradict backend plurality
     versus anti-scope without explanation.
  3. App engine construction goes through `engines/registry` factory (no direct
     `VLLMEngine(` in `api/deps.py`).
  4. Admission uses declared durable capability methods; degradation when
     unsupported is observable; no new `getattr` discovery for those methods.
  5. No `inference_x/execution/` package exists as a deliverable of this
     decision.
  6. `vllm` remains a required dependency (DEC-007 intact).
  7. Phase B / AsyncLLM is not blocked on items 3–4.
- Supersession: This decision remains in force until explicitly superseded by a
  future DEC. In particular, any decision introducing a second concrete
  inference backend, backend-neutral execution packages, optional backend
  dependency layouts, or changes to the Engine Boundary ownership model must
  explicitly reference and supersede DEC-047 where appropriate.
  Acceptance of this ADR establishes repository policy for the Engine Boundary;
  it does not by itself authorize factory/capability implementation, AsyncLLM
  work, or a second backend — those require their own accepted changes.

### DEC-048
- Date: 2026-08-04
- Status: accepted
- Title: Lint rule set and type-check baseline for the CI gate (OS-1)
- Context: OS-1
  (`openspec/changes/2026-08-04-add-ci-and-static-analysis-gate/`) adds the
  repository's first merge-blocking gate: `pytest tests/unit`, `ruff check .`,
  and `mypy src/` on every pull request. Measured against the tree at
  `develop` before any change:
  - `ruff check .` with the tool's own default rule selection reported **232
    findings** across 26 rules.
  - `mypy src/` reported **27 errors in 9 files** (51 source files checked).

  OS-1 forbids editing anything under `src/` or `tests/`. That constraint is
  not incidental: five Phase A units follow OS-1 and several will be in flight
  simultaneously over a shared file set
  (`docs/PHASE-A-EXECUTION-PLAN.md` §7.3), so a tree-wide cleanup landing first
  would collide with all of them and make every subsequent Phase A diff
  unreviewable. Three options existed — weaken the gate until it asserts
  nothing, edit the code, or record the known-failing surfaces explicitly and
  gate everything else.
- Decision:
  1. **Lint rule set.** Select `E4`, `E7`, `E9`, `F` — the set covering broken
     code (syntax errors, undefined names, redefinitions) rather than style.
     Three rules are ignored, each for a stated reason:
     - `E402` (module import not at top of file) — **deliberate design, not
       debt.** `api/main.py` applies the vLLM platform patch before importing
       any vLLM-touching module, and `scripts/` set `sys.path` before importing
       the package. Enabling it would flag correct code.
     - `F401` (unused import, 17 occurrences) and `F841` (unused variable, 1) —
       **deferred, not endorsed.** Clearing either requires source edits, which
       OS-1 forbids.
     The remaining ~200 default findings (`BLE001`, `SIM117`, `I001`, `S110`,
     `UP*`, `RUF*` and others) are outside the selection entirely. Widening the
     selection is a separate change and must not ride along with a Phase A unit.
  2. **Third-party stubs are not a code suppression.** `yaml` (4 errors) and
     `pynvml` (1) ship no type information. These are handled with
     `ignore_missing_imports` scoped to those two packages. This says nothing
     about `inference_x`'s own types and is not part of the baseline. It
     resolved 5 of the 27 errors without suppressing a single project module.
  3. **Type-check baseline — four modules, 22 errors.** The remaining errors
     are confined to modules that sit against untyped surfaces or are already
     scheduled for replacement:

     | Module | Errors | Why |
     |---|---|---|
     | `engines/vllm_engine` | 10 | vLLM's `LLM` is untyped; mypy resolves it as `LLM?` and rejects attribute access. |
     | `engines/driver` | 7 | `Optional` narrowing on `Queue`/`Future`. Deleted by Phase B1 (AsyncLLM). |
     | `services/chat_service` | 3 | `generate_stream` is declared `async def -> AsyncGenerator[str, None]`, which mypy reads as a coroutine. |
     | `api/deps` | 2 | Composition-root argument types. |

     The baseline is an **enumerated, closed list**. No repository-wide
     suppression and no wildcard, so a module added later is type-checked by
     default — verified during OS-1 validation by adding a scratch module with
     a deliberate type error and confirming the gate went red.
  4. **`api/main.py` is deliberately excluded from the baseline.** Its only
     error was the `yaml` stub, resolved by decision 2. Baselining it would
     have suppressed a module that needs no suppression, silently exempting
     future errors there. Over-suppression is a defect, not a safe default.
  5. **The baseline does not grow during Phase A.** Adding a module to it in
     OS-2 through OS-6 is a review finding — a signal the unit is touching more
     than its scope allows (`docs/PHASE-A-EXECUTION-PLAN.md` §7.4) — not a
     routine edit.
- Consequences:
  - CI landed with **zero changes under `src/` or `tests/`**. 430/430 unit
    tests pass; `ruff check .` and `mypy src/` both report clean.
  - The gate is real but bounded. It catches broken code, undefined names, and
    type errors in the 47 non-baselined modules. It does **not** assert style
    consistency, import ordering, or type correctness inside the four
    baselined modules.
  - A clean `mypy src/` run is not a Phase A goal and should not be pursued
    inside a Phase A unit.
  - `services/chat_service`'s three errors are the same `generate_stream`
    typing defect the Phase A plan identified independently
    (`docs/PHASE-A-EXECUTION-PLAN.md` §1.1). The type checker found it without
    being told to look. OS-2 widens that contract and should shrink or remove
    this baseline entry as a side effect.
  - `engines/driver`'s entry is expected to disappear with the module itself in
    Phase B1.
- Supersession: baseline reduction and lint-selection widening each require
  their own change. Neither is authorized by this decision.

### DEC-049
- Date: 2026-08-04
- Status: accepted
- Title: Widen BaseEngine.generate_stream for truthful streaming usage (OS-2)
- Context: Phase A OS-2 (`docs/PHASE-A-EXECUTION-PLAN.md` §1.1, §3.4, §4).
  `BaseEngine.generate_stream` is typed `AsyncGenerator[str, None]`. A bare
  string cannot carry `usage` or `finish_reason`. Observability middleware and
  the benchmark runner therefore approximate streamed completion tokens with
  whitespace word counts (`_count_sse_delta_tokens`, runner stream loop).
  Non-streaming `generate()` already returns engine-accounted
  `ChatCompletionUsage`. DEC-023 enabled SSE with the explicit consequence that
  streamed token counts would remain estimates until a usage event existed.
  DEC-047 forbids introducing `inference_x/execution/` or a second type system;
  engines today speak `schemas.chat` wire types. DEC-048's mypy baseline already
  flags `services/chat_service` for the same `generate_stream` typing defect.
- Problem: Truthful Phase A metrics require streamed completion tokens and
  finish reasons to originate from the engine. That is impossible while
  `generate_stream` yields only `str`. Leaving the contract unchanged forces
  continued approximation; inventing a backend-neutral chunk package outside
  `schemas/` violates DEC-047.
- Decision:
  1. **Add a streaming chunk model to `schemas/chat.py`** (name to be chosen at
     implementation, e.g. `ChatStreamChunk`) carrying:
     - `content: str` (delta text; empty on a terminal-only chunk)
     - `finish_reason: Literal["stop", "length", "error"] | None`
     - `usage: ChatCompletionUsage | None`
     Content deltas set `content` and leave `finish_reason`/`usage` null. The
     terminal engine event sets `finish_reason` and, when available, `usage`.
  2. **Widen `BaseEngine.generate_stream`** to
     `AsyncGenerator[<chunk model>, None]`. Update `VLLMEngine`, all test stubs,
     and `ChatService.stream_response` accordingly.
  3. **Surface terminal metadata from `EngineDriver`** on the stream channel
     (today the queue is `str | BaseException | None` and drops
     `RequestOutput` fields on finish). Without this, the vLLM path cannot
     yield real `usage`/`finish_reason`. This is an implementation detail of the
     current offline-`LLM` stack; Phase B1 deletes the driver and must preserve
     the `BaseEngine` chunk contract.
  4. **Wire `stream_options.include_usage`** on `ChatCompletionRequest` (OpenAI-
     compatible, optional, default off or follow OpenAI defaults as implemented).
     When usage is requested (or as required to satisfy Phase A metric truth for
     `/v1/metrics` and benchmarks — see Consequences), `ChatService` emits a
     terminal SSE chunk before `data: [DONE]` carrying `usage` and places
     `finish_reason` on the last content chunk per OpenAI streaming conventions.
  5. **Delete `_count_sse_delta_tokens`** and repoint SSE observability to the
     usage chunk. Benchmark runner stream measurement reads
     `usage.completion_tokens` instead of `len(content.split())`.
  6. **Metric discontinuity.** Figures produced before this change (in-process
     `/v1/metrics`, stored benchmark `tokens_per_sec` derived from word counts,
     and published claims in `article-final.md`) are **not comparable** to
     figures produced after. Record that supersession here; add a correction
     note to `article-final.md` (gitignored private writing — still required).
  7. **Docstring honesty.** Update `BaseEngine`'s claim that a second engine
     needs no changes here: this widening is exactly the change DEC-047
     problem statement §1 anticipated as overdue honesty about a decorative
     boundary.
- Alternatives considered:
  - **Keep yielding `str`; estimate tokens forever.** Rejected: contradicts
    Phase A objective and leaves DEC-023's known gap permanent.
  - **Yield `tuple[str, Usage | None]`.** Conformant with DEC-047 but worse:
    untyped positional contract; Phase B3 timings would widen it again; mypy
    cannot usefully check it. Rejected.
  - **Introduce a neutral chunk type outside `schemas/`.** Forbidden by DEC-047
    (`inference_x/execution/` prohibition in all but name). Rejected.
  - **Add chunk model in `schemas/chat.py` and widen `generate_stream` (this
    decision).** Accepted.
- Consequences:
  - **Positive:** streamed and non-streamed `usage.completion_tokens` can agree;
    `/v1/metrics` and benchmarks stop lying; DEC-048 baseline entry for
    `chat_service` should shrink or clear; OpenAI clients that understand
    terminal usage chunks gain real counts.
  - **Negative:** every `BaseEngine` stub and the driver stream channel change;
    historical metrics are discontinuous (must be labeled, not silently mixed);
    `article-final.md` correction is irreversible once published.
  - **Out of scope (other OpenSpecs):** `seed` (OS-3); `warnings`/`resolved`/
    `strict`/`count_prompt_tokens` (OS-4); suite hash (OS-5); advisor rename/
    quant_score (OS-6); `engines/registry` factory (deferred).
- DEC-047 compliance:
  - Chunk type remains in `schemas.chat` — current borrowed vocabulary.
  - No `inference_x/execution/`, no dual wire↔execution layer, no second backend.
  - Engine Boundary hygiene still must not gate AsyncLLM; this widening is the
    single Phase A Engine Boundary change authorized for OS-2 (§1.1).
  - Capability declaration (`count_prompt_tokens`) remains OS-4.
- Compatibility analysis:
  - `playground/streaming.py` returns on `data: [DONE]` and skips lines with no
    token — additive terminal usage chunk is ignored safely.
  - `benchmarks/runner.py` skips chunks without `delta.content` today; OS-2
    updates it to *prefer* `usage` when present (owned by OS-2).
  - Existing clients that ignore unknown chunk fields remain valid; request field
    `stream_options` is additive/optional.
- Migration notes:
  - Land as one PR with the ADR accepted (or accept ADR in the same PR).
  - Do not grow the DEC-048 mypy baseline; prefer removing `chat_service` from it
    when the contract type-checks.
  - Reference the `engines/registry` Phase A/B seam as an open item from this ADR
    (plan §5.4) without implementing the factory.
  - If OS-2 is reverted, keep this ADR's supersession statement for prior
    approximate figures — those numbers were always wrong (plan §8.2).
  - **`derive_terminal_metadata` relocation constraint (OWN-B5).**
    `derive_terminal_metadata` currently lives in `engines/driver.py`, called
    from both the non-streaming path (`driver.py`) and the streaming path
    (`vllm_engine.py`) — this is the single helper DEC-050 §3 names as the
    reason streamed and non-streamed usage cannot drift apart. Roadmap B1
    removes `EngineDriver`. The helper MUST be relocated, not reimplemented,
    before or during B1: it MUST remain the single source of truth for
    deriving terminal metadata on both the streamed and non-streamed paths,
    and B1 MUST NOT duplicate or fork this translation logic into a
    second implementation. This note does not prescribe the destination
    module — that is a B1 decision.
- Supersession: remains in force until a future DEC changes the streaming
  engine contract (e.g. Phase B AsyncLLM adaptation must preserve or explicitly
  replace this chunk model on `BaseEngine`).

### DEC-050
- Date: 2026-08-04
- Status: accepted
- Title: Streamed token counts before OS-2 are superseded and incomparable
- Context: Until OS-2
  (`openspec/changes/2026-08-04-add-truthful-token-accounting/`), two sites in
  `src/` derived completion-token counts by counting whitespace-delimited words
  in generated text:
  - `observability/middleware.py::_count_sse_delta_tokens`, feeding
    `/v1/metrics` (`completion_tokens`, `total_tokens`, `tokens_per_sec`);
  - the `benchmarks/runner.py` streaming loop, feeding `PromptResult`
    (`tokens_generated`, `tokens_per_sec`) and, through it, the benchmark
    advisor's model recommendations.
  DEC-023 accepted this knowingly, because streamed responses carried no usage
  event. DEC-049 removed that constraint. Words are not tokens; the error is
  model- and tokenizer-dependent and always understates the true count, because
  a word is one or more tokens and never fewer.
- Problem: correcting a metric silently is the same category of dishonesty as
  reporting it wrongly. Anyone comparing a figure recorded before this change
  with one recorded after would be comparing two different quantities that
  share a name. Phase A exists to make reported numbers true; it must not
  create an undocumented discontinuity while doing so.
- Decision:
  1. **Every streamed completion-token count produced before OS-2 is
     superseded.** This includes `/v1/metrics` token fields and rates,
     every stored benchmark result's `tokens_generated` and `tokens_per_sec`,
     and any throughput figure published from them.
  2. **Pre-OS-2 and post-OS-2 figures are not comparable in either
     direction.** They are not off by a constant factor and cannot be
     reconciled by rescaling — the ratio depends on the tokenizer and on the
     text. Do not mix them in a series, a chart, or a claim.
  3. **Non-streamed `usage` is unaffected.** `generate()` always reported
     engine-accounted counts. OS-2 makes the streamed path agree with the
     non-streamed one, not the reverse, and both now derive from the single
     `derive_terminal_metadata` helper so they cannot drift apart again.
  4. **Absence replaces estimation.** When a streamed request does not set
     `stream_options.include_usage`, no usage event reaches the middleware and
     **no token figure is recorded at all** — not zero, not an estimate. A
     missing number is honest; a wrong one is not. This is a deliberate loss of
     metric coverage for default streaming clients, accepted in exchange for
     correctness.
  5. **`article-final.md` correction remains an author obligation.** The file
     is gitignored under an explicit "private writing" policy, so it is outside
     the repository and outside any acceptance criterion. `README.md` was
     checked and publishes no throughput figures, so no in-repository published
     number requires correction.
- Consequences:
  - Historical benchmark JSON under `benchmarks/results/` is machine-specific
    and gitignored; it is not migrated. Results produced before this change
    should be regenerated rather than compared. Filtering or partitioning a
    results directory into "pre-OS-2" vs "post-OS-2" is not viable: files carry
    no durable marker of which counting method produced `tokens_generated` /
    `tokens_per_sec`, timestamps alone cannot recover that, and mixing the two
    quantities in one series would reintroduce the discontinuity this decision
    forbids. Regeneration is therefore the only safe path.
  - `/v1/metrics` `avg_tokens_per_sec` degrades to `None` when no request in
    the window carried a usage event. `metrics_service` already filters
    `tokens_per_sec is not None`, so this is a graceful absence rather than a
    break.
  - The benchmark runner sets `stream_options.include_usage` on its own
    requests, so benchmark figures stay populated and are now true.
  - If OS-2 is reverted, this supersession still stands. The old numbers were
    always wrong; reverting the fix does not make them right
    (`docs/PHASE-A-EXECUTION-PLAN.md` §8.2).
- Supersession: none. This is a statement of fact about historical data and
  does not expire.

### DEC-051
- Date: 2026-08-04
- Status: accepted
- Title: Seed support / Deterministic Generation Contract (OS-3)
- Context: Phase A OS-3 (`docs/PHASE-A-EXECUTION-PLAN.md` §4; review task A3).
  `ChatCompletionRequest` had no `seed` field, so clients that pinned a seed
  (notably Varex) experienced silent Pydantic drop — worse than unsupported.
  `VLLMEngine._sampling_params` built `SamplingParams` without seed. OS-2
  (DEC-049) already spent Phase A's Engine Boundary change; OS-3 must not
  widen `BaseEngine`. Ownership: *The client owns requesting determinism; the
  backend owns honouring it; the runtime must not invent it.*
- Problem: Without a first-class `seed` that reaches the live sampler,
  reproducibility harnesses cannot drive the server honestly. Overclaiming
  end-to-end determinism would be a separate defect (batch composition is
  Phase C3).
- Decision — Deterministic Generation Contract:
  - **G1 Acceptance.** `seed: Optional[int] = None` on `ChatCompletionRequest`
    (appended after `stream_options`). Not silently dropped.
  - **G2 Unchanged forward.** When `seed is not None` and the live vLLM engine
    builds `SamplingParams`, pass that integer exactly as received. No rewrite,
    clamp, or runtime normalization (including `-1`).
  - **G3 Omission equivalence.** When `seed is None`, omit the `seed` key —
    pre-OS-3 sampling construction for all other parameters.
  - **G4 Path parity.** Streaming and non-streaming use the same
    `_sampling_params` builder.
  - **G5 Honesty.** Docs say seed is *honoured* (reaches the sampler); never
    that the server is deterministic or runs are reproducible end-to-end.
  - **Intentionally non-guaranteed (N1–N8):** concurrent/batch identity;
    cross-hardware/version identity; CUDA-graph/JIT/prefix-cache/spec-decode
    identity; replay/manifests; runtime-invented determinism; default
    benchmark determinism; response seed echo (OS-4); backend sentinel
    interpretation (e.g. what vLLM does with `-1`).
- Alternatives considered:
  - **Keep silent drop.** Rejected: Varex failure mode.
  - **Normalize `-1` → omit in Inference-X.** Rejected: runtime must not
    invent backend semantics (N8).
  - **Echo effective seed now.** Rejected: OS-4 owns `resolved`.
  - **`deterministic: true` / `VLLM_BATCH_INVARIANT`.** Rejected: Phase C3.
- Consequences:
  - Positive: pinned seeds reach the sampler; silent-drop failure mode
    superseded.
  - Negative: clients may over-read identity tests; docs must stay honest (G5).
  - Out of scope: OS-4 echo/`warnings`/`strict`/`count_prompt_tokens`; OS-5
    suite hash; OS-6 advisor; Phase B AsyncLLM; Phase C replay/oracle.
- DEC-047 compliance: wire field + concrete engine wiring only; no
  `execution/`; no Engine Boundary change; no second backend.
- Supersession: remains in force until a future DEC changes the sampling-input
  contract. Phase C may *add* guarantees without rewriting G1–G5.

### DEC-052
- Date: 2026-08-04
- Status: accepted
- Title: `strict` may only convert substitution into rejection
- Context: Phase A OS-4 (`docs/PHASE-A-EXECUTION-PLAN.md` §4, §9 C.7). Request
  flag `strict: bool = false` must reject where the default clamps, without
  becoming a general “do not batch / fail if cache cold / reject on revision
  drift” runtime-policy switch for Phase B/C.
- Problem: Loose wording (“strict changes response policy, never runtime
  policy”) is too weak to enforce — rejection changes whether the request runs
  at all. Without a sharp invariant, later phases will overload `strict` with
  unrelated runtime policy.
- Decision:
  1. **`strict` may only convert a substitution into a rejection. It may never
     change the substitution itself.** Under `strict`, outcomes partition into
     `{rejected, executed exactly as asked}`. It never produces
     `{executed differently}`.
  2. **One predicate, two outcomes.** The condition that emits a default-mode
     `ResponseWarning` with `type: "substituted"` is exactly the condition that
     raises under `strict`. Verified by tests that enumerate both sets.
  3. **Acceptance check:** for any request accepted under both modes, given an
     identical seed, the completion content is byte-identical.
  4. Phase B/C needs (batch isolation, cold-cache fail, revision pin, etc.) get
     **their own fields**; they must not be folded into `strict`.
- Alternatives considered:
  - **Loose “response policy only” wording.** Rejected: not CI-checkable;
    admits overload.
  - **Let `strict` grow into a policy bundle.** Rejected: destroys the single
    meaning that makes C.7 falsifiable.
- Consequences:
  - Positive: `strict` stays one thing; Phase B/C cannot smuggle runtime policy
    through it without a new DEC.
  - Negative: callers wanting broader “strict serving” need additional flags
    later.
- Compatibility: default / absent `strict` preserves today’s clamp-and-continue
  behaviour aside from additive Effective Request surfaces (`resolved` /
  `warnings`).
- Supersession: remains in force until a future DEC explicitly widens `strict`.

### DEC-053
- Date: 2026-08-04
- Status: accepted
- Title: Pre-generation and post-generation metadata lifecycle (OS-4)
- Context: Phase A OS-4 (`docs/PHASE-A-EXECUTION-PLAN.md` §4, §9 C.6;
  `docs/REVIEW-2026-08-04-os4-architecture-freeze.md`). OS-4 surfaces the
  Effective Request (`resolved`) and typed degradation (`warnings`) on the
  streamed path as well as the non-streamed one, which adds an event to the SSE
  order DEC-049 fixed. Phase B3 will add per-request timings and Phase C will
  add replay metadata to the same stream.
- Problem: Without a rule saying *which* metadata may be emitted *when*, each
  later phase re-argues placement, and a fact emitted in the wrong phase is
  either lost on the error path or physically un-emittable in its assigned slot.
  Two loose framings were considered and rejected below.
- Decision:
  1. **Pre-generation phase.** A streamed response has a pre-generation metadata
     phase that ends when the first token is sampled. An event MAY be emitted in
     that phase **if and only if** every field it carries is fully determined and
     immutable at the moment the effective request is finalized. A fact that can
     change during or after generation MUST NOT be emitted in this phase.
  2. **Cardinality.** The pre-generation phase contains **exactly one** event.
     Adding a second is a modification of the platform requirement, not an
     extension of it.
  3. **Post-generation phase.** Events after the terminal event carry only facts
     about what generation produced. The usage event is the last event before
     `data: [DONE]`. **No event is emitted after the usage event** — OpenAI
     documents the usage chunk as the one streamed before `[DONE]` and clients
     use it as an end sentinel.
  4. The rule is streaming-only. `ChatCompletionResponse` has no phases and
     carries `resolved` and `warnings` in one body regardless.
- Alternatives considered:
  - **Anchor on "determined before the first token is sampled."** Rejected: it
    is near-tautological (an emitted event is trivially determined before
    emission), and it leaks. Queue/wait time is determined when the engine
    begins processing — before the first token but *after* the prologue is
    already on the wire — so that anchor classifies as pre-generation a fact
    that cannot occupy the slot, splitting the Phase B3 timing class in two.
  - **"One prologue event" as the invariant.** Rejected as the *invariant*:
    cardinality constrains how many events there are and says nothing about what
    may go in them. Kept as the current instantiation (decision 2) instead.
  - **"One or more" cardinality in the normative wire text.** Rejected: it
    weakens the exact-sequence assertion DEC-049/OS-2 explicitly protected, for
    a second event that does not exist.
  - **Carry `resolved` as a trailer before `[DONE]`.** Rejected: the facts are
    determined at admission, a trailer is lost on the timeout path, the client
    cannot act on a clamp until the generation it no longer wants has finished,
    and it collides with the slot Phase B3 needs.
- Consequences:
  - **TTFT is measured from the first content event**, not the first SSE chunk.
    `observability/middleware.py` is changed accordingly by OS-4;
    `benchmarks/runner.py` already did this. Phase B3 inherits the definition.
  - **`resolved ⊂ prologue`.** Two independent rules: *derivability* governs
    what is in `resolved` (a field appears iff it exists on
    `ChatCompletionRequest`, minus `messages` and the transport/policy controls);
    *determinacy* governs what is in the prologue. The prologue is the superset.
    A later phase must not conclude "prologue means `resolved`" and widen
    `resolved` to fit something that is merely prologue-eligible — model
    revision and any replay handle are the concrete cases.
  - **Trace/request ids, if introduced, are minted at admission**, making them
    pre-generation and therefore available on the error path — where DEC-049
    emits an error event and `[DONE]` with no terminal and no usage event, and
    where correlation matters most.
  - **Warning-code registry.** The closed set OS-4 introduces:

    | `type` | `code` | `field` |
    |---|---|---|
    | `substituted` | `max_tokens_clamped_to_context` | `max_tokens` |
    | `substituted` | `max_tokens_clamped_to_kv_budget` | `max_tokens` |
    | `degraded` | `prompt_tokens_estimated` | `messages` |
    | `degraded` | `kv_gate_skipped` | — |
    | `degraded` | `sequence_gate_skipped` | — |

    Adding a code is a spec change. All OS-4 warnings are pre-generation; a
    post-generation warning, if one ever exists, attaches to the usage event
    rather than being retrofitted into the prologue.
  - Negative: a first-party consumer that timestamped first-chunk-received must
    be updated. Exactly one existed (`observability/middleware.py`).
- Compatibility: additive. The pre-generation event carries `choices: []` and no
  top-level `usage`, so `middleware._extract_sse_usage`, `benchmarks/runner.py`
  and `playground/streaming.py` all skip it unmodified. DEC-049's event order is
  extended at the head, never after the usage event.
- Supersession: does not supersede DEC-049 — it constrains what may be added to
  the order DEC-049 fixed. Remains in force until a future DEC changes the phase
  boundary or the cardinality.

### DEC-054
- Date: 2026-08-04
- Status: accepted
- Title: Benchmark suite identity is a pinned content hash (OS-5)
- Context: Phase A OS-5 (`docs/PHASE-A-EXECUTION-PLAN.md` §5;
  `docs/REVIEW-2026-08-04-os5-os6-finalization.md` §0, §2). `suite_version` was
  documented as a content hash of the prompt suite but nothing computed or
  verified it; `benchmarks/runner.py:_load_suite` read the stored literal
  verbatim. Direct computation confirmed the shipped literal already equals the
  SHA-256 of the prompt list under a compact-JSON canonicalization.
- Problem: Without a pinned canonical form and a verifier, the reproducibility
  primitive is decorative — an edited prompt silently keeps the same
  `suite_version`. The inverse risk is equally real: an implementer picks a
  *different* canonicalization and manufactures a break that need not happen.
- Decision:
  1. **Canonical form (pinned):**
     `sha256(json.dumps(prompts, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()`,
     hashing `data["prompts"]` only. The `suite_version` key is excluded because
     it lives inside the file it identifies.
  2. **Hashed object is the parsed in-memory prompt collection.** File encoding,
     whitespace, indentation, line endings, JSON object key order, and
     serialization formatting are excluded from identity. Only prompt ordering
     and prompt values participate.
  3. **Stored bare hex** (no `sha256:` algorithm tag). The 25 existing results and
     the shipped suite use bare hex.
  4. **One shared canonicalizer** (`benchmarks/suite_identity.py`) owns identity.
     Load-time verification and the `make suite-version` regeneration command both
     call it. No duplicate hashing logic.
  5. **Fail loud at load.** A missing `suite_version` key or a computed/stored
     mismatch raises an actionable error naming `make suite-version` — not a bare
     `KeyError`, not a silent accept.
  6. **Characterization proof:** the computed digest of the shipped suite equals
     `b47066414716cf4a0970adc790384f5173bd488cf80da47d28936b3d2ce5cfa4`.
- Alternatives considered:
  - **Hand-bumped semver (`v1`, `v2`).** Rejected: reproduces the current defect —
    a self-declared string nothing verifies.
  - **Git blob / commit hash.** Rejected: VCS-coupled, breaks outside a checkout,
    over-invalidates on whitespace-only edits.
  - **Signed manifest.** Rejected: wrong threat model; Phase A has no adversary,
    the problem is accidental drift.
  - **Algorithm-tagged storage now.** Deferred: bare hex retained; tagging is a
    future forward-provision, not required to verify today.
- Consequences:
  - **This is verification, not migration.** The plan's migration premise (§457,
    §808 item 2, §256, §889 — regenerated hash invalidating stored results) is
    **void**. All 25 stored results remain comparable; nothing is regenerated;
    there is no rollback hazard.
  - The earlier reviews were correct that no `sha256` call existed; only the
    inference "therefore the literal is arbitrary" is disproved.
  - **Fail-loud posture does not contradict DEC-047 §4.** DEC-047's fail-open
    applies to the serving admission path with a live client; the benchmark
    harness is first-party tooling where a silent wrong number is worse than a
    stopped run.
  - Limitation (recorded, not fixed): byte-identity of canonical JSON; an NFC vs
    NFD spelling of the same prompt hashes differently. The suite is ASCII; no
    Unicode normalization is added.
- Compatibility: additive. `benchmarks/prompts/standard.json` and
  `benchmarks/results/*.json` are read-only under this change.
- Supersession: supersedes the plan's OS-5 migration framing. Remains in force
  until a future DEC changes the canonical form or the stored digest format.

### DEC-055
- Date: 2026-08-04
- Status: accepted
- Title: `suite_version` is a necessary but not sufficient comparability key (OS-5)
- Context: Phase A OS-5 (`docs/REVIEW-2026-08-04-os5-os6-finalization.md` §1.1,
  §4). Before this change `storage.py:latest_per_model()` selected the newest
  result per model regardless of suite version, and `advisor.py:rank()` ranked
  whatever it was handed — so a verified `suite_version` had no consumer.
- Problem: Verification without a consumer is decorative. Two results are only
  comparable if their benchmark *input* matches; ranking across suite versions
  produces a silent wrong answer.
- Decision:
  1. **Option A — filter in storage.** `storage.py` filters results by the expected
     `suite_version` **before** latest-per-model selection. Mixed-suite sets never
     reach the advisor. `advisor.py` is not touched by OS-5 (it is OS-6's file;
     touching it would break the plan's OS-5 ∥ OS-2/OS-3 parallelism).
  2. **Empty ≠ mismatch.** "No benchmark results exist" and "results exist but none
     match the requested `suite_version`" are distinct, distinguishable outcomes.
  3. **Immutable and append-only.** Incomparable results are excluded by the
     consumer, never deleted or rewritten. Rollback never requires rewriting
     historical benchmark files.
  4. **Necessary, not sufficient.** `suite_version` pins the benchmark *input*, not
     the runtime that consumed it. It is one necessary comparability gate alongside
     hardware (DEC-036) and does not by itself prove full provenance.
- Alternatives considered:
  - **Close it in `advisor.py`.** Rejected: creates OS-5/OS-6 shared-file
    contention the plan claims does not exist; `latest_per_model()` is already the
    function that decides which results reach the advisor.
  - **Defer with reason (closure b).** Rejected in favour of closing it now — the
    filter is small and the 25 stored results share one suite version, so it is a
    no-op on existing data.
- Consequences:
  - Positive: a verified `suite_version` becomes a *used* key; the advisor cannot
    rank across suites.
  - Negative: comparability remains broader than this key (concurrency, runtime
    version). Those stay out of scope; the ADR records `suite_version` as
    necessary-not-sufficient so the field is not later mistaken for a full
    provenance token.
- Compatibility: additive. Existing `latest_per_model()` is preserved; suite-aware
  selection is a new method. No historical result file is modified.
- Supersession: remains in force until a future DEC adds further comparability
  keys or a run manifest.

### DEC-056
- Date: 2026-08-05
- Status: accepted
- Title: Advisor score reflects measured quantities only (OS-6)
- Context: Phase A OS-6 (`docs/REVIEW-2026-08-04-os5-os6-finalization.md`). The
  advisor blended a constant `quant_score = 1.0` (weight `0.10`) into every score.
  A constant term contributes an identical additive floor to all models, so the
  score neither ranked nor discriminated on that axis — it silently inflated every
  number and implied a quantization signal that did not exist.
- Problem: A score that mixes measured components with a constant placeholder is
  dishonest: it looks like a four-factor judgement but is a three-factor one plus a
  fixed offset. It also risks being read as an absolute quality metric.
- Decision:
  1. **Delete `quant_score`.** The advisor scores only measured quantities:
     throughput, warm TTFT, and VRAM headroom.
  2. **Exact weights.** Re-normalise the surviving three weights to exact fractions
     `4/9` (throughput), `1/3` (TTFT), `2/9` (VRAM). These preserve the prior
     `0.40 : 0.30 : 0.20` ratio. The `* 100.0` display scale is retained.
  3. **Weights are uncalibrated editorial preference**, not a reasoned inference
     from data. They are not tuned against any outcome and may change.
  4. **`viable` is the sole viability signal.** A VRAM-gated model collapses to
     `score = 0.0`; a viable-but-worst model may also be `0.0`. The two are
     distinguished only by `viable`, never by the score value. No consumer may
     derive viability from the score.
  5. **Score is a within-report ordinal** used only to rank viable models produced
     from the same benchmark suite. Score is NOT portable across benchmark suites,
     NOT portable across hardware, NOT portable across future weighting changes, and
     NOT an absolute quality metric. Normalisation is relative to the models in the
     same report: a single result yields a zero TTFT component, and adding a model
     can change the scores of the others.
  6. **Ranking preservation.** Because all three components are non-negative linear
     terms, the relative ordering of viable models is preserved under the
     re-normalisation (affine-invariant); only the absolute numbers shift.
- Alternatives considered:
  - **Keep `quant_score` as a real signal.** Rejected: no quantization data exists
    to populate it; a placeholder that always returns `1.0` is not a signal.
  - **Expose per-component sub-scores.** Rejected: out of scope; no component-score
    API in Phase A.
- Consequences:
  - Positive: the score reflects only measured quantities; the floor is gone; the
    weights are exact and honestly labelled.
  - Negative: absolute score values change (they no longer carry the `+10` floor);
    this is a display change only and does not reorder viable models.
- Compatibility: advisor-owned. Runner, storage, schemas, and `suite_identity` must
  never embed these weights. Historical benchmark JSON is unaffected.
- Supersession: remains in force until a future DEC recalibrates or replaces the
  scoring weights.

### DEC-057
- Date: 2026-08-05
- Status: accepted
- Title: The benchmark VRAM number is device occupancy, named `vram_device_occupied_gib` (OS-6)
- Context: Phase A OS-6. `BenchmarkResult.peak_vram_delta_gb` and
  `AdvisorResult.vram_gb` named the number a "delta", but the runner measures
  `total − min(free)` across before/after snapshots — device-wide occupied VRAM,
  not a per-process delta. The unit was already GiB despite the `_gb` suffix.
- Problem: A field name that misdescribes its referent invites wrong reasoning
  (e.g. treating device occupancy as this process's marginal footprint).
- Decision:
  1. **Canonical field `vram_device_occupied_gib`** on `BenchmarkResult`, replacing
     `peak_vram_delta_gb` as the stored/serialised name.
  2. **Measurement unchanged:** `total − min(free)` across before/after GPU
     snapshots. The number does not change; only its name does.
  3. **Deprecated aliases through Phase A:** `peak_vram_delta_gb` (BenchmarkResult)
     and `vram_gb` (AdvisorResult) remain readable and, for `vram_gb`, remain in the
     serialised advisor payload. Removing either alias REQUIRES a future ADR.
  4. **Canonical precedence (permanent).** Canonical fields always take precedence
     over deprecated aliases during deserialization. If both are present with
     conflicting values, the canonical field wins. This precedence rule is permanent
     unless superseded by a future ADR.
  5. **`HardwareProfile.vram_total_gb` / `vram_free_gb` are unchanged.** They name a
     device capacity/availability, not an occupancy measurement; renaming them is
     out of scope for this change (frozen unchanged, not deferred).
  6. **Known coupling recorded, not fixed.** `runner._check_vram_budget` compares the
     occupancy number against an estimated engine footprint. That comparison logic is
     left unchanged in this change; the coupling is documented here so a future change
     can revisit it deliberately.
- Alternatives considered:
  - **Hard rename with a migration pass over stored JSON.** Rejected: historical
    benchmark JSON must remain readable without migration and is never rewritten.
  - **Rename HardwareProfile fields for symmetry.** Rejected: different referent;
    out of scope.
- Consequences:
  - Positive: the field name matches what is measured; historical files still load.
  - Negative: two names for one number exist through Phase A (canonical + alias).
- Compatibility: additive alias. New JSON uses the canonical field; historical JSON
  loads via the alias and is never rewritten.
- Supersession: alias removal requires a future ADR; the canonical-precedence rule is
  permanent unless a future ADR supersedes it.

### DEC-058
- Date: 2026-08-05
- Status: accepted
- Title: `VLLMEngine` migrates from `LLM` + `EngineDriver` to `AsyncLLM` (B1, `migrate-async-llm-engine`)
- Context: Phase B1 (`docs/PHASE-A-ARCHITECTURE.md` §10, `docs/REVIEW-2026-08-03-architecture.md`
  §9). `EngineDriver` (`engines/driver.py`) owned a synchronous `vllm.LLM.llm_engine`
  exclusively and was the sole caller of `add_request`/`step()`, demultiplexing `step()`
  output to the right request by id — built to fix a specific, reproduced race
  (DEC-038: a one-shot `finished=True` handoff losing output when a different thread's
  `step()` call surfaces your request's terminal output; DEC-039: the driver-thread fix;
  DEC-043: the `_dead_lock` atomicity fix). vLLM 0.22.1 ships `AsyncLLM`
  (`vllm/v1/engine/async_llm.py`), a v1-engine async client that already solves
  per-request delivery, cancellation, and health signaling without a driver thread.
- Problem: `EngineDriver`'s race-class history (DEC-038/039/043) is real complexity
  carried solely to work around the offline `LLM` class having no async per-request
  API. `AsyncLLM` removes the reason for that complexity to exist.
- Decision (full rationale and verification evidence: `openspec/changes/migrate-async-llm-engine/design.md`,
  archived under `openspec/changes/archive/`):
  1. **`_POOL_STEP_LOCK` deleted, not relocated.** Each `AsyncLLM` instance owns an
     independent background engine-core *process* (`async_llm.py:146`), not a shared
     in-process step loop — there is no shared resource left to guard. This does not
     prove multiple co-located `AsyncLLM` instances are safe, only that the specific
     mechanism the old lock guarded against has no equivalent here (see 3).
  2. **`generate()` is derived from `generate_stream()`.** Exactly one code path calls
     into `AsyncLLM.generate()`; both public methods are expressed in terms of it (or a
     shared internal primitive), so terminal metadata cannot be derived twice and drift
     (preserves the DEC-050 single-source guarantee).
  3. **`pool_size > 1`: not guaranteed, not forbidden.** Nothing verified (not
     `PHASE-A-ARCHITECTURE.md`, not DEC-047, not `async_llm.py`) demonstrates `AsyncLLM`
     cannot safely support multiple co-located instances, so B1 adds no new
     construction-time validation for `pool_size` in either direction. B6 remains the
     only phase authorized to redesign multi-engine serving.
  4. **Cancellation is a compatibility invariant.** `AsyncLLM.generate()`'s
     `except (asyncio.CancelledError, GeneratorExit)` handler calls a real
     `engine_core.abort_requests_async(...)` — a genuine behavioral improvement over
     `EngineDriver`, where an abandoned consumer only stopped *reading*
     (`_pending[request_id]` removed solely on `output.finished`), so a disconnected
     client's computation kept running to completion regardless. This is an intentional
     behavior change, recorded as a compatibility invariant so a later "simplification"
     cannot silently swallow `GeneratorExit` and regress it back to
     consumer-side-only abandonment.
  5. **Health: `AsyncLLM.errored` / `dead_error` replace `is_dead` / `EngineDriverDeadError`.**
     `VLLMEngine.is_healthy()` becomes `not self._llm.errored`. The repo-local
     `EngineDriverDeadError` wrapper is deleted; vLLM's own `EngineDeadError` is used
     directly.
  6. **KV cache stats: same attribute chain, different root.** `_log_kv_cache_stats()`
     reads `self._llm.vllm_config.cache_config` instead of
     `self._llm.llm_engine.vllm_config.cache_config` — root object only; the
     `.num_gpu_blocks`/`.block_size` chain is unchanged.
  7. **Timeout: the wrapper becomes effective, not cosmetic.** `AsyncLLM.generate()` has
     no native per-request timeout. `VLLMEngine` still supplies its own
     `_COMPLETION_TIMEOUT_S` wrapper, but where a pre-migration timeout only stopped
     *waiting* (the underlying computation ran to completion regardless), the same
     wrapper now delivers a real `CancelledError` into `AsyncLLM.generate()`, which
     (per 4) triggers a real engine-side abort. A timeout now actually stops the
     engine-side computation, for the first time. Intentional behavioral change, not a
     preserved one.
  8. **`derive_terminal_metadata` relocated, not reimplemented.** Moved from
     `driver.py` into `vllm_engine.py` verbatim (signature and body unchanged) per
     DEC-050 §3 and the Phase A audit's OWN-B5 finding — the function is not
     `EngineDriver`-specific, and reimplementing it would have reopened the exact
     usage-drift risk DEC-050 closed.
  - Verified before implementation (not assumed): installed vLLM version (0.22.1);
    `AsyncLLM.generate()`'s cancellation → abort path; `AsyncLLM.errored`/`dead_error`/
    `check_health()` semantics; and — the one substantive open question carried into
    implementation — that a request submitted after the engine has entered a terminal
    failure state is rejected immediately and synchronously, via two independent
    redundant guards (`AsyncLLM.add_request()`'s own `errored` check, and the transport
    client's `ensure_alive()`), not accepted-then-failed-later or orphaned.
- Alternatives considered:
  - **Keep `EngineDriver`, relocate it unchanged.** Rejected: keeps the DEC-038/039/043
    race-class complexity alive for no reason once `AsyncLLM` provides the same
    guarantees natively.
  - **A second independent `AsyncLLM.generate()` call site for the non-streaming path.**
    Rejected (Decision 2): would re-derive terminal metadata outside
    `derive_terminal_metadata`, reopening the DEC-050 drift risk, and double the surface
    that must implement cancellation/timeout handling correctly.
  - **Fail loudly on `pool_size > 1`.** Rejected (Decision 3): no verified evidence
    supports it; would be a construction-time claim the evidence does not back.
- Consequences:
  - Positive: `EngineDriver` and its race-class history (DEC-038/039/043) are deleted
    outright, not carried forward. Cancellation and timeout become real engine-side
    signals for the first time — see the changelog entry for the user-facing framing.
  - Negative: two real behavioral changes (Decision 4, Decision 7) mean this is not a
    pure refactor; anyone benchmarking timeout- or disconnect-adjacent scenarios will
    see different engine-side resource usage than before this change.
  - Neutral: `pool_size > 1` remains an open architectural question, explicitly not
    resolved by this change (Decision 3) — deferred to B6.
- Compatibility: `BaseEngine`'s declared contract, `ChatService`, `AdmissionController`,
  and every route handler required zero code changes (verified: no diff outside
  `engines/`, `tests/`, `pyproject.toml`, and documentation). Streamed and non-streamed
  usage continue to derive from exactly one function. No new package or dual type
  system introduced.
- Supersession: `pool_size > 1` semantics remain open until a future phase (most likely
  B6) verifies `AsyncLLM` multi-instance behavior explicitly; this DEC does not resolve
  that question and is not superseded by leaving it open.
