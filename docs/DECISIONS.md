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
  desktop VRAM deltas against a laptop's current free VRAM. Legacy JSON in `docs/benchmarks/`
  had no provenance.
- Decision: Add optional `hardware: HardwareProfile` to `BenchmarkResult` (default `None` for
  legacy files). Persist `hardware_before` from `BenchmarkRunner.run()`. Return `AdvisorReport`
  (`ranked` + `warnings`) from `ModelAdvisor.rank()`: skip results when saved hardware
  mismatches current GPU name (substring match) or VRAM total (>0.5 GB tolerance); soft-warn
  when `hardware` is null but still rank. Expose `warnings` on `GET /v1/benchmark/advise` and
  print `WARNING:` lines in `scripts/advise.py`.
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
- Status: accepted
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
- Consequences: Marginal fits (e.g. 3.0 GB footprint, 3.3 GB free) are correctly marked
  non-viable (3.6 GB required). Default 1.20 is conservative but not extreme. Measuring
  true cold-load peak in the benchmark runner (server restart per model) is deferred to a
  later phase.

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
  (covers local chat; drops KV cache from ~2 GB to ~0.5 GB); `llama3-8b` uses 4096.
- Why free-VRAM basis: vLLM's utilization is a fraction of total VRAM — deriving the
  fraction from free VRAM is the only way to stay within actually available memory.
  Sizing uses `torch.cuda.mem_get_info` when CUDA is active (same allocator view as
  vLLM); nvidia-smi is used only as fallback before torch init.
- Why 8192: ~6000 words of context; sufficient for playground and benchmark use cases;
  raise per-model in `models.yaml` when longer context is needed.
- Why 0.4 GB buffer: covers CUDA context growth during inference.
- Deferred: llama3-8b quantization for 8 GB GPUs; multi-model sequential VRAM profiling.
- Consequences: Startup logs show resolved utilization with `(free − buffer) / total`
  breakdown. `BenchmarkResult` stores `max_model_len`; advisor warns on config drift.