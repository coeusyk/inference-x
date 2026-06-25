# InferenceX — Phases

**All phases 0–6 are complete (2026-06).** This file is the historical milestone record
and phase-gating rules. New work uses OpenSpec (`openspec/changes/`) — do not add features
here without a new phase section and exit criteria.

## Status

| Phase | Name | Status |
|-------|------|--------|
| 0 | Repo foundation | Done |
| 1 | Core inference engine | Done |
| 2 | Model registry and routing | Done |
| 3 | Observability pipeline | Done |
| 4 | Playground and evaluation | Done |
| 5 | Hardening and publication | Done |
| 6 | Benchmark suite and model advisor | Done |

---

## Phase 0 — Repo Foundation
**Goal:** Establish repo structure, documentation skeleton, OpenSpec workflow, and Cursor rules before any code lands.

Exit criteria:
- [x] AGENTS.md present and respected by agent
- [x] ARCHITECTURE.md, PHASES.md, DECISIONS.md present
- [x] OpenSpec initialized with platform spec
- [x] config/ scaffolded with models.yaml, routing.yaml, server.yaml, logging.yaml

---

## Phase 1 — Core Inference Engine
**Goal:** One runnable vLLM-backed endpoint. Nothing else.

Deliverables:
- `POST /v1/chat/completions` — streaming and non-streaming
- `GET /health` — reflects real engine state
- Typed schemas (ChatCompletionRequest, ChatCompletionResponse)
- Settings loader from config/

Exit criteria:
- [x] `uv run pytest tests/unit -v` — all pass
- [x] Smoke test against real model succeeds
- [x] Health returns 503 when engine fails to load
- [x] Non-obvious choices recorded in DECISIONS.md where applicable

---

## Phase 2 — Model Registry and Routing
**Goal:** Multiple models loadable; routing layer decides which model handles which request.

Deliverables:
- `GET /v1/models` — lists loaded models
- ModelRegistry backed by `config/models.yaml`
- TaskRouter with default fallback chain
- Routing config in `config/routing.yaml`

Exit criteria:
- [x] All Phase 1 tests still pass
- [x] Routing logic covered by unit tests
- [x] `GET /v1/models` reflects models.yaml
- [x] Non-obvious choices recorded in DECISIONS.md where applicable

---

## Phase 3 — Observability Pipeline
**Goal:** Every request is measured. No route handler changes.

Deliverables:
- ObservabilityMiddleware — latency, token counts, error flag
- MetricsRecorder + InMemoryStorage (capped deque, 1000 records)
- MetricsService — recent(n), summary, p95 latency
- JsonLineExporter (opt-in via INFERENCE_X_METRICS_FILE)

Exit criteria:
- [x] All Phase 1-2 tests still pass
- [x] 30+ observability-specific unit tests
- [x] No change to API request/response contracts
- [x] Non-obvious choices recorded in DECISIONS.md where applicable

---

## Phase 4 — Playground and Evaluation
**Goal:** Interactive terminal tools to chat, compare models, and run batch prompts.

Deliverables:
- **Daily-driver chat** — `playground/chat.py` + `make chat` (multi-turn history, SSE streaming)
- **Compare playground** — `playground/app.py` + `make playground` (two models side-by-side)
- **Batch CLI** — `playground/client.py` (rich terminal output, compare across servers)
- `ModelSelectScreen` — single-model (`chat.py`) or two-model compare (`app.py`) picker
- `LoadingScreen` — phase titles, step progress, live tail of `logs/playground-server.log`;
  actionable error summary on startup failure
- Markdown rendering for streamed responses
- Per-panel usage footers in compare mode

Exit criteria:
- [x] `make chat` starts server + single-model chat CLI
- [x] `make playground` starts server + compare UI (two models)
- [x] Model selection screen shown before main UI
- [x] Compare mode streams the same prompt to ≥2 models
- [x] Markdown renders correctly in response panels
- [x] Non-obvious choices recorded in DECISIONS.md where applicable

**Post-phase note (2026-06):** Benchmark UI was briefly in `app.py` but removed; benchmarks
stay CLI-only (`make benchmark`, `make advise`). Chat and compare are separate entry points.

---

## Phase 5 — Hardening and Publication
**Goal:** Security-hardened, documented, and ready to be open-sourced and written about.

Deliverables:
- Schema constraints: `max_length` on content, `le=4096` on max_tokens
- Sanitized error responses (no internal paths to clients)
- Default bind to `127.0.0.1`
- `config/logging.yaml` fully populated
- Playground base-URL SSRF guard
- `make stop` target
- README complete (install, quickstart, make targets, config reference)
- MIT LICENSE at repo root

Exit criteria:
- [x] Security audit findings H1-H5 from audit report resolved (DEC-SEC-01: one CVE in transitive dep, accepted)
- [x] Deferred items (auth, rate limiting, streaming timeout) documented in DECISIONS.md (DEC-DEFER-01/02/03)
- [x] All unit tests pass (273/273 as of 2026-06-25)
- [x] README covers setup from scratch on a fresh WSL2 machine
- [x] MIT LICENSE present at repo root

---

## Phase 6 — Benchmark Suite and Model Advisor
**Goal:** Help the user decide which model is best for their hardware. Produces reproducible
benchmark results and a recommendation report.

### Problem being solved
Different hardware (VRAM, CPU, RAM) makes different models viable. A user with a 6GB GPU
should not have to manually test every model to discover that qwen2.5-0.5b outperforms
tinyllama-chat on their setup. InferenceX should tell them.

### What this phase builds

**6.1 — Benchmark runner** (`scripts/benchmark.py`)
- Runs a fixed prompt suite against one or more loaded models
- Measures: tokens/sec (throughput), time-to-first-token (TTFT), p50/p95/p99 latency,
  VRAM footprint (`peak_vram_delta_gb` = total − min free VRAM before/after; works when
  the model is already loaded on the server)
- Saves results to `docs/benchmarks/results-{model}-{date}.json`
- Invoked via `make benchmark MODEL=qwen2.5-0.5b` or `make benchmark-all`

**6.2 — Benchmark results schema** (`src/inference_x/benchmarks/schemas.py`)
- `BenchmarkResult`: model name, prompt suite id, per-prompt metrics, timestamp,
  `peak_vram_delta_gb`, optional `hardware` snapshot (`HardwareProfile`)
- `AdvisorReport`: ranked `AdvisorResult` list plus `warnings` (hardware mismatch skips,
  legacy results without `hardware`)
- Hardware snapshot captured at run time: GPU name, VRAM total/free, CPU cores, RAM total
- Legacy JSON without `hardware` deserialises with `hardware: null` (no migration)

**6.3 — Hardware profiler** (`src/inference_x/benchmarks/hardware.py`)
- Detects GPU via `pynvml` or falls back to `nvidia-smi` subprocess
- Detects CPU and RAM via `psutil`
- Returns a `HardwareProfile` used by advisor and stored in benchmark results

**6.4 — Model advisor** (`src/inference_x/benchmarks/advisor.py`)
- Takes a `HardwareProfile` and a set of `BenchmarkResult` records
- Skips results whose saved `hardware` mismatches current GPU/VRAM (warning only)
- Soft-warns on legacy results with `hardware: null` but still ranks them
- Applies a scoring function across: throughput, TTFT, VRAM headroom, quantization fit
- Returns `AdvisorReport` with ranked list and plain-language reasoning per model:
  `"qwen2.5-0.5b is fastest on your hardware (42 tok/s, 3.1GB VRAM). Use for chat."`
  `"llama3-8b requires 7.2GB VRAM — exceeds your free headroom of 5.1GB. Skip for now."`
- Does not make network calls; advice is purely local from observed results

**6.5 — Advisor CLI surface**
- `make advise` — runs advisor against latest benchmark results; prints `WARNING:` lines
  to stderr when results are skipped or lack hardware provenance
- `GET /v1/benchmark/results` — returns stored results as JSON (read-only)
- `GET /v1/benchmark/advise` — returns advisor output as JSON (includes `warnings`)

**6.6 — Benchmark CLI and API** (playground TUI integration removed 2026-06)
- `make benchmark` / `make benchmark-all` — run prompt suite via `scripts/benchmark.py`
- `make advise` — ranked report via `scripts/advise.py`
- `GET /v1/benchmark/results` and `GET /v1/benchmark/advise` — read-only JSON APIs
- A Benchmark tab existed briefly in `playground/app.py`; removed in favor of CLI-only
  workflows to keep the compare TUI focused

**6.7 — Benchmark prompt suite** (`benchmarks/prompts/`)
- `standard.json` — 10 prompts covering: short factual, long generation, code, reasoning
- Suite is fixed and versioned so results are comparable across runs

### What this phase does NOT do
- Does not train or fine-tune models
- Does not make cloud API calls
- Does not implement automated model downloading
- Does not score quality (e.g. MMLU) — throughput and resource metrics only

### Exit criteria
- [x] `make benchmark MODEL=qwen2.5-0.5b` produces a result JSON (live run: 175.6 tok/s, 2026-06-08)
- [x] `make benchmark-all` iterates qwen2.5-0.5b + tinyllama-chat (server must have each model loaded)
- [x] `make advise` prints a ranked recommendation with reasoning (scripts/advise.py implemented)
- [x] Hardware profiler correctly reads GPU name and VRAM on WSL2 with CUDA (pynvml → nvidia-smi → CPU-only chain)
- [x] Advisor scoring function unit-tested with fixture hardware profiles (10 tests: 6GB, 24GB, CPU-only)
- [x] Benchmark results schema validated with Pydantic (round-trip test passing)
- [x] Benchmark CLI and API routes implemented; playground uses CLI only (no Benchmark tab)
- [x] Results stored in `docs/benchmarks/` for reproducible comparison
- [x] Benchmark methodology documented in DECISIONS.md or README where applicable

**Post-phase note (2026-06-25):** Benchmark results now persist `hardware` at run time;
advisor skips cross-machine mismatches. `peak_vram_delta_gb` measures VRAM footprint
(not a naive before−after free-VRAM delta, which read 0 when the model was pre-loaded).
Loading-screen failures surface actionable errors from `logs/playground-server.log`.

---

## Decision rule: when to create a new phase

A new phase is warranted when:
1. The feature requires new top-level modules (not additive files inside existing ones)
2. It has its own exit criteria that can be validated independently
3. It does not require changes to the stable API contract (or explicitly versions the contract)

If a feature fits inside an existing phase's module boundary, it is a task, not a phase.