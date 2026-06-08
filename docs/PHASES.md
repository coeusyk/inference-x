# InferenceX — Phases

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
- [ ] AGENTS.md present and respected by agent
- [ ] ARCHITECTURE.md, PHASES.md, DECISIONS.md, ARTICLENOTES.md present
- [ ] OpenSpec initialized with platform spec
- [ ] config/ scaffolded with models.yaml, routing.yaml, server.yaml, logging.yaml

---

## Phase 1 — Core Inference Engine
**Goal:** One runnable vLLM-backed endpoint. Nothing else.

Deliverables:
- `POST /v1/chat/completions` — streaming and non-streaming
- `GET /health` — reflects real engine state
- Typed schemas (ChatCompletionRequest, ChatCompletionResponse)
- Settings loader from config/

Exit criteria:
- [ ] `uv run pytest tests/unit -v` — all pass
- [ ] Smoke test against real model succeeds
- [ ] Health returns 503 when engine fails to load
- [ ] Article notes updated

---

## Phase 2 — Model Registry and Routing
**Goal:** Multiple models loadable; routing layer decides which model handles which request.

Deliverables:
- `GET /v1/models` — lists loaded models
- ModelRegistry backed by `config/models.yaml`
- TaskRouter with default fallback chain
- Routing config in `config/routing.yaml`

Exit criteria:
- [ ] All Phase 1 tests still pass
- [ ] Routing logic covered by unit tests
- [ ] `GET /v1/models` reflects models.yaml
- [ ] Article notes updated

---

## Phase 3 — Observability Pipeline
**Goal:** Every request is measured. No route handler changes.

Deliverables:
- ObservabilityMiddleware — latency, token counts, error flag
- MetricsRecorder + InMemoryStorage (capped deque, 1000 records)
- MetricsService — recent(n), summary, p95 latency
- JsonLineExporter (opt-in via INFERENCE_X_METRICS_FILE)

Exit criteria:
- [ ] All Phase 1–2 tests still pass
- [ ] 30+ observability-specific unit tests
- [ ] No change to API request/response contracts
- [ ] Article notes updated

---

## Phase 4 — Playground and Evaluation
**Goal:** Interactive terminal UI to send prompts, compare models, and see metrics.

Deliverables:
- Textual-based playground (`playground/app.py`)
- ModelSelectScreen — interactive model picker on startup (replaces env var)
- Side-by-side model compare mode
- Markdown rendering for model responses
- Status bar with token counts and latency
- `playground/client.py` — headless CLI batch runner
- `make playground` — single command start

Exit criteria:
- [ ] `make playground` starts server + UI in one command
- [ ] Model selection screen shown before main UI
- [ ] Compare mode works across ≥2 models
- [ ] Markdown renders correctly in response panel
- [ ] Article notes updated

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
- ARTICLENOTES.md converted to article draft

Exit criteria:
- [x] Security audit findings H1–H5 from audit report resolved (DEC-SEC-01: one CVE in transitive dep, accepted)
- [x] Deferred items (auth, rate limiting, streaming timeout) documented in DECISIONS.md (DEC-DEFER-01/02/03)
- [x] All unit tests pass (195/195 as of 2026-06-08)
- [x] README covers setup from scratch on a fresh WSL2 machine
- [x] Article draft written (docs/ARTICLE_DRAFT.md)

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
  peak VRAM usage (via `nvidia-smi` or `pynvml`), requests/sec at concurrency 1/4/8
- Saves results to `docs/benchmarks/results-{model}-{date}.json`
- Invoked via `make benchmark MODEL=qwen2.5-0.5b` or `make benchmark-all`

**6.2 — Benchmark results schema** (`src/inference_x/benchmarks/schemas.py`)
- `BenchmarkResult` dataclass: model name, hardware snapshot, prompt suite id,
  metrics dict, timestamp
- Hardware snapshot captured at run time: GPU name, VRAM total/free, CPU cores, RAM total

**6.3 — Hardware profiler** (`src/inference_x/benchmarks/hardware.py`)
- Detects GPU via `pynvml` or falls back to `nvidia-smi` subprocess
- Detects CPU and RAM via `psutil`
- Returns a `HardwareProfile` used by advisor and stored in benchmark results

**6.4 — Model advisor** (`src/inference_x/benchmarks/advisor.py`)
- Takes a `HardwareProfile` and a set of `BenchmarkResult` records
- Applies a scoring function across: throughput, TTFT, VRAM headroom, quantization fit
- Returns a ranked list with plain-language reasoning per model:
  `"qwen2.5-0.5b is fastest on your hardware (42 tok/s, 3.1GB VRAM). Use for chat."`
  `"llama3-8b requires 7.2GB VRAM — exceeds your free headroom of 5.1GB. Skip for now."`
- Does not make network calls; advice is purely local from observed results

**6.5 — Advisor CLI surface**
- `make advise` — runs advisor against latest benchmark results, prints report to terminal
- `GET /v1/benchmark/results` — returns stored results as JSON (read-only)
- `GET /v1/benchmark/advise` — returns advisor output as JSON

**6.6 — Playground integration**
- New "Benchmark" tab in Textual playground
- Shows hardware profile, per-model throughput bars, and advisor recommendation
- "Run benchmark" button triggers benchmark runner for selected model

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
- [x] Playground benchmark tab shows results and recommendation (BenchmarkTab wired into TabbedContent)
- [x] Results stored in `docs/benchmarks/` for article inclusion
- [x] Article notes updated with benchmark methodology and sample output

---

## Decision rule: when to create a new phase

A new phase is warranted when:
1. The feature requires new top-level modules (not additive files inside existing ones)
2. It has its own exit criteria that can be validated independently
3. It does not require changes to the stable API contract (or explicitly versions the contract)

If a feature fits inside an existing phase's module boundary, it is a task, not a phase.