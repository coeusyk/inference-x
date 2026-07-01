# InferenceX — Phases

**All phases 0–8 are complete (2026-07).** This file is the historical milestone record
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
| 7 | VRAM-aware sizing, tiers, and observability wiring | Done |
| 8 | Admission control (context/KV enforcement) | Done (partial scope, see notes) |

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
- Saves results to `benchmarks/results/results-{model}-{date}.json`
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
- Skips results whose saved `hardware` mismatches current GPU (exact name) or VRAM total
  (>0.5 GB tolerance); warning only
- Soft-warns on legacy results with `hardware: null` but still ranks them
- TTFT scoring uses warm prompts only (`prompt_results[1:]`), skipping cold-start outliers
- VRAM gate: `peak_vram_delta_gb + 0.5 GB ≤ vram_total_gb` (not free VRAM × margin)
- Applies a scoring function across: throughput, TTFT, VRAM headroom, quantization fit
- Returns `AdvisorReport` with ranked list and plain-language reasoning per model:
  `"qwen2.5-0.5b — 168 tok/s, 12ms TTFT, 7.15 GB VRAM"`
  `"llama3-8b — requires 7.50 GB + 0.5 GB buffer = 8.00 GB, only 6.0 GB total. Skip."`
- Does not make network calls; advice is purely local from observed results

**6.5 — Advisor CLI surface**
- `make advise` — runs advisor against latest benchmark results; prints `WARNING:` lines
  to stderr when results are skipped or lack hardware provenance
- When no benchmark JSON exists, prints a static VRAM fit table from `config/models.yaml`
  (parameter-count estimates) instead of exiting with an error
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
- [x] Results stored in `benchmarks/results/` for reproducible comparison
- [x] Benchmark methodology documented in DECISIONS.md or README where applicable

**Post-phase note (2026-06-25):** Benchmark results now persist `hardware` at run time;
advisor skips cross-machine mismatches. `peak_vram_delta_gb` measures VRAM footprint
(not a naive before−after free-VRAM delta, which read 0 when the model was pre-loaded).
Loading-screen failures surface actionable errors from `logs/playground-server.log`.

**Post-phase note (2026-06-30):** Advisor scoring fixes (DEC-036): warm TTFT, VRAM gate
against total VRAM + 0.5 GB buffer, exact GPU name matching, static config fallback when
no results exist. Benchmark JSON lives under `benchmarks/results/` (gitignored).

---

## Phase 7 — VRAM-Aware Sizing, Tiers, and Observability Wiring
**Goal:** Make VRAM budgeting quant-aware, give the deployment an explicit per-GPU-class
capacity contract, and wire the previously-stubbed observability HTTP surface (including
streaming TTFT/tokens-per-sec, which the middleware could not see before this phase).
This is Phase 1 of the larger VRAM-aware architecture plan (see DEC-037); admission
control (Phase 2 of that plan) and CPU offload (Phase 3) are deferred to future phases.

Deliverables:
- Quant-aware `estimate_weight_gib()` and every caller in `utils/vllm_pool_config.py`
  (bf16/None → 2 bytes/param, int8/fp8 → 1.0, awq/gptq/int4 → ~0.55)
- `config/vram_tiers.yaml` + `utils/vram_tiers.py` — 6gb/12gb/24gb tier resolution from
  probed VRAM, exposed via `AppSettings.get_vram_tier()` and logged at startup
  (resolution and logging only in this phase; enforcement is future work)
- `qwen2.5-7b-awq` added to `config/models.yaml` as the first exercised 4-bit model
- `GET /v1/metrics` (`schemas/metrics.py`, `api/routes/metrics.py`) — request metrics
  plus a live per-model VRAM breakdown (weights, real `kv_capacity_tokens`, free/total)
- Streaming (SSE) chat completions now record TTFT and approximate tokens/sec via a
  wrapped `body_iterator` in `ObservabilityMiddleware`
- `GET /v1/models` additively exposes `quantization`, `max_model_len`,
  `estimated_weights_gib` per model

Exit criteria:
- [x] `uv run pytest tests/unit -v` — 352/352 pass
- [x] VRAM tier resolves and logs correctly at startup (verified: `6gb` tier on the RTX
  3060 Laptop 6GB dev box)
- [x] `GET /v1/metrics` returns a live weights/KV/free VRAM breakdown against a running
  server
- [x] Streaming request populates `avg_ttft_ms` / `avg_tokens_per_sec` on
  `GET /v1/metrics` (verified live: ~255ms TTFT, ~29 tok/s on qwen2.5-0.5b)
- [x] `GET /v1/models` returns quantization/VRAM metadata for all registered models,
  including the new AWQ variant
- [x] Non-obvious choices recorded in DECISIONS.md (DEC-037)

**Post-phase note (2026-07-01):** `qwen2.5-7b-awq` is sized for the 12GB tier and not
part of the default 6GB dev pool — quant-aware sizing is unit-tested but not yet
validated on real 12GB+ hardware. Tier *enforcement* against `max_num_seqs`/
`max_model_len_cap`, the `AdmissionController`, and non-streaming continuous-batching
fix are Phase 2 of the VRAM-aware plan (DEC-037) and remain unbuilt.

---

## Phase 8 — Admission Control (Context/KV Enforcement)
**Goal:** Enforce context-length and KV-pool limits before a request reaches vLLM, instead
of letting oversized requests fail unpredictably inside the engine. This is Phase 2 of the
VRAM-aware architecture plan (DEC-037/DEC-038). Scope was narrowed mid-phase after a real
concurrency bug was found in live testing — see notes below.

Deliverables:
- `routing/admission.py` — `AdmissionController` enforces prompt/output context limits
  (`ModelEntry.max_model_len` ∩ VRAM tier `max_model_len_cap` ∩ request
  `max_context_tokens`) and KV-pool pressure (in-flight token reservations vs.
  `engine.kv_capacity_tokens`), clamping for `priority: interactive` or rejecting for
  `priority: batch`
- New request fields on `ChatCompletionRequest`: `max_context_tokens`,
  `max_output_tokens` (alias for `max_tokens`), `priority`
- `VLLMEngine.count_prompt_tokens()` — real tokenizer-based prompt token count for
  admission math
- `ContextTooLongError` (400, reuses the existing sanitized ValueError handler) and
  `EngineSaturatedError` (429 + `Retry-After`, new handler in `api/errors.py`)
- `ChatService` wires `AdmissionController` around every `generate()`/`generate_stream()`
  call, releasing the KV reservation in a `finally` block

Exit criteria:
- [x] `uv run pytest tests/unit -v` — 371/371 pass (17 new: 15 `test_admission.py`, 2
  route-level 400/429 integration tests)
- [x] A prompt over `max_context_tokens` returns 400 (verified live against a running
  server)
- [x] A batch-tier request against a saturated KV budget returns 429 with `Retry-After`
  (verified via unit + route-level tests)
- [x] Normal chat and streaming continue to work unchanged against a live server
- [x] Non-obvious choices recorded in DECISIONS.md (DEC-038)

**Not delivered this phase (see DEC-038):**
- **Non-streaming continuous-batching fix** — attempted, unit-tested, then reverted after
  a live 2-concurrent-request test reproducibly lost one request's output. The original
  plan called this "low-risk, self-contained"; it isn't — a correct fix needs a shared
  per-engine driver thread multiplexing `step()` output to per-request queues, not N
  independent request-owned loops. Non-streaming requests still serialize per model,
  same as before this phase.
- **Engine knob surfacing** (`block_size`, `max_num_batched_tokens`, `kv_cache_dtype`,
  `enable_prefix_caching`) — not started.
- **`precision` request field / variant selection** — deliberately not added; meaningless
  without model variant sets (VRAM-aware plan Component 1), which don't exist.
- Phase 3 of the VRAM-aware plan (CPU/weight offload, prefix caching) remains untouched.

---

## Phase 9 — Shared Engine Driver Thread (Non-Streaming Batching Fix)
**Goal:** Ship the non-streaming continuous-batching fix DEC-038 named but didn't deliver —
a single shared per-engine driver thread that is the only caller of `add_request`/`step()`,
eliminating the request-id-discard race a per-request step loop hit under live concurrency.

Deliverables:
- New `engines/driver.py` — `EngineDriver`: owns one vLLM sync `llm_engine` exclusively;
  `submit_stream()`/`submit_complete()` hand back a queue/future; the driver thread drains
  submissions, calls `add_request`/`step()`, and dispatches each output to its registered
  channel by `request_id`
- `VLLMEngine._run_completion` and `generate_stream` both submit through the driver instead
  of running independent step loops — unifying streaming and non-streaming onto one step
  loop per engine
- Driver failure (`step()` raising) broadcasts to every pending channel and flips
  `EngineDriver.is_dead`, checked by `VLLMEngine.is_healthy()`
- `_POOL_STEP_LOCK` cross-engine serialization (`pool_size > 1`) preserved unchanged in
  meaning, now acquired by the driver thread instead of by each request's own thread

Exit criteria:
- [x] `uv run pytest tests/unit -v` — 379/379 pass (8 new `test_engine_driver.py` cases,
  1 new non-streaming-via-driver case, existing streaming tests updated to wire a real
  `EngineDriver` over a mocked `llm_engine`)
- [x] 2 and 4 concurrent non-streaming requests against a live server all return
  complete, correctly-attributed, non-truncated output
- [x] Determinism check (`temperature=0`, unique per-request tokens): concurrent-run
  output byte-identical to the same prompts run fully sequentially — no cross-request
  state leakage
- [x] Streaming chat completions unaffected (verified live)
- [x] Non-obvious choices recorded in DECISIONS.md (DEC-039)

**Not delivered this phase:**
- Engine knob surfacing (`max_num_batched_tokens`, `enable_prefix_caching`, wiring
  `block_size`/`kv_cache_dtype` through to `LLM(...)`) — separately proposed under
  `openspec/changes/add-engine-knob-surfacing`, not implemented.
- Model variant routing (`family` grouping, load-time variant selection) — separately
  proposed under `openspec/changes/add-model-variant-routing`, not implemented.

---

## Phase 10 — Engine Knob Surfacing

**Goal:** Make the VRAM/concurrency knobs `config/vram_tiers.yaml` has declared since
DEC-037 (`block_size`, `kv_cache_dtype`, `max_num_seqs`) actually reach the vLLM engine,
add the two knobs that were missing entirely (`max_num_batched_tokens`,
`enable_prefix_caching`), and give `AdmissionController` visibility into sequence
concurrency saturation.

Deliverables:
- `VramTier` / `config/vram_tiers.yaml` — additive `max_num_batched_tokens`,
  `enable_prefix_caching` per tier; `ModelEntry.max_num_batched_tokens` per-model
  override
- `apply_tier_knobs()` (`utils/vllm_pool_config.py`) — resolves tier ceiling ∩ per-model
  override for `max_num_seqs`/`max_num_batched_tokens`; passes `block_size`/
  `kv_cache_dtype`/`enable_prefix_caching` straight from the tier
- `VLLMEngine.__init__` forwards all five knobs to `LLM(**kwargs)`; `_build_engine_pool`
  (`api/deps.py`) resolves the VRAM tier and applies knobs before engine construction
  (previously never resolved a tier at all)
- `AdmissionController` gains an in-flight-sequence-count gate
  (`_InFlightSeqTracker`) — 429 for both priorities when a model's resolved
  `max_num_seqs` is saturated, no clamp path

Exit criteria:
- [x] `uv run pytest tests/unit -v` — 401/401 pass (17 new across
  `test_vllm_pool_config.py`, `test_vllm_engine_knobs.py`, `test_deps_tier_knobs.py`,
  `test_admission.py`, `test_routes.py`)
- [x] Live-verified: a running `opt-125m` server on the 6gb tier shows vLLM's own startup
  log reporting `max_num_batched_tokens=2048`/`enable_prefix_caching=False` — the
  resolved tier values — and serves a normal chat completion afterward
- [x] Model-level knob overrides compose correctly with tier ceilings (unit-tested:
  tighter override honored, looser override clamped down)
- [x] Non-obvious choices recorded in DECISIONS.md (DEC-040)

**Not delivered this phase:**
- Model variant routing — separately proposed under
  `openspec/changes/add-model-variant-routing`, not implemented.

---

## Phase 11 — Model Variant Routing

**Goal:** Let one logical model be declared as multiple config entries at different
quantizations, grouped by `family`, and automatically load the highest-precision variant
that fits the current VRAM tier's budget — the variant-set prerequisite DEC-038 named
for a future `precision` request field (not added this phase; still not needed until a
per-request use case exists).

Deliverables:
- `ModelEntry.family` (additive), `ModelRegistry.variants(family)` — groups config
  entries; ungrouped entries are unaffected (a "family of one")
- `routing/variant_selector.select_variant()` — highest-precision-that-fits selection
  reusing `utils/vllm_pool_config`'s existing quant-aware bytes-per-param table as the
  sole precision-ordering source; `NoVariantFitsError` when nothing fits
- `_build_engine_pool` (`api/deps.py`) resolves each `INFERENCE_X_LOADED_MODELS` entry
  to a concrete name — a registered name bypasses the selector unchanged; an unknown
  name is treated as a family and resolved, or falls through to the existing
  "not registered" error
- Worked example in `config/models.yaml`: `qwen2.5-7b-bf16`/`qwen2.5-7b-awq` grouped
  under `family: qwen2.5-7b`

Exit criteria:
- [x] `uv run pytest tests/unit -v` — 419/419 pass (18 new)
- [x] Live-verified: a temporary family of two real small models resolved and loaded
  through the actual server startup path (not just mocked)
- [x] A model with no `family` set behaves identically to before this phase
  (regression-tested against the full existing suite)
- [x] Non-obvious choices recorded in DECISIONS.md (DEC-041)

**Known scope boundary:** `INFERENCE_X_DEFAULT_MODEL` is not resolved through the
selector — operators loading a family should set the default model to one of its
concrete variant names. Resolving the router's default would require threading VRAM
tier state into `_build_router`, out of scope for this phase.

---

## Phase 12 — Default-Model Variant Resolution

**Goal:** Close Phase 11's known scope boundary — `INFERENCE_X_DEFAULT_MODEL` now
resolves a registered `family` name to its best-fitting variant at startup, the same
rule `INFERENCE_X_LOADED_MODELS` already follows.

Deliverables:
- `_resolve_default_model()` (`api/deps.py`) — concrete names pass through unchanged;
  a family name (with a resolved VRAM tier) is resolved via the existing
  `variant_selector.select_variant()`; anything else defers to `TaskRouter`'s
  pre-existing `DefaultModelPolicy` error, unchanged
- `_build_router` resolves the VRAM tier and available VRAM before constructing
  `TaskRouter`, threading both through
- Startup INFO log: `Default model resolved: {family} → {variant} (tier: {tier_name})`

Exit criteria:
- [x] `uv run pytest tests/unit -v` — 425/425 pass (6 new in
  `test_default_model_resolution.py`)
- [x] Live-verified: startup log shows the resolved variant for a family-name default;
  a request routes to that resolved variant
- [x] Non-obvious choices recorded in DECISIONS.md (DEC-042)

**Phase 3C's deferred/scope-boundary list is now empty.**

---

## Decision rule: when to create a new phase

A new phase is warranted when:
1. The feature requires new top-level modules (not additive files inside existing ones)
2. It has its own exit criteria that can be validated independently
3. It does not require changes to the stable API contract (or explicitly versions the contract)

If a feature fits inside an existing phase's module boundary, it is a task, not a phase.