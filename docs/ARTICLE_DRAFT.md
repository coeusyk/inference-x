# Building InferenceX: A Self-Hosted LLM Inference Platform

## Introduction

I wanted to run large language models on my own hardware — not for privacy reasons
specifically, but because I wanted to understand the full stack. Cloud APIs abstract
away the interesting parts: how models are loaded, how tokens are generated, how
latency actually breaks down, and why some models feel faster than others on the same
GPU. I wanted to see all of that, and I wanted it to be controllable, scriptable, and
demo-able from a terminal.

The result is InferenceX: a self-hosted LLM inference platform backed by vLLM, with
an OpenAI-compatible API, a model registry, an observability pipeline, and an
interactive Textual playground. It runs entirely on a single WSL2 machine with one
consumer-grade GPU and installs in a single `uv sync`.

This is for developers who want to host their own inference endpoint, understand how
production inference servers are structured, or run repeatable benchmarks against
small open-weight models without sending data to a cloud provider.

---

## Architecture overview

InferenceX is organized as a layered system. Requests flow inward through four layers,
with no layer skipping its neighbor:

```
HTTP request
     │
     ▼
API layer (FastAPI routes + error mapping)
     │
     ▼
Service layer (ChatService — routing, validation, orchestration)
     │
     ▼
Engine interface (BaseEngine — generate / generate_stream / is_healthy)
     │
     ▼
Engine implementation (VLLMEngine — wraps vLLM sync LLM API)
```

**API layer** (`src/inference_x/api/`) is thin: route handlers do nothing except call
a service and return the result. Error mapping converts `ValueError` → HTTP 400 and
`RuntimeError` → HTTP 500, sanitizing internal details before they reach clients.

**Service layer** (`src/inference_x/services/`) owns orchestration. `ChatService`
resolves the routing policy, enforces per-model token caps, and drives the engine. It
is unaware of HTTP — it speaks in typed domain objects.

**Engine interface** (`src/inference_x/engines/`) defines `BaseEngine` as an abstract
class. `VLLMEngine` implements it. The `EnginePool` holds one engine per loaded model
and is indexed by model name.

**Observability** (`src/inference_x/observability/`) is a Starlette middleware. It
records latency, token counts, and error flags for every request without touching any
route handler or service. Metrics live in an in-memory ring buffer (`deque`, capped at
1000 records). An optional `JsonLineExporter` writes NDJSON to disk.

**Routing** (`src/inference_x/routing/`) applies a chain of policies. An
`ExplicitModelPolicy` honors the `model` field in the request. A `DefaultModelPolicy`
falls back to the operator-configured default. The entire routing decision is in-process
and synchronous — zero extra HTTP round-trips.

---

## Phase by phase

### Phase 0 — Repo foundation

Before writing any code I set up the repo structure, documentation skeleton, and
OpenSpec workflow. The key decision here was to use `.cursor/rules/` for persistent
agent guidance and `AGENTS.md` for repo-level instructions. This proved useful
throughout — the agent consistently respected boundaries like "keep route handlers thin"
without repeated prompting.

### Phase 1 — Core inference engine

The first working endpoint: `POST /v1/chat/completions` and `GET /health`. Nothing else.

The most important decision was keeping vLLM in the main `[project.dependencies]`
rather than making it optional. Making it optional caused `uv sync` to uninstall it and
broke smoke tests (DEC-007). It was simpler to just have it present and mock the engine
in unit tests via FastAPI dependency overrides.

The health endpoint returns HTTP 200 / 503 / 500 — not 200 with a `status: degraded`
body. Load balancers and smoke scripts need a status code they can check, not a JSON
field they have to parse.

WSL2 with vLLM required a platform patch: vLLM defaults to pinned memory and FlashInfer
JIT compilation, both of which fail on WSL2 without special handling. I added
`utils/vllm_platform_patch.py` and `utils/cuda_env.py` to detect WSL and set the right
environment before any vLLM import.

First smoke test: `qwen2.5-0.5b`, 90 seconds cold start, chat reply `'Hello.'`, 38
tokens. That was the first real inference call and it felt significant.

**Validation:** 32/32 unit tests passing. Live smoke on WSL2 + CUDA succeeded.

### Phase 2 — Model registry and routing

Added `ModelRegistry` (config-driven from `models.yaml`), `TaskRouter`, and
`GET /v1/models`. Routing is a simple chain-of-responsibility: explicit model field
first, then default.

The key operational insight from this phase: fail at startup, not at first request.
`initialize_app()` in `deps.py` eagerly builds the registry, router, and engine before
accepting traffic. A misconfigured `default_model` crashes the process immediately with
a CRITICAL log, not silently on the first real request.

**Validation:** 63/63 unit tests passing.

### Phase 3 — Observability pipeline

Every request is timed, counted, and flagged for errors — without touching a single
route handler or service. The implementation is entirely in `ObservabilityMiddleware`.

The non-obvious part was response body buffering. `BaseHTTPMiddleware` exposes responses
as a streaming `body_iterator`. Once you consume it for token extraction, the client
gets an empty response. The fix: consume all bytes, parse JSON, then construct a new
`Response` object with the same bytes, headers (minus `content-length`), and status.
Starlette recalculates `content-length` automatically.

The in-memory ring buffer (`collections.deque`, capped at 1000 records) was a
deliberate choice. No I/O in the request path, no external dependencies, trivially
testable. Data doesn't survive restarts, which is fine for a dev server.

**Validation:** 94/94 unit tests passing.

### Phase 4 — Playground and evaluation

I built two playground interfaces: a batch CLI (`playground/client.py`) using only
stdlib, and an interactive Textual TUI (`playground/app.py`).

The TUI uses `httpx` async streaming to display tokens in real time. This was the first
time the platform felt like something you would actually want to demo — watching tokens
arrive token by token in a terminal is more compelling than a curl response.

The multi-model compare mode required replacing the single-engine design with an
`EnginePool`. `INFERENCE_X_LOADED_MODELS=qwen2.5-0.5b,tinyllama-chat` loads both
models at startup and the playground can compare them side by side in a single server
process. vLLM's `gpu_memory_utilization` is split automatically between engines.

The startup model-select screen (`playground/startup_screen.py`) fetches `/v1/models`
on mount and presents a `RadioSet`. No env var required — the server tells the playground
what is available.

**Validation:** 158+ unit tests passing (playground covered with mocked HTTP; TUI tested
manually).

### Phase 5 — Hardening and publication

Security pass: schema constraints (max 32k content, ≤4096 max_tokens, temperature
0.0–2.0), sanitized error responses, default bind to `127.0.0.1`, `config/logging.yaml`
with rotating file handler, SSRF guard in the playground URL validator.

The three items that did not make Phase 5 — authentication, rate limiting, and a
streaming timeout — are documented in `DECISIONS.md` as DEC-DEFER-01/02/03. The
streaming timeout was subsequently implemented using `asyncio.wait_for` around the
per-token `gen.__anext__()` call, controlled by `INFERENCE_X_STREAM_TIMEOUT_S`.

A `pip-audit` run found one CVE: `CVE-2025-69872` in `diskcache 5.6.3`, which is a
transitive vLLM dependency not directly used by InferenceX. Accepted under local-only
deployment assumptions.

---

## Benchmark results

### Methodology

The benchmark suite (`benchmarks/prompts/standard.json`) is a fixed set of 10 prompts
across five categories: short factual questions (×3), long generation (×2), code
generation (×2), reasoning/math (×2), and chat/roleplay (×1). The suite is SHA256-versioned
so results from different runs are directly comparable.

For each prompt the runner:
1. POSTs to `POST /v1/chat/completions` with `stream=True`
2. Records **time-to-first-token (TTFT)** — time from request send to first non-empty token chunk
3. Records **total latency** — wall time from request send to `[DONE]`
4. Counts tokens from whitespace-split of streamed content (client-side estimate)
5. Computes **tokens/sec** from total latency

Aggregate metrics per run: p50/p95/p99 of total latency, mean throughput. VRAM delta
is measured by calling the hardware profiler before and after the run (free VRAM
difference). If no GPU is detected, VRAM delta is 0.

The **Model Advisor** ranks results with a weighted composite score (0–100):
- 40% throughput (normalized against fastest model in the set)
- 30% TTFT (inverted — lower is better)
- 20% VRAM headroom (remaining free VRAM after model load)
- 10% quantization fit (placeholder — always 1.0 until INT8/FP8 data is added)

A model that requires more VRAM than is free receives score=0 and is marked `viable=False`.

Sample run (2026-06-08, WSL2, single model loaded):

| Model | Mean throughput (tok/s) | p50 latency (ms) | p95 latency (ms) | p99 latency (ms) | VRAM delta (GB) |
|-------|-------------------------|------------------|------------------|------------------|-----------------|
| qwen2.5-0.5b | 175.6 | 821 | 1224 | 1225 | 0.0 |

> VRAM delta was 0 on this run because the hardware profiler fell back to CPU-only
> detection in the test environment. Install optional deps with `uv sync --extra hardware`
> for accurate VRAM readings on CUDA hosts.

Run additional models with:
```bash
INFERENCE_X_LOADED_MODELS=qwen2.5-0.5b,tinyllama-chat ./scripts/dev.sh serve
make benchmark-all
make advise
```

Advisor ranking (populate after `make advise` on your hardware):

```
[TODO: paste make advise output]
```

---

## What I'd do differently

**Authentication first.** Binding to 127.0.0.1 is a practical shortcut for local dev,
but it means any process on the same machine can hit the API without credentials. This
is a known gap (DEC-DEFER-01) that needs addressing before any networked deployment.

**Async metrics writes.** `MetricsRecorder.record()` runs synchronously in the request
path. For the current scale (single user, low QPS) this is fine — the lock+append
benchmarks at ~0.14 µs. But a background metrics writer would be the right design for
anything resembling production load.

**Optional vLLM sooner.** I went back and forth on whether vLLM should be optional or
required (DEC-005/007). The right answer depends on whether you want the test suite to
run without a GPU. I landed on "required" because making it optional broke `uv sync`.
A cleaner solution would be a proper mock-or-skip strategy at the integration test level.

**Config validation at boot.** The settings loader validates at construction time, but
config errors in `models.yaml` (wrong type, missing field) surface as generic YAML
parse errors, not as actionable messages. Stricter Pydantic models for config files
would improve the operator experience.

**The FlashInfer WSL2 problem.** Setting `VLLM_USE_FLASHINFER_SAMPLER=0` works, but it
is a workaround for a real incompatibility. FlashInfer's JIT compilation assumes a
system CUDA toolkit at a fixed path — a poor assumption for any non-standard Linux
environment. This burned several hours and belongs in a troubleshooting document
(which it now is, in `README.md`).

---

## Conclusion

InferenceX is now a functional self-hosted inference platform: one command to start the
server, one command to launch the interactive playground, and a clean API surface that
is compatible with any OpenAI client. It measures what it does (observability), explains
what it loads (model registry), and is safe to expose on a development machine
(127.0.0.1 bind, sanitized errors, input constraints).

Phase 6 adds the benchmark suite and model advisor — the part that answers the question
"which model should I actually use on this hardware?" That data will make this a more
useful tool and a more interesting article.

The full source is at `src/inference_x/` and the OpenSpec change history is in
`openspec/changes/`.
