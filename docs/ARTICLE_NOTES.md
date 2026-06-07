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
