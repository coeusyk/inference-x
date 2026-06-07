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

- **Change**: Implemented the first runnable vLLM-backed inference path end-to-end.
- **Files added/written**:
  - `src/inference_x/schemas/chat.py` — typed OpenAI-compatible request/response models (Pydantic v2)
  - `src/inference_x/schemas/common.py` — shared `ErrorResponse` / `ErrorDetail` models
  - `src/inference_x/engines/base.py` — `BaseEngine` abstract interface (`generate`, `is_healthy`)
  - `src/inference_x/engines/vllm_engine.py` — vLLM adapter; import-guarded for GPU-less dev
  - `src/inference_x/core/settings.py` — `AppSettings` env+yaml config loader; `get_settings()` cached singleton
  - `src/inference_x/services/chat_service.py` — thin orchestration layer; wraps engine
  - `src/inference_x/api/deps.py` — FastAPI dependency factories (engine + service singletons)
  - `src/inference_x/api/errors.py` — exception handlers for `RuntimeError` / `ValueError`
  - `src/inference_x/api/routes/chat_completions.py` — `POST /v1/chat/completions`
  - `src/inference_x/api/routes/health.py` — `GET /health`
  - `src/inference_x/api/main.py` — FastAPI app wiring with exception handlers
  - `tests/unit/` — 28 unit/contract tests (schemas, engine interface, service, settings, routes)
  - `scripts/smoke_test.py` — standalone HTTP smoke test (stdlib only)
- **Why**: First runnable slice proves the architectural layers hold and gives a testable baseline before routing/observability work.
- **Tradeoffs**:
  - vllm stays in main `[project.dependencies]` — Phase 1 requires it; `uv sync` must install the full stack.
  - Dev deps live in `[dependency-groups] dev` (uv includes them on sync by default).
  - Run server and scripts with `uv run` — bare `uvicorn`/`python` from pyenv bypasses `.venv` and breaks vllm import.
- **Validation**: `python -m pytest tests/unit/ -v` → **28/28 passed** (no GPU required; vLLM mocked via stub engine + FastAPI dependency override).
- **Server start command**: `uvicorn inference_x.api.main:app --host 0.0.0.0 --port 8000`
- **Model selection**: set `INFERENCE_X_DEFAULT_MODEL=<name>` where name is a key in `config/models.yaml` (default: `qwen2.5-0.5b`)
- **Smoke test command**: `python scripts/smoke_test.py` (requires running server)
