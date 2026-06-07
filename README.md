# InferenceX

InferenceX is a self-hosted LLM inference platform built incrementally on top of vLLM.

## Current direction

The project begins with a stable OpenAI-compatible chat completions API and expands in phases:
- Phase 1: core vLLM-backed inference
- Phase 2: model registry and routing
- Phase 3: observability
- Phase 4: playground and evaluation
- Phase 5: hardening and publication readiness

## Key docs

- `docs/ARCHITECTURE.md`
- `docs/PHASES.md`
- `docs/DECISIONS.md`
- `.cursor/rules/`

## Development environment

- Windows host
- WSL2 Ubuntu for runtime and development
- Cursor as the editor
- Python 3.13+ (managed by uv)
- vLLM for inference

## Quick start (WSL2)

```bash
# Install dependencies (vllm is a main dependency — always installed by sync)
uv sync

# Terminal 1 — start server (must use uv run so .venv Python is used)
INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b uv run uvicorn inference_x.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — smoke test
uv run python scripts/smoke_test.py
```

Or use `./scripts/dev.sh sync`, `./scripts/dev.sh serve`, `./scripts/dev.sh smoke`.

**Important:** Do not run bare `uvicorn` or `python` from pyenv/shims — that bypasses the project `.venv` and vllm will appear missing even after `uv sync`.

## Status

- Phase 1 core inference slice is implemented. See `docs/PHASES.md` for exit criteria.