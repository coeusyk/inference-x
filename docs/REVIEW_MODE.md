# InferenceX Review Mode

Use this mode when reviewing changes before merge or when validating phase exit criteria.

## Verdict (2026-06-24)

Phases 0-6 complete. Core API, routing, observability, playground, hardening, and benchmark/advisor are implemented and covered by unit tests.

## What to verify on each review

- **API contract:** `POST /v1/chat/completions`, `GET /health`, `GET /v1/models`, benchmark read routes unchanged unless versioned.
- **Layer boundaries:** Routes stay thin; orchestration in services; inference in engines; routing in `routing/`; metrics in observability middleware only.
- **Config vs runtime:** Model list from `config/models.yaml`; default model from `INFERENCE_X_DEFAULT_MODEL`; loaded engines from `INFERENCE_X_LOADED_MODELS`. `config/routing.yaml` is documentation-only for the default model name.
- **Tests:** `uv run pytest tests/unit -v` must pass without a GPU (stub engines via dependency overrides).
- **Docs:** Contract or behavior changes update `docs/DECISIONS.md` and relevant OpenSpec deltas.

## Common gaps to watch for

- Bare `python` / `uvicorn` outside `uv run` (missing vLLM in PATH).
- Second vLLM process on the same GPU (KV cache OOM).
- Gated models without `HF_TOKEN` or HuggingFace approval.
- Playground base URLs without `--allow-internal` for localhost/private IPs.
- Benchmark CLI invoked while the target model is not in the running server's loaded pool.

## Review checklist

1. Read the change proposal or PR summary against `docs/PHASES.md` scope.
2. Confirm affected routes have unit or contract-style tests.
3. Run unit tests locally.
4. For engine or config changes, note whether a smoke test or live benchmark is required.
5. Ensure no secrets (.env, tokens) in the diff.

## Related docs

- `docs/BUILD_MODE.md` — implementation workflow
- `docs/ARCHITECTURE.md` — layer map and dependency direction
- `AGENTS.md` — agent working agreement
