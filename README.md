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
- Python 3.11+
- vLLM for inference

## Next step

Bootstrap the Phase 1 backend skeleton under `src/inferencex/` and wire up the first engine-backed endpoint.