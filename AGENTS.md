# OpenSpec Agent Instructions for InferenceX

This repository uses OpenSpec for spec-driven development.

## Working agreement

- Do not implement non-trivial work without an active change under `openspec/changes/`.
- Before making changes, read:
  - `docs/ARCHITECTURE.md`
  - `docs/PHASES.md`
  - `docs/DECISIONS.md`
  - `docs/REVIEW_MODE.md`
  - `docs/BUILD_MODE.md`
  - relevant files in `openspec/specs/`
  - active change artifacts in `openspec/changes/<change-id>/`
- For each change, work in this order:
  1. `proposal.md`
  2. spec deltas under `specs/`
  3. `design.md`
  4. `tasks.md`
  5. implementation
  6. validation
  7. archive

## Repo-specific rules

- InferenceX is a backend-first, layered LLM inference platform.
- Keep the public API stable.
- Prefer additive changes over breaking rewrites only when they genuinely reduce risk.
- Keep route handlers thin and move orchestration to services.
- Use typed schemas and config-driven behavior.
- Develop for WSL2 Ubuntu as the primary local runtime.
- Do not over-index on tiny phases if it slows the project down without reducing risk.
- Prefer one clear vertical slice for a feature rather than excessive pre-scaffolding.

## File boundaries

- `docs/` contains architecture, phase, decision, review, build, and article notes docs only.
- `openspec/` contains specs, proposals, designs, tasks, and change deltas only.
- `config/` contains environment and runtime configuration only.
- `src/inference_x/api/` contains FastAPI wiring, routes, dependencies, and error mapping.
- `src/inference_x/services/` contains application orchestration and use-case logic.
- `src/inference_x/engines/` contains engine interfaces and inference implementations.
- `src/inference_x/routing/` contains model selection policies and routing logic.
- `src/inference_x/observability/` contains middleware, metrics, storage, and exporters.
- `src/inference_x/schemas/` contains request/response models only.
- `src/inference_x/benchmarks/` contains hardware profiling, benchmark runner, storage, and advisor logic.
- `benchmarks/prompts/` contains versioned benchmark prompt suites (not Python code).
- `tests/` contains unit, integration, and contract tests only.

## Phase 1 contract

Phase 1 must remain focused on one stable inference path.

Required endpoints (Phase 1):
- `POST /v1/chat/completions`
- `GET /health`

Additional endpoints (later phases, additive only):
- `GET /v1/models` (Phase 2)
- `GET /v1/benchmark/results`, `GET /v1/benchmark/advise` (Phase 6)

Phase 1 expectations:
- one configured vLLM-backed model
- typed request validation
- typed response formatting
- clear error handling for configuration, model load, and inference failures
- smoke-testable local execution in WSL2

## Anti-scope rules

Do not add these before the relevant phase:
- model routing policies beyond a minimal Phase 1 dependency
- observability dashboards or storage backends
- playground UI work
- multiple engine implementations
- premature abstractions that do not yet support a second implementation
- config sprawl without validation

## Validation rules

- Every non-trivial change must include tests or a smoke check.
- Contract changes must update specs and docs in the same change.
- Engine changes must be validated against the Phase 1 contract.
- If a change cannot be validated, do not mark it complete.

## Article notes

- After completing any phase milestone, architectural decision, benchmark, or noteworthy implementation detail, add a note to `docs/ARTICLE_NOTES.md`.
- Notes should be short, factual bullets — not polished prose.
- Do not wait until the end of the project to write notes.
- Capture: what changed, why, what tradeoff was accepted, validation run, and any useful numbers or commands.

## Phase discipline

- Follow `docs/PHASES.md`.
- Do not let phase planning block obvious, low-risk progress.
- Record non-obvious design choices in `docs/DECISIONS.md`.
