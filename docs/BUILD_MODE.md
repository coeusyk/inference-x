# InferenceX Build Mode

## Purpose
Use this mode when generating code, folder structures, docs, and OpenSpec artifacts for InferenceX.

## Operating principles
- Build one vertical slice at a time.
- Prefer files that can be filled incrementally instead of speculative scaffolding.
- Keep API contracts explicit and typed.
- Keep route handlers thin and push logic into services and engine modules.
- Use WSL2-friendly commands and Linux paths for local development.

## Implementation order
1. Establish repo instructions and planning docs.
2. Initialize OpenSpec and create the first change.
3. Implement the Phase 1 backend skeleton.
4. Add tests and smoke checks.
5. Document decisions and benchmark results.
6. Expand to routing, observability, and playground only after Phase 1 is stable.

## File boundaries
- `AGENTS.md`: root agent instructions.
- `.cursor/rules/*`: persistent Cursor behavior rules.
- `docs/ARCHITECTURE.md`: system layout and dependency direction.
- `docs/PHASES.md`: execution roadmap.
- `docs/DECISIONS.md`: non-obvious design choices.
- `openspec/`: proposal, spec, design, and task artifacts.
- `src/inference_x/`: actual application code.
- `config/`: environment and runtime configuration.
- `tests/`: unit, integration, and contract tests.

## Code generation preferences
- Use Python 3.11+.
- Use Pydantic models for request and response schemas.
- Use dependency injection for engines and services.
- Keep filenames stable once created.
- Avoid premature abstraction unless a second implementation exists.

## Phase 1 target
Build only the first runnable slice:
- config loading
- engine interface
- vLLM engine implementation
- chat completion route
- health route
- basic smoke test