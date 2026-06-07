# InferenceX Decisions

Use this document to capture non-obvious design decisions as the project evolves.

## Decision template

### DEC-000
- Date:
- Status: proposed | accepted | superseded
- Context:
- Decision:
- Consequences:

## Initial decisions

### DEC-001
- Date: 2026-06-07
- Status: accepted
- Context: The project needs a clear first milestone and must remain incremental.
- Decision: Start with one vLLM-backed OpenAI-compatible chat completions API before adding routing, observability, or UI.
- Consequences: Early progress stays measurable and later phases can build on a stable contract.

### DEC-002
- Date: 2026-06-07
- Status: accepted
- Context: Development is happening on Windows, while vLLM requires Linux tooling.
- Decision: Standardize local development on WSL2 Ubuntu and treat it as the primary runtime environment.
- Consequences: Commands, setup steps, and scripts should target Linux inside WSL2 first.

### DEC-003
- Date: 2026-06-07
- Status: accepted
- Context: The repo must support agent-assisted development in Cursor.
- Decision: Use `.cursor/rules/` for persistent repository guidance and keep the rules split by concern.
- Consequences: Repo-level instructions remain concise and easier for the editor to apply consistently.

### DEC-004
- Date: 2026-06-07
- Status: accepted
- Context: The vLLM engine must be importable without a GPU for local dev and testing.
- Decision: Guard `from vllm import ...` inside a try/except in `vllm_engine.py`; raise a clear `RuntimeError` at instantiation time if vllm is absent.
- Consequences: Tests and schema validation work without vLLM. The engine fails fast with a clear message when vLLM is missing at runtime.

### DEC-005
- Date: 2026-06-07
- Status: superseded
- Context: `pyproject.toml` listed `requires-python = ">=3.13"` but WSL2 runtime is Python 3.12.
- Decision: Change to `>=3.11` and separate `vllm` into an optional extra (`[vllm]`) so dev installs don't require GPU wheels.
- Consequences: Dev setup is lighter; production installs must use `pip install inferencex[vllm]`.
- Superseded by: DEC-007 (2026-06-07) — optional vllm broke `uv sync` and Phase 1 smoke tests.

### DEC-007
- Date: 2026-06-07
- Status: accepted
- Context: Moving vllm to `[project.optional-dependencies]` caused `uv sync` to uninstall vllm; smoke test returned HTTP 500 "vllm is not installed".
- Decision: Keep vllm in main `[project.dependencies]` (Phase 1 is vLLM-backed). Use `[dependency-groups] dev` for test deps (uv includes dev group by default on sync). Keep `requires-python = ">=3.13"` to match the uv-managed `.venv`.
- Consequences: `uv sync` restores the full runtime stack including vllm. CI without GPU must use a separate strategy (mock engine / skip integration), not drop vllm from default deps.

### DEC-006
- Date: 2026-06-07
- Status: accepted
- Context: `core/engine.py` and `core/schemas.py` contained implementation code in the wrong layer.
- Decision: Migrate all engine logic to `engines/` and schema models to `schemas/` per the architecture spec. Legacy stub files left in place; remove in a cleanup change.
- Consequences: Dependency flow API → Services → Interfaces → Implementations is now enforced by module layout.