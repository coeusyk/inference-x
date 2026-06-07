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

### DEC-008
- Date: 2026-06-07
- Status: accepted
- Context: Phase 1 exit review found `/health` returned HTTP 200 with `status: degraded` when the engine health flag was false.
- Decision: Return HTTP 503 when `is_healthy()` is false; return HTTP 500 when engine initialization raises `RuntimeError`; return HTTP 200 only when healthy.
- Consequences: Load balancers and smoke tests can distinguish unavailable from ready. Body still includes `status` and `engine` fields.

### DEC-009
- Date: 2026-06-07
- Status: accepted
- Context: Phase 1 needs a health signal without running inference on every poll.
- Decision: Set `_healthy = True` only after successful vLLM `LLM()` init; `is_healthy()` returns that flag with no live generation.
- Consequences: Health reflects init-time readiness, not runtime degradation after load. Acceptable for Phase 1; runtime probes can be added in observability phase.

### DEC-010
- Date: 2026-06-07
- Status: accepted
- Context: Request schema accepts `stream: bool` but Phase 1 has no SSE implementation.
- Decision: Reject `stream=true` in `ChatService.complete()` with `ValueError` → HTTP 400 and structured error body.
- Consequences: Clients get an explicit error instead of a misleading non-streaming 200 response.

---

## Phase 2 decisions

### DEC-011
- Date: 2026-06-07
- Status: accepted
- Context: Phase 2 adds a registry and router but the hardware is still one GPU / one loaded engine.
- Decision: `ChatService` resolves a model name via `TaskRouter` then validates it against `loaded_model`. If the routed model differs from the loaded engine, a `ValueError` → HTTP 400 is returned with a clear restart instruction.
- Consequences: Multi-model serving is explicitly deferred. Clients are told exactly what model is loaded and how to switch. No silent wrong-model responses.

### DEC-012
- Date: 2026-06-07
- Status: accepted
- Context: Choosing where to validate the `default_model` setting (startup vs. first request).
- Decision: Validate at `TaskRouter` and `DefaultModelPolicy` construction — not at `ChatService.complete()`. If the default model is absent from the registry, the app fails to start with a clear `ValueError`.
- Consequences: Misconfiguration surfaces immediately on startup, not mid-request. Operationally safer; no request can succeed against a missing default model.

### DEC-013
- Date: 2026-06-07
- Status: accepted
- Context: `GET /v1/models` was optional in the proposal.
- Decision: Include it. It costs one thin route handler and two Pydantic models. It lets clients enumerate registered models without reading `models.yaml` directly and removes the need for out-of-band documentation.
- Consequences: Adds one endpoint to the public contract. Follows OpenAI API shape (`object: "list"`, `data: [...]`).

### DEC-014
- Date: 2026-06-07
- Status: accepted
- Context: `ModelRegistry` could be derived from `AppSettings.get_model_config()` or owned separately.
- Decision: Give `ModelRegistry` its own `from_config(config_dir)` factory that reads `models.yaml` and validates all entries via Pydantic. `AppSettings` retains `get_model_config` for backward compat but is no longer the canonical registry.
- Consequences: Registry is independently testable with a `tmp_path` fixture. Settings retains minimal config surface for env vars.