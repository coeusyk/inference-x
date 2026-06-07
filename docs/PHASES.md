# InferenceX Phases

## Phase 0 - Repository foundation

### Goal
Prepare the repository for disciplined, incremental development in Cursor on WSL2.

### Deliverables
- `.cursor/rules/` ruleset
- `docs/ARCHITECTURE.md`
- `docs/PHASES.md`
- `docs/DECISIONS.md`
- `pyproject.toml`
- initial folder structure
- README skeleton
- base configuration files
- optional OpenSpec initialization

### Exit criteria
- the repo opens cleanly in Cursor
- Python environment and dev commands are defined
- the project structure matches the architecture docs
- the first planned implementation change is written down

## Phase 1 - Core inference engine

### Goal
Deliver one stable vLLM-backed inference path through an OpenAI-compatible API.

### Deliverables
- engine interface in `src/inferencex/engines/base.py`
- vLLM implementation in `src/inferencex/engines/vllm_engine.py`
- chat schemas in `src/inferencex/schemas/chat.py`
- `POST /v1/chat/completions`
- `GET /health`
- smoke test script
- baseline benchmark notes

### Exit criteria
- the app starts locally in WSL2
- one configured model can answer a chat request
- request and response validation work correctly
- the health endpoint reflects engine status
- at least one integration or smoke test passes

## Phase 2 - Model registry and routing

### Goal
Support multiple configured models and route requests by policy or explicit model choice.

### Deliverables
- model registry
- routing configuration file
- routing interface and default task router
- optional `/v1/models` endpoint
- unit tests for routing behavior

### Exit criteria
- requests can target a configured model explicitly
- default routing policy works for supported cases
- fallback or failure behavior is documented
- the Phase 1 contract remains stable

## Phase 3 - Observability

### Goal
Capture operational metrics without changing public request contracts.

### Deliverables
- request timing middleware
- metrics recorder
- storage adapter or file-backed recorder
- metrics schema
- metrics endpoint or export path
- comparative benchmark notes

### Exit criteria
- latency, failures, and selected usage metrics are recorded
- instrumentation does not break the Phase 1 and 2 behavior
- one query or export path exists for inspection

## Phase 4 - Playground and evaluation

### Goal
Provide a lightweight UI or workflow to compare outputs and support demo usage.

### Deliverables
- minimal playground client
- compare flow for prompts across models or configs
- sample prompt set
- screenshots or demo notes

### Exit criteria
- the playground works against the existing API
- at least two model or config variants can be compared
- demo assets exist for documentation or article writing

## Phase 5 - Hardening and publication readiness

### Goal
Make the project reproducible, explainable, and ready for sharing.

### Deliverables
- stronger integration coverage
- setup verification steps
- cleaned documentation
- benchmark summaries
- deployment notes
- article-supporting notes and visuals

### Exit criteria
- a new developer can follow the README and run the project
- architecture, phases, and key decisions are documented
- the repo supports a strong technical article or portfolio write-up

## Execution rules

- Finish a thin vertical slice before broadening scope.
- Keep each phase deployable or runnable.
- Prefer additive change over breaking refactors.
- Update docs when a phase goal, contract, or structure changes.
- Do not begin a later phase before current exit criteria are met unless intentionally overriding the plan.