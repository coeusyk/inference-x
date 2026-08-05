# InferenceX Architecture

## Purpose

InferenceX is a self-hosted inference runtime with an OpenAI-compatible API.

Today, vLLM is the sole supported inference backend.

The runtime owns execution policy while inference backends remain replaceable
implementation details over the lifetime of the project (DEC-047).

The project evolves incrementally from a stable chat-completions runtime toward
a more capable local inference platform without requiring large architectural
rewrites.

## Document authority

Normative architecture lives in, in this order:

1. `docs/DECISIONS.md`
2. `docs/ARCHITECTURE.md`
3. `CONTRIBUTING.md`
4. `AGENTS.md`

`docs/UNDERSTANDING-INFERENCE-X.md` describes the current codebase. Architecture
review documents under `docs/` are historical artifacts, not policy.

## Architecture goals

- Keep the external API stable while internal modules evolve.
- Add capabilities by adding modules and layers rather than replacing earlier code.
- Make core behavior configuration-driven.
- Keep business logic out of route handlers.
- Make each phase independently runnable and testable.

## System layers

### 1. API layer

The API layer is responsible for HTTP concerns only:
- request parsing and validation
- dependency resolution
- response serialization
- mapping exceptions to HTTP errors

This layer lives under `src/inferencex/api/`.

### 2. Service layer

The service layer coordinates application behavior:
- converting API requests into engine calls
- orchestrating routing decisions
- invoking observability recorders
- returning normalized outputs to the API layer

This layer lives under `src/inferencex/services/`.

### 3. Engine layer

The engine layer defines how inference is performed:
- a stable engine interface in `engines/base.py`
- a first implementation in `engines/vllm_engine.py`
- later support for additional engines if needed

The rest of the system should depend on the engine interface, not directly on vLLM internals.

#### Engine Boundary (DEC-047)

- The runtime owns architectural / execution policy; backends own inference execution.
- vLLM (`engines/vllm_engine.py`) is the only concrete backend today.
- App construction should go through `engines/registry` (`create_engine`); route
  handlers must not import concrete engines.
- Backend-neutral abstractions must not be introduced until justified by at least
  two concrete backend implementations. Do not add `inference_x/execution/` ahead
  of that. Thin factory and durable capability declarations are hygiene, not a
  gate on AsyncLLM (Phase B).

### 4. Routing layer

The routing layer is optional in Phase 1 and introduced in Phase 2. It decides which model or engine should serve a request based on policy, explicit model selection, or request metadata.

This layer lives under `src/inferencex/routing/`.

### 5. Observability layer

The observability layer records system behavior without polluting business logic:
- latency
- token counts
- errors
- request metadata
- optional exporters or storage adapters

This layer lives under `src/inferencex/observability/`.

### 6. Benchmark layer (Phase 6)

The benchmark layer measures inference performance and recommends models for local hardware:
- hardware profiling (`benchmarks/hardware.py`) — GPU VRAM via pynvml or nvidia-smi
- benchmark runner (`benchmarks/runner.py`) — streaming HTTP against `/v1/chat/completions`;
  records VRAM footprint and a per-run `hardware` snapshot in each result
- result storage (`benchmarks/storage.py`) — JSON files under `benchmarks/results/`
- model advisor (`benchmarks/advisor.py`) — weighted scoring (throughput, warm TTFT, VRAM
  headroom); viability gate compares footprint + 0.5 GB buffer to total VRAM; exact GPU
  name match for hardware provenance; returns `AdvisorReport` with warnings

This layer lives under `src/inference_x/benchmarks/`. It does not call vLLM directly;
the benchmark CLI requires a running server.

## Dependency direction

The allowed dependency flow is:

`API -> Services -> Interfaces -> Implementations`

Practical interpretation:
- routes can depend on services and schemas
- services can depend on engine and routing interfaces
- concrete engine implementations should not be imported directly into route handlers
- observability should be attached through services, middleware, or explicit adapters

## Core repository structure

```text
src/inferencex/
├── api/
│   ├── main.py
│   ├── deps.py
│   ├── errors.py
│   └── routes/
├── core/
├── schemas/
├── engines/
├── routing/
├── observability/
├── benchmarks/
├── services/
└── utils/
```

### Module responsibilities

#### `api/`
Exposes HTTP endpoints and FastAPI wiring.

#### `schemas/`
Contains typed request and response models only.

#### `services/`
Contains use-case-oriented orchestration code.

#### `engines/`
Contains the abstract engine contract, construction registry/factory, and
concrete inference implementations.

#### `routing/`
Contains model-selection policies and future fallback logic.

#### `observability/`
Contains middleware, metrics recorders, storage adapters, and exporters.

#### `benchmarks/`
Contains hardware profiling, benchmark runner, result storage, and model advisor logic.

#### `core/`
Contains settings, lifecycle hooks, and shared app setup.

#### `utils/`
Contains small reusable helpers that do not own domain behavior.

## Runtime environments

### Development environment
- Windows host
- WSL2 Ubuntu environment
- Cursor as the editor
- Python 3.11+
- NVIDIA GPU passed into WSL2 for vLLM runtime

### Runtime assumptions
- Local development uses Linux tooling inside WSL2
- vLLM runs inside WSL2, not native Windows
- future deployment may target a Linux host or containerized setup

## Incremental growth rules

New features should fit one of these extension paths:
- new endpoint under `api/routes/`
- new schema under `schemas/`
- new service under `services/`
- new engine implementation under `engines/`
- new policy under `routing/`
- new recorder, middleware, or exporter under `observability/`

If a feature does not fit this model, document the reason before changing the structure.

## Phase 1 stable contract

Phase 1 guarantees:
- `POST /v1/chat/completions`
- `GET /health`
- typed request and response schemas
- one configured vLLM-backed model
- smoke-testable local execution

## Future evolution

### Phase 2
Add a model registry and routing policies without breaking the chat completions contract.

### Phase 3
Add observability middleware and metrics storage without moving business logic back into routes.

### Phase 4
Add a lightweight playground that consumes the existing API rather than introducing UI-specific backend logic where avoidable.

### Phase 5
Add hardening, deployment guidance, and publication-ready artifacts.

### Phase 6
Add benchmark runner, hardware profiler, and model advisor (CLI + read-only API).
Additive API: `GET /v1/benchmark/results`, `GET /v1/benchmark/advise` (includes `warnings`).