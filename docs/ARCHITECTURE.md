# Architecture

## System Overview

A modular LLM serving platform built incrementally in phases.

## Design Principles

1. **Interface stability**: Core interfaces never change
2. **Config-driven**: New features added via YAML, not code edits
3. **Layered architecture**: Each phase adds a layer without touching previous ones
4. **Independent modules**: Each folder is self-contained

## Current Architecture (Phase 1)

```
┌─────────────┐
│ Client │
└──────┬──────┘
│ HTTP
┌──────▼──────────┐
│ FastAPI App │ (src/api/main.py)
└──────┬──────────┘
│
┌──────▼──────────┐
│ VLLMEngine │ (src/core/engine.py)
└──────┬──────────┘
│
┌──────▼──────────┐
│ vLLM Server │
└─────────────────┘
```

## Planned Evolution

### Phase 2: Routing Layer
- Add `src/routing/` that sits between API and Engine
- Router implements same `generate()` interface
- API layer unchanged

### Phase 3: Observability
- Add middleware to FastAPI app
- No changes to existing endpoints or engine

### Phase 4: Playground
- Separate frontend consuming existing API
- Zero backend changes

## Module Responsibilities

### src/core/ (Phase 1)
- `engine.py`: Abstract LLMEngine interface + VLLMEngine implementation
- `schemas.py`: Pydantic models for requests/responses

### src/api/ (Phase 1)
- `main.py`: FastAPI app initialization
- `v1/completions.py`: OpenAI-compatible endpoints

### config/ (Phase 1)
- `models.yaml`: Model definitions
- `server.yaml`: Port, host, CORS settings

## Adding New Features

**To add a new model**: Edit `config/models.yaml` only
**To add routing**: Create `src/routing/`, modify dependency injection
**To add observability**: Add middleware to `main.py`
**To add UI**: Create `playground/` and consume existing API