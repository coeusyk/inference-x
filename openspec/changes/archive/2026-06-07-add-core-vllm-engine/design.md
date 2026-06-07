# Design: add-core-vllm-engine

## Overview

This change introduces the minimal production-shaped backend path for InferenceX.

## Design goals

- keep route handlers thin
- keep the engine behind a stable interface
- load all runtime values from config or environment
- make the result smoke-testable in WSL2

## Planned structure

- `src/inference_x/engines/base.py` defines the engine interface
- `src/inference_x/engines/vllm_engine.py` adapts vLLM to the interface
- `src/inference_x/schemas/chat.py` defines the public request and response models
- `src/inference_x/api/routes/chat_completions.py` exposes the chat endpoint
- `src/inference_x/api/routes/health.py` exposes health checks
- `src/inference_x/services/chat_service.py` coordinates request execution
- `config/models.yaml` and `config/server.yaml` define runtime behavior

## Validation

- unit tests for schema and service logic
- smoke test for endpoint behavior
- integration check with a small local model or mocked engine path

## Compatibility rule

This change should not force later routing or observability decisions into the API surface.