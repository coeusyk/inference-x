# Design: add-model-registry-routing

## Overview

Introduce a routing layer between API and engine selection without changing the core chat endpoint shape.

## Design goals

- preserve the chat completion contract
- keep routing declarative where possible
- support future fallback behavior
- avoid coupling routing logic to route handlers

## Planned structure

- `src/inference_x/routing/base.py`
- `src/inference_x/routing/task_router.py`
- `src/inference_x/routing/policies.py`
- `config/routing.yaml`
- `src/inference_x/services/model_service.py`

## Validation

- unit tests for routing decisions
- config-driven policy verification
- contract checks to ensure the API remains stable