# Design: bootstrap-phase-1

## Overview

This change does not implement backend logic. It establishes OpenSpec as the planning system for InferenceX and creates the first implementation-ready Phase 1 change structure.

## Design goals

- align planning with the existing architecture and phases docs
- make OpenSpec the default workflow for non-trivial work
- prepare a small, clear first execution path for the backend

## Planned Phase 1 backend slice

The first implementation slice should introduce:
- `src/inference_x/engines/base.py`
- `src/inference_x/engines/vllm_engine.py`
- `src/inference_x/schemas/chat.py`
- `src/inference_x/api/main.py`
- `src/inference_x/api/routes/chat_completions.py`
- `src/inference_x/api/routes/health.py`
- minimal config files for server and model definition
- a smoke test path

## Why this approach

A small vertical slice gives the project a runnable milestone quickly while preserving room for routing and observability as later additive layers.

## Artifact model

This repo should use OpenSpec in the following way:
- `openspec/specs/` holds durable source-of-truth platform specs
- `openspec/changes/<change-id>/proposal.md` explains why the change exists
- `openspec/changes/<change-id>/specs/` stores spec deltas
- `openspec/changes/<change-id>/design.md` explains architecture impact
- `openspec/changes/<change-id>/tasks.md` drives implementation execution

## Recommended future changes

After this bootstrap, create separate changes for:
- `add-core-vllm-engine`
- `add-model-registry-routing`
- `add-observability-pipeline`
- `add-playground-eval`