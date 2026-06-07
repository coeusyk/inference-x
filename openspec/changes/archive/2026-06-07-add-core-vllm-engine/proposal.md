# Proposal: add-core-vllm-engine

## Summary

Implement the first runnable inference slice for InferenceX: one vLLM-backed model exposed through an OpenAI-compatible API.

## Why

The project needs a real backend before routing, observability, or UI work can be justified. This change creates the first stable execution path.

## In scope

- engine interface
- vLLM engine implementation
- chat request and response schemas
- `/v1/chat/completions`
- `/health`
- config loading for one model
- smoke test path

## Out of scope

- multi-model routing
- metrics dashboards
- playground UI
- second engine implementation
- any contract beyond the initial chat and health endpoints