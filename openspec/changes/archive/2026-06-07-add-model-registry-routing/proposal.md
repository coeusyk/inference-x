# Proposal: add-model-registry-routing

## Summary

Add a model registry and routing layer so InferenceX can select models by policy without changing the core API contract.

## Why

A single model is enough for the first slice, but the platform value increases when it can route requests intentionally based on task type or model choice.

## In scope

- model registry
- routing configuration
- routing interface
- basic route selection policies
- optional model listing endpoint if justified by the contract

## Out of scope

- observability storage
- UI playground work
- premature ML optimization beyond routing policy
- engine rewrites unrelated to routing