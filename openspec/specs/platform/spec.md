# Platform Specification

## Purpose

InferenceX is a self-hosted inference platform that exposes a stable API for LLM serving and grows incrementally from a single vLLM-backed engine into a more complete system with routing, observability, and evaluation support.

## Requirements

### Requirement: Stable API-first platform
The system SHALL expose stable HTTP interfaces before layering on optional capabilities.

#### Scenario: Initial stable API
- **WHEN** the first backend milestone is implemented
- **THEN** the system provides a health endpoint and a chat completions endpoint
- **AND** request and response contracts are documented and validated

### Requirement: Incremental architecture
The system SHALL support additive growth without requiring large structural rewrites.

#### Scenario: New capability is added
- **WHEN** routing, observability, or a playground feature is introduced
- **THEN** the existing API contract remains stable unless explicitly versioned
- **AND** the new capability is added within the appropriate module boundary

### Requirement: WSL2-first development
The system SHALL assume local development on WSL2 Ubuntu for Linux-only runtime components.

#### Scenario: Developer setup
- **WHEN** local setup instructions are followed
- **THEN** the runtime path targets Linux tooling inside WSL2
- **AND** Windows-native execution is not required for vLLM-bound components

### Requirement: Spec-driven changes
The project SHALL track non-trivial work through OpenSpec change artifacts.

#### Scenario: New feature is proposed
- **WHEN** a non-trivial feature or refactor is requested
- **THEN** a change folder is created under `openspec/changes/`
- **AND** the proposal, design, and task artifacts are completed before implementation proceeds