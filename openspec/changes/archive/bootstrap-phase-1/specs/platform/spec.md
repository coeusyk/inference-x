# Platform Specification Delta

## MODIFIED Requirements

### Requirement: Stable API-first platform
The Phase 1 bootstrap SHALL define the initial backend contract and file boundaries needed for implementation.

#### Scenario: Phase 1 planning is completed
- **WHEN** the bootstrap change is accepted
- **THEN** the repository contains a documented Phase 1 implementation plan
- **AND** the planned backend structure identifies config, schemas, engine, routes, and tests
- **AND** the first stable endpoints are `POST /v1/chat/completions` and `GET /health`

### Requirement: Spec-driven changes
The bootstrap SHALL produce a reusable OpenSpec workflow for future changes.

#### Scenario: First change is created
- **WHEN** the team starts the first backend implementation change
- **THEN** proposal, design, and task artifacts exist
- **AND** the task list is granular enough to implement in small batches