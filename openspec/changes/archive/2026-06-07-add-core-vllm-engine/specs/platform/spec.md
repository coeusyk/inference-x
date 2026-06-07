# Platform Specification Delta

## MODIFIED Requirements

### Requirement: Stable API-first platform
The first backend change SHALL expose a single stable chat-completions path and a health path.

#### Scenario: Core engine slice is implemented
- **WHEN** the change is complete
- **THEN** `POST /v1/chat/completions` is available
- **AND** `GET /health` is available
- **AND** both endpoints use typed schemas
- **AND** the response format is stable enough for later extension without rewriting the route layer

### Requirement: Incremental architecture
The core backend SHALL be structured so later routing and observability layers can be added without breaking the first engine path.

#### Scenario: Later features are planned
- **WHEN** routing or observability work begins
- **THEN** the existing engine interface remains the default abstraction boundary
- **AND** route handlers remain thin