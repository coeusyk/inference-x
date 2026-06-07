# Platform Specification Delta

## MODIFIED Requirements

### Requirement: Incremental architecture
The platform SHALL support routing as an additive layer above the core engine path.

#### Scenario: Routing is added
- **WHEN** routing is introduced
- **THEN** the existing chat completions contract remains stable
- **AND** model selection is controlled through config or explicit request context
- **AND** route handlers do not become responsible for routing policy logic