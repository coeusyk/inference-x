## MODIFIED Requirements

### Requirement: Stable API-first platform
The system SHALL expose stable HTTP interfaces before layering on optional capabilities.

#### Scenario: Initial stable API
- **WHEN** the first backend milestone is implemented
- **THEN** the system provides a health endpoint and a chat completions endpoint
- **AND** request and response contracts are documented and validated

#### Scenario: Core engine slice is implemented
- **WHEN** the add-core-vllm-engine change is complete
- **THEN** `POST /v1/chat/completions` is available
- **AND** `GET /health` is available
- **AND** both endpoints use typed schemas
- **AND** the response format is stable enough for later extension without rewriting the route layer

#### Scenario: Chat completion streaming is requested
- **WHEN** a client sends `POST /v1/chat/completions` with `stream=true`
- **THEN** the system responds with `text/event-stream`
- **AND** each generated text chunk is emitted as an OpenAI-compatible `data: {...}` chat completion chunk event
- **AND** the stream ends with `data: [DONE]`

#### Scenario: Non-streaming chat completion remains stable
- **WHEN** a client sends `POST /v1/chat/completions` with `stream=false` or omits `stream`
- **THEN** the system returns the existing JSON chat completion response
- **AND** the non-streaming response schema is unchanged

## ADDED Requirements

### Requirement: Interactive terminal playground
The system SHALL provide a Textual terminal playground that consumes the public InferenceX API without adding backend-only UI endpoints.

#### Scenario: Playground starts against a server
- **WHEN** the user runs `uv run python playground/app.py`
- **THEN** the playground checks `GET /health`
- **AND** displays the server URL and connection status in the header

#### Scenario: Playground streams a prompt
- **WHEN** the user submits a prompt
- **THEN** the playground sends `stream=true` to `POST /v1/chat/completions`
- **AND** appends received SSE token chunks to the response panel as they arrive
- **AND** shows completion status when `data: [DONE]` is received

#### Scenario: Playground compare mode
- **WHEN** the user runs `uv run python playground/app.py --compare MODEL_A MODEL_B`
- **THEN** the playground displays two response panels
- **AND** streams responses for both selected models using the same public chat completions API
