## ADDED Requirements

### Requirement: Non-streaming chat completion responses carry the run manifest inline

The platform SHALL include the full run manifest as an additive `manifest` field on
`ChatCompletionResponse` for non-streaming `POST /v1/chat/completions` responses, resolving the
manifest-delivery question `add-run-manifest`'s design left open. The `manifest` field SHALL be
`null` whenever `run_id` is `null`, and SHALL equal the same manifest object whose `run_id` is
carried on `run_id` and `X-Run-Id`. The platform SHALL NOT be required to include `manifest` on
streaming responses in this capability.

#### Scenario: Non-streaming response body carries the manifest

- **WHEN** a non-streaming chat completion is served
- **THEN** the JSON response body includes a `manifest` object
- **AND** `manifest.run_id` equals both the top-level `run_id` field and the `X-Run-Id` header

#### Scenario: The inline manifest is independently verifiable

- **WHEN** a client recomputes `run_id` from the returned manifest's own `engine`, `model`,
  `runtime`, `sampling`, `request`, and `warnings` fields (each warning projected to its `type`,
  `code`, and `field` only) using the same content-addressing algorithm the platform used
- **THEN** the recomputed value equals the manifest's `run_id`

#### Scenario: Streaming responses carry no inline manifest field in this capability

- **WHEN** a streaming chat completion is served
- **THEN** no `manifest` body field is required of this capability
