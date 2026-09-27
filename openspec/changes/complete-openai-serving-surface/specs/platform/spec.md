## ADDED Requirements

### Requirement: Unsupported request fields are rejected, never silently dropped
The platform SHALL reject a `POST /v1/chat/completions` request that carries a field not
declared on the chat completion request schema, or a message that carries a field not
declared on the message schema, with HTTP 400 naming the offending field. The platform
SHALL NOT accept and silently ignore such a field.

#### Scenario: Unknown top-level field
- **WHEN** a client sends a chat completion request containing `"n": 2`
- **THEN** the response is HTTP 400 with an OpenAI error envelope whose `param` is `n`

#### Scenario: Unknown message field
- **WHEN** a message carries a `name` field
- **THEN** the response is HTTP 400 and the field is not silently discarded

### Requirement: Request validation errors use the OpenAI error envelope
The platform SHALL report request validation failures on the OpenAI-compatible endpoints
as HTTP 400 with body `{"error": {"message", "type": "invalid_request_error", "param",
"code"}}`, where `param` names the first invalid field when one can be identified.

#### Scenario: Out-of-range parameter
- **WHEN** a client sends `temperature: 5`
- **THEN** the response is HTTP 400, `error.type` is `invalid_request_error`, and
  `error.param` is `temperature`

### Requirement: Stop sequences reach the sampler
The platform SHALL accept `stop` as a string or a list of at most 4 non-empty strings,
SHALL forward it to the backend sampler, SHALL report the generation as
`finish_reason: "stop"` when a stop sequence ended it, SHALL echo it in the `resolved`
block, and SHALL include it in the run manifest's `sampling` block. The stop sequence
itself SHALL NOT appear in the returned content.

#### Scenario: Stop sequence ends generation
- **WHEN** a request sets `stop: ["\n\n"]` and the model emits a blank line
- **THEN** the returned content ends before that sequence and `finish_reason` is `stop`

#### Scenario: Too many stop sequences
- **WHEN** a request sets `stop` to a list of 5 strings
- **THEN** the response is HTTP 400 with `param` `stop`

### Requirement: Omitted max_tokens resolves to the remaining context window
The platform SHALL, when a request sets neither `max_tokens` nor `max_output_tokens`,
resolve the output budget to the model's effective context ceiling minus the prompt's
token count (further limited by the model's `max_completion_tokens` when configured), SHALL NOT emit a substitution warning for that resolution, and SHALL report
the resolved value in `resolved.max_tokens`. The KV-pressure gate SHALL still apply to the
resolved value, with its existing warning and strict-mode behavior. The platform SHALL NOT
impose a fixed upper bound on `max_tokens` or `max_output_tokens` other than the context
ceiling.

#### Scenario: Aider-style request without max_tokens
- **WHEN** a request omits `max_tokens` and its prompt is 5000 tokens against an
  8192-token context ceiling
- **THEN** the request is admitted with an output budget of 3192 tokens
- **AND** no `max_tokens_clamped_to_context` warning is emitted

#### Scenario: Large explicit max_tokens
- **WHEN** a request sets `max_tokens: 6000` against an 8192-token ceiling with a
  1000-token prompt
- **THEN** the request is admitted without a schema error

### Requirement: Request size is bounded by tokens, not by fixed character caps
The platform SHALL enforce prompt size through the context gate, measured in tokens. Message count and
per-message character limits SHALL exist only as request-body sanity guards, set at no
fewer than 2048 messages and 1,000,000 characters per message.

#### Scenario: Long coding session
- **WHEN** a request carries 120 messages whose total tokens fit the context ceiling
- **THEN** the request is admitted

### Requirement: Context overflow is reported in a client-actionable form
The platform SHALL, when a prompt or a prompt plus an explicitly requested output cannot
fit the context ceiling, return HTTP 400 with `error.code` `context_length_exceeded`
and a message beginning `This model's maximum context length is <N> tokens`. It SHALL NOT
return a generic message. A strict-mode rejection SHALL return HTTP 400 with its real
message and `error.code` `strict_violation`.

#### Scenario: Prompt exceeds the context
- **WHEN** a prompt of 9000 tokens is sent to a model whose context ceiling is 8192
- **THEN** the response is HTTP 400, `error.code` is `context_length_exceeded`, and the
  message states the 8192-token limit

### Requirement: The full run manifest is opt-in
The platform SHALL include the full run manifest in a non-streaming chat completion
response body, as a `manifest` field, only when the request sets `include_manifest: true`.
The default response body SHALL NOT contain a `manifest` key. The `run_id` body field and
the `X-Run-Id` header SHALL be emitted exactly as without this capability, regardless of
`include_manifest`. The `timing`, `resolved`, and `warnings` fields SHALL remain in the
default response. A request that sets both `include_manifest: true` and `stream: true`
SHALL be rejected with HTTP 400, `param` `include_manifest`. The returned manifest's
`run_id` SHALL be recomputable from its own `engine`, `model`, `runtime`, `sampling`,
`request`, and warning `type`/`code`/`field` fields.

#### Scenario: Default response is clean
- **WHEN** a non-streaming request omits `include_manifest`
- **THEN** the body has no `manifest` key, still carries `run_id`, and the response
  carries `X-Run-Id`

#### Scenario: Opt-in manifest
- **WHEN** a non-streaming request sets `include_manifest: true`
- **THEN** the body carries `manifest`, and `manifest.run_id` equals the body's `run_id`
  and the `X-Run-Id` header

### Requirement: VRAM tier context cap composes with concurrency as a KV-budget envelope
The platform SHALL treat a VRAM tier's `max_model_len_cap × max_num_seqs` as a policy
envelope for a model's context and sequence concurrency, rather than treating
`max_model_len_cap` as an unconditional per-model ceiling. A model's effective context
ceiling SHALL be its configured `max_model_len` when that value does not exceed the tier
cap. It SHALL also be its configured `max_model_len` when that value exceeds the cap but
`max_model_len × effective max_num_seqs ≤ max_model_len_cap × tier max_num_seqs`.
Otherwise the effective context ceiling SHALL be the tier cap. The effective
`max_num_seqs` SHALL remain the minimum of the model's and the tier's values. The platform
SHALL report each model's effective context ceiling, effective `max_num_seqs`, and whether
the tier cap limited the context in `GET /v1/models` and `GET /v1/plan`. The envelope
SHALL NOT be presented as a measured VRAM figure. Per-request KV safety SHALL continue to
come from the engine-reported KV capacity.

#### Scenario: Concurrency traded for context
- **WHEN** the tier is `6gb` (cap 2048, 4 sequences) and a model entry sets
  `max_model_len: 8192, max_num_seqs: 1`
- **THEN** that model's effective context ceiling is 8192 with 1 sequence, and
  `/v1/plan` reports it as a composed envelope

#### Scenario: Trade not made
- **WHEN** the tier is `6gb` and a model entry sets `max_model_len: 8192` without lowering
  `max_num_seqs`
- **THEN** that model's effective context ceiling is 2048, and `/v1/models` and
  `/v1/plan` report that the tier cap limited it

### Requirement: A model's completion cap is resolved before dispatch, never inside the engine
The platform SHALL apply a model's configured `max_completion_tokens` during admission, before `resolved` and the manifest are built, and the engine SHALL generate exactly the admitted output budget with no further model-level adjustment. When an explicit `max_tokens` or `max_output_tokens` exceeds the cap, the platform SHALL clamp it to the cap and emit a `substituted` warning with code `max_tokens_clamped_to_model_cap`, or, when the request sets `strict`, SHALL reject it with `strict_violation`. When the request omits the output budget, the platform SHALL resolve it to the smaller of the remaining context and the cap without a warning.

#### Scenario: Request above the model cap
- **WHEN** a model is configured with `max_completion_tokens: 4` and a request sets `max_tokens: 64`
- **THEN** the engine is asked for exactly 4 tokens
- **AND** `resolved.max_tokens` and `manifest.sampling.resolved_max_tokens` are 4
- **AND** a `max_tokens_clamped_to_model_cap` warning is emitted

#### Scenario: Request below the model cap
- **WHEN** a model is configured with `max_completion_tokens: 256` and a request sets `max_tokens: 8`
- **THEN** the engine is asked for exactly 8 tokens and no substitution warning is emitted

#### Scenario: Strict request above the model cap
- **WHEN** the same over-cap request sets `strict: true`
- **THEN** the response is 400 with code `strict_violation` and nothing is generated
