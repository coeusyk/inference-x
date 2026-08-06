## ADDED Requirements

### Requirement: Per-request engine timing is exposed as an additive, optional response field

The system SHALL expose queue, prefill, decode, and total engine execution
time for each chat completion request, derived from the engine's own
per-request timing state, as an optional `timing` field on
`ChatCompletionResponse` and on the terminal `ChatStreamChunk` event.

#### Scenario: Timing is present in a non-streaming response

- **WHEN** a non-streaming chat completion is served by an engine that
  supplies per-request timing state
- **THEN** the response's `timing` field is populated with
  `queue_time_ms`, `prefill_time_ms`, `decode_time_ms`, and
  `inference_time_ms`

#### Scenario: Timing is present in the streaming usage event

- **WHEN** a streaming chat completion reaches its terminal event and the
  client requested `stream_options.include_usage`
- **THEN** the usage event carries `timing` alongside `usage`, populated
  identically to what a non-streaming request for the same prompt would
  report, and every prior content event's `timing` field is `None`

#### Scenario: Timing follows usage's existing opt-in, not a new flag

- **WHEN** a streaming client does not set `stream_options.include_usage`
- **THEN** no `timing` event is emitted — the same opt-in that already
  gates `usage` also gates `timing`, so this change adds no new request
  field or stream-protocol flag

#### Scenario: Timing is absent when the engine does not supply it

- **WHEN** the engine's finished output carries no per-request timing
  state, or any required timing field cannot be read
- **THEN** `timing` is `None` on the response — never a partial
  `EngineTiming`, never an estimated or zero-filled value

#### Scenario: /v1/metrics and /metrics are unaffected by this change

- **WHEN** a request's `timing` field is populated or absent
- **THEN** `GET /v1/metrics`'s aggregates and `GET /metrics`'s Prometheus
  series are computed exactly as before — this field adds no new
  computation to either surface and reads no data from either

### Requirement: Engine timing intervals are derived from a single clock domain

The system SHALL compute every `EngineTiming` field exclusively from
timestamps captured within the engine's own execution process, and SHALL
NOT combine them with any wall-clock or HTTP-boundary timestamp.

#### Scenario: Intervals are computed only from engine-core timestamps

- **WHEN** `EngineTiming` is derived for a request
- **THEN** every field is a difference between two timestamps captured in
  the engine's own process clock, and no field is derived from a
  frontend-process wall-clock timestamp or an HTTP-boundary timestamp

#### Scenario: A cross-domain metric is not exposed

- **WHEN** a candidate timing value would require comparing a
  frontend-process wall-clock timestamp against an engine-process clock
  timestamp
- **THEN** that value is not exposed by this requirement — engine timing
  and HTTP-boundary timing remain reportable only through their own
  respective, already-existing surfaces

### Requirement: Engine timing derivation shares the single terminal-metadata source

The system SHALL derive `EngineTiming`, `finish_reason`, and `usage` from
the same finished engine output, at the same call site, so that no two
response paths can report different timing for the same request.

#### Scenario: Timing and usage are derived from the same finished output

- **WHEN** a request completes
- **THEN** `EngineTiming`, `finish_reason`, and `usage` are all derived
  from a single read of the engine's finished output for that request

#### Scenario: Non-streaming and streaming paths report identical timing

- **WHEN** the same prompt is served once as a non-streaming request and
  once as a streaming request
- **THEN** both report the same `EngineTiming` values, because both derive
  from the same underlying function
