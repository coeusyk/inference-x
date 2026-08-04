# Platform Specification Delta — add-deterministic-generation

## ADDED Requirements

### Requirement: Optional request seed reaches the sampler
The platform SHALL accept an optional integer `seed` on
`POST /v1/chat/completions` as a first-class field of `ChatCompletionRequest`.
When `seed` is set and the live vLLM engine builds sampling parameters, the
runtime SHALL pass that integer into the backend sampling API exactly as
received, without rewriting, clamping, or normalizing backend sentinel
semantics. When `seed` is absent or null, the runtime SHALL omit `seed` from
sampling construction so pre-OS-3 sampling behaviour is preserved for all other
parameters. Streaming and non-streaming generation SHALL use the same sampling
parameter builder. The platform MUST NOT echo the effective seed on the response
in this change, and documentation MUST NOT claim end-to-end determinism or that
runs are reproducible.

#### Scenario: Seed is accepted on the wire
- **WHEN** a client sends `seed` as an integer on a chat completion request
- **THEN** the request validates successfully
- **AND** the field is not silently dropped by schema validation

#### Scenario: Seed reaches live SamplingParams unchanged
- **WHEN** a live vLLM engine builds sampling parameters for a request whose
  `seed` is set (including the value `-1`)
- **THEN** the backend sampling API receives that exact integer as `seed`
- **AND** the runtime has not mapped, rejected, or warned about the value

#### Scenario: Absent seed preserves prior sampling construction
- **WHEN** a request omits `seed` or sets it to null
- **THEN** sampling construction does not include a `seed` key
- **AND** temperature, max_tokens, top_p, and repetition_penalty rules match
  pre-OS-3 behaviour for the same other fields

#### Scenario: Stream and non-stream share the sampling builder
- **WHEN** the same request parameters including `seed` are used for streaming
  and non-streaming generation
- **THEN** both paths obtain sampling parameters from the same builder
- **AND** seed honour does not diverge by transport

#### Scenario: No response echo and no overclaim
- **WHEN** this change is applied
- **THEN** `ChatCompletionResponse` gains no seed, resolved, warnings, or strict
  fields
- **AND** documentation states that seed is honoured (reaches the sampler)
- **AND** documentation does not claim the server is deterministic or that runs
  are reproducible end-to-end

#### Scenario: Degraded engine path does not invent seed errors
- **WHEN** the vLLM engine is unavailable and streaming returns the degraded
  content-only path
- **THEN** no new seed-related warning or HTTP error is introduced
- **AND** the degraded stream remains a single content message without a
  terminal `finish_reason` of `error`

#### Scenario: Default benchmarks remain unseeded
- **WHEN** the first-party benchmark runner issues its default chat requests
- **THEN** those requests remain without a pinned `seed`
- **AND** the runner stays throughput-oriented (deterministic benchmarking is
  deferred)
