# Platform Specification

## Purpose

InferenceX is a self-hosted inference platform that exposes a stable API for LLM serving and grows incrementally from a single vLLM-backed engine into a more complete system with routing, observability, and evaluation support.
## Requirements
### Requirement: Stable API-first platform
The system SHALL expose stable HTTP interfaces before layering on optional
capabilities, and non-streaming chat completions SHALL support the same
continuous-batching concurrency as streaming completions without dropping or
corrupting any in-flight request's output.

#### Scenario: Concurrent non-streaming requests
- WHEN 2 or more non-streaming chat completion requests are in flight against the
  same loaded model at the same time
- THEN every request completes with its own, complete, non-truncated output
- AND no request's terminal output is discarded or delivered to a different
  request

#### Scenario: Engine driver failure
- WHEN the shared per-engine driver thread's call into the underlying inference
  engine raises an exception
- THEN every request currently in flight against that engine receives that
  exception instead of hanging indefinitely
- AND the engine is reported unhealthy so new requests are not accepted by a dead
  driver

#### Scenario: Change is applied
- WHEN this change is applied
- THEN `POST /v1/chat/completions` request/response contracts are unchanged for
  both streaming and non-streaming callers
- AND all existing unit tests continue to pass unchanged

### Requirement: Incremental architecture
The system SHALL support additive growth without requiring large structural
rewrites, and VRAM/concurrency tuning knobs declared in tier or model config SHALL
actually be applied at engine construction time rather than left inert.

#### Scenario: Tier and model knobs are resolved together
- WHEN a model is loaded under a resolved VRAM tier
- THEN the effective `max_num_seqs` and `max_num_batched_tokens` used to construct
  the engine are the tighter of the tier's ceiling and any per-model override
- AND `block_size`, `kv_cache_dtype`, and `enable_prefix_caching` from the resolved
  tier are passed to the underlying inference engine

#### Scenario: Sequence-concurrency ceiling is enforced pre-dispatch
- WHEN the number of in-flight requests against a model is at or above its
  resolved `max_num_seqs`
- THEN a new request for that model is rejected with HTTP 429 and a
  `Retry-After` header, regardless of request priority
- AND the engine itself is never invoked for the rejected request

#### Scenario: Knob resolution is unavailable
- WHEN no VRAM tier can be resolved for the current hardware
- THEN engine construction and admission control continue using each model's own
  configured values (or built-in defaults), consistent with this codebase's
  existing fail-open posture for advisory signals

#### Scenario: Change is applied
- WHEN this change is applied
- THEN existing `vram_tiers.yaml` and `models.yaml` files without the new fields
  continue to work unchanged
- AND all existing unit tests continue to pass unchanged

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

### Requirement: Model variant selection at load time
The system SHALL support grouping model configuration entries into named families
and, when an operator requests a family rather than a concrete model, SHALL select
the highest-precision variant that fits the available VRAM budget at load time.

#### Scenario: Multiple variants, one fits
- WHEN a model family has bf16, int8, and 4-bit quantized variants registered
- AND only the 4-bit variant's estimated weight size fits the available VRAM
  budget
- THEN the 4-bit variant is loaded
- AND the higher-precision variants are not loaded

#### Scenario: No variant fits
- WHEN no variant in a requested family fits the available VRAM budget
- THEN startup fails with an error naming the family and each variant's
  estimated size versus the available budget
- AND the process does not begin accepting requests

#### Scenario: Concrete model name bypasses selection
- WHEN `INFERENCE_X_LOADED_MODELS` names a concrete, registered model entry
  directly (not a family)
- THEN that exact entry is loaded, unaffected by variant selection

#### Scenario: Ungrouped model is unaffected
- WHEN a model entry has no `family` set
- THEN it loads exactly as it did before this change

#### Scenario: Change is applied
- WHEN this change is applied
- THEN `TaskRouter` and `AdmissionController` behavior is unchanged — both
  continue to operate only on the concrete, already-resolved model name
- AND all existing unit tests continue to pass unchanged

### Requirement: Streamed token counts originate from the engine
The platform SHALL derive every completion-token figure it reports from engine
accounting. It MUST NOT approximate token counts from response text, and it MUST
report no figure rather than an estimated one.

#### Scenario: Streamed and non-streamed counts agree
- **WHEN** the same prompt and sampling parameters are sent once as a streaming
  request with usage requested, and once as a non-streaming request
- **THEN** both report the same `completion_tokens`
- **AND** both values originate from the engine's own accounting

#### Scenario: Usage is unavailable
- **WHEN** a streamed request completes without usage being requested
- **THEN** the platform records no completion-token figure for that request
- **AND** it records neither zero nor a value derived from counting words in the
  response text

#### Scenario: Text is never counted as tokens
- **WHEN** any component reports a completion-token count
- **THEN** that count came from the engine
- **AND** no code path derives a reported token count from response text

### Requirement: Engine streaming contract carries terminal metadata
The engine streaming interface SHALL yield structured chunks capable of carrying
delta text, a finish reason, and usage, so that a backend can report what it
actually did. The chunk type MUST live in the existing wire-schema package; no
backend-neutral execution package may be introduced (DEC-047, DEC-049).

#### Scenario: Content chunk
- **WHEN** an engine emits generated text mid-stream
- **THEN** the chunk carries the delta text
- **AND** its finish reason and usage are both absent

#### Scenario: Terminal chunk
- **WHEN** an engine finishes generating
- **THEN** it emits a final chunk carrying the finish reason
- **AND** that chunk carries usage when the backend can account it

#### Scenario: Boundary is not widened further
- **WHEN** the streaming contract changes
- **THEN** the chunk type is defined in the wire-schema package
- **AND** no `inference_x/execution/` package, dual type system, or
  wire-to-execution translation layer is introduced

### Requirement: OpenAI-compatible streaming event order
The platform SHALL emit server-sent events for a streamed chat completion in a
fixed, OpenAI-compatible order, and SHALL emit a usage event only when the client
requests one.

#### Scenario: Default stream without usage requested
- **WHEN** a client streams a chat completion without requesting usage
- **THEN** it receives zero or more content events, each carrying a null finish
  reason
- **AND** then exactly one terminal event carrying an empty delta and a non-null
  finish reason
- **AND** then `[DONE]` as the final event
- **AND** no usage event is emitted

#### Scenario: Usage requested
- **WHEN** a client streams a chat completion and requests usage
- **THEN** it additionally receives exactly one usage event, after the terminal
  event and before `[DONE]`
- **AND** that event carries an empty choices array and a populated usage object

#### Scenario: Stream times out
- **WHEN** the per-token stream timeout elapses
- **THEN** the platform emits its error event and stops generating
- **AND** `[DONE]` is still emitted as the final event
- **AND** no terminal event and no usage event are emitted

#### Scenario: Existing consumers are unaffected
- **WHEN** a client that terminates on `[DONE]` and ignores events carrying no
  delta text reads a stream containing terminal and usage events
- **THEN** it behaves exactly as it did before those events existed

#### Scenario: Change is applied
- **WHEN** this change is archived
- **THEN** the engine streaming contract yields structured chunks
- **AND** no component derives a reported token count from response text
- **AND** the superseding decision record states that previously reported
  streamed token counts and every throughput figure derived from them are not
  comparable with figures produced afterwards

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

