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
fixed, OpenAI-compatible order, SHALL begin every stream with one pre-generation
metadata event, and SHALL emit a usage event only when the client requests one.

#### Scenario: Default stream without usage requested
- **WHEN** a client streams a chat completion without requesting usage
- **THEN** it receives exactly one pre-generation metadata event carrying an empty
  choices array, the resolved block and the warnings collection
- **AND** then zero or more content events, each carrying a null finish reason
- **AND** then exactly one terminal event carrying an empty delta and a non-null
  finish reason
- **AND** then `[DONE]` as the final event
- **AND** no usage event is emitted

#### Scenario: Usage requested
- **WHEN** a client streams a chat completion and requests usage
- **THEN** it additionally receives exactly one usage event, after the terminal
  event and before `[DONE]`
- **AND** that event carries an empty choices array and a populated usage object
- **AND** no event is emitted between the usage event and `[DONE]`

#### Scenario: Stream times out
- **WHEN** the per-token stream timeout elapses
- **THEN** the pre-generation metadata event has already been emitted
- **AND** the platform emits its error event and stops generating
- **AND** `[DONE]` is still emitted as the final event
- **AND** no terminal event and no usage event are emitted

#### Scenario: Existing consumers are unaffected
- **WHEN** a client that terminates on `[DONE]` and ignores events carrying no
  delta text reads a stream containing pre-generation, terminal and usage events
- **THEN** it behaves exactly as it did before those events existed

#### Scenario: Time to first token is measured from content
- **WHEN** the platform records time to first token for a streamed request
- **THEN** the measurement is anchored to the first content event
- **AND** the pre-generation metadata event does not enter the measurement

#### Scenario: Change is applied
- **WHEN** this change is archived
- **THEN** the engine streaming contract yields structured chunks
- **AND** no component derives a reported token count from response text
- **AND** the superseding decision record states that previously reported
  streamed token counts and every throughput figure derived from them are not
  comparable with figures produced afterwards
- **AND** a decision record states which metadata may be emitted before generation
  and which may not

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

### Requirement: Substitutions are visible in the response
The platform SHALL report, in the response to a chat completion, every request
parameter it substituted for one the client supplied, and SHALL do so for both the
streamed and the non-streamed path.

#### Scenario: Output tokens are clamped
- **WHEN** admission clamps the requested output-token count to fit the context
  ceiling or the remaining KV budget
- **THEN** the response carries a warning of type `substituted` naming the affected
  field
- **AND** the resolved block reports the output-token count that was actually used

#### Scenario: Nothing was substituted
- **WHEN** a request is admitted without any parameter being changed and every
  admission input is available
- **THEN** the response carries an empty warnings collection
- **AND** the collection is present and empty rather than absent or null

#### Scenario: Client can reconstruct what ran
- **WHEN** a client reads the resolved block of a completed request
- **THEN** it contains every generation parameter the platform used, including the
  effective seed
- **AND** re-submitting those parameters with the original messages describes the
  same execution

### Requirement: The resolved block is the effective request
The platform SHALL derive the resolved block from the request the platform actually
executed, and SHALL NOT populate it from process-level or server-level state.

#### Scenario: Field membership is derivable
- **WHEN** a field is present in the resolved block
- **THEN** that field exists on the chat completion request schema
- **AND** it is not one of the message content or the transport and policy controls

#### Scenario: Server configuration is excluded
- **WHEN** a load-time engine or tier setting influenced admission
- **THEN** the resolved block does not report that setting as a resolved value
- **AND** its influence is reported as the reason of a warning instead

#### Scenario: Seed is echoed without overclaim
- **WHEN** a client supplies a seed
- **THEN** the resolved block reports that seed unchanged
- **AND** when no seed was supplied the resolved block reports no seed rather than
  inventing one

### Requirement: Degradation is typed and observable but never fail-closed
The platform SHALL report every admission gate it skipped, and every admission input
it estimated, as a typed warning and a structured log record, and SHALL continue to
admit the request.

#### Scenario: A gate is skipped because a capability is unavailable
- **WHEN** an admission gate cannot run because the engine reports no value for the
  number it needs
- **THEN** the response carries a warning of type `degraded` identifying the skipped
  gate
- **AND** a structured log record is written for the same condition
- **AND** the request is still admitted

#### Scenario: A prompt token count is estimated
- **WHEN** the engine cannot supply a prompt-token count
- **THEN** the platform uses its character-length heuristic as the gate input
- **AND** the response carries a warning of type `degraded` for the estimate

#### Scenario: Degradation never rejects
- **WHEN** a request would carry only degraded warnings
- **THEN** it is admitted regardless of whether strict mode was requested
- **AND** the admitted output-token count is the same as it would have been before
  degradation was made observable

### Requirement: Strict mode converts substitution into rejection only
The platform SHALL, when a client requests strict mode, reject any request it would
otherwise have executed with a substituted parameter, and SHALL NOT otherwise change
how an admitted request is executed.

#### Scenario: Strict rejects where the default clamps
- **WHEN** a client requests strict mode and admission would clamp a parameter
- **THEN** the request is rejected with a client error
- **AND** no generation is performed

#### Scenario: Strict does not change accepted executions
- **WHEN** a request is accepted under both the default and strict mode with an
  identical seed
- **THEN** the generated content is byte-identical between the two
- **AND** the resolved block is identical between the two

#### Scenario: One predicate, two outcomes
- **WHEN** the set of conditions that reject under strict mode is compared with the
  set that emits a substituted warning by default
- **THEN** the two sets are equal

### Requirement: Pre-generation metadata lifecycle
A streamed response SHALL have a pre-generation metadata phase that ends when the
first token is sampled. The platform SHALL emit an event in that phase if and only if
every field the event carries is fully determined and immutable at the moment the
effective request is finalized, and SHALL NOT emit in that phase any fact that can
change during or after generation.

#### Scenario: Resolution precedes content
- **WHEN** a client streams a chat completion
- **THEN** the platform emits exactly one pre-generation event carrying the resolved
  block and the warnings collection
- **AND** that event precedes every content event
- **AND** it carries an empty choices array

#### Scenario: Pre-generation metadata survives a failed generation
- **WHEN** the per-token stream timeout elapses before generation completes
- **THEN** the pre-generation event has already been emitted
- **AND** the client can still determine what the platform resolved

#### Scenario: A post-generation fact may not be emitted before generation
- **WHEN** a fact is not determined at the moment the effective request is finalized
- **THEN** it is not carried by any pre-generation event

#### Scenario: Cardinality is fixed
- **WHEN** the pre-generation phase is inspected
- **THEN** it contains exactly one event
- **AND** adding a second event is a modification of this requirement

### Requirement: Post-generation metadata lifecycle
Events emitted after the terminal event SHALL carry only facts about what actually
happened during generation, and the usage event SHALL be the last event before the
stream terminator.

#### Scenario: Usage is the final metadata event
- **WHEN** a client streams a chat completion and requests usage
- **THEN** no event is emitted between the usage event and the stream terminator

#### Scenario: No trailing resolution
- **WHEN** a stream completes normally
- **THEN** no resolved block and no warnings collection is emitted after the terminal
  event

### Requirement: Prompt token counting is a declared engine capability
The Engine Boundary SHALL declare prompt-token counting as part of its interface, and
callers SHALL treat an unavailable count as a degraded input rather than an error.

#### Scenario: An engine that can count is asked directly
- **WHEN** admission needs a prompt-token count
- **THEN** it invokes the declared interface method rather than probing the engine
  for an undeclared attribute

#### Scenario: An engine that cannot count degrades
- **WHEN** an engine does not supply a prompt-token count
- **THEN** admission falls back to its character-length heuristic
- **AND** the degradation is reported per the degradation requirement

#### Scenario: The provisional capability stays undeclared
- **WHEN** the Engine Boundary is inspected after this change
- **THEN** KV capacity is not declared on it
- **AND** it is still discovered as an optional attribute by its existing callers

### Requirement: Engine driver rejects requests immediately after failure
The system SHALL guarantee that once an `EngineDriver`'s underlying `step()` call
has failed and the driver has been marked dead, no request submitted afterward is
silently enqueued and left unserved — it SHALL be rejected immediately with the
driver's failure, not left to time out.

#### Scenario: Submission during the failure transition is rejected, not orphaned
- WHEN a request is submitted concurrently with a driver's `step()` call failing
- AND the driver's dead-flag transition and the request's enqueue decision race
- THEN the request either lands in the pre-failure queue and is included in the
  broadcast of the failure to all pending requests, or is rejected immediately by
  the submission call — never silently enqueued with no thread left to serve it

#### Scenario: Submission after death raises immediately
- WHEN a request is submitted to a driver that is already marked dead
- THEN the submission call raises immediately, carrying the original failure
- AND the caller does not wait for any completion or streaming timeout to learn
  the driver has failed

### Requirement: Default model resolves family names via variant selection
The system SHALL resolve `INFERENCE_X_DEFAULT_MODEL` through
`variant_selector.select_variant()` when its value matches a registered model
family, so the highest-precision variant that fits the current VRAM tier is
selected at startup. A value that is already a concrete registered model name
SHALL continue to pass through unchanged.

#### Scenario: Family name resolves to the best-fit variant at startup
- WHEN `INFERENCE_X_DEFAULT_MODEL` matches a `ModelEntry.family` in the loaded
  registry and a VRAM tier is resolved
- THEN the value is resolved via `select_variant()` to the highest-precision
  variant that fits the current tier's available VRAM
- AND that resolved variant name is used to construct the default-model policy

#### Scenario: Concrete model names pass through unchanged
- WHEN `INFERENCE_X_DEFAULT_MODEL` is already an exact registered model name
- THEN the value is used unchanged
- AND no call to `select_variant()` is made

### Requirement: Default model resolution surfaces its outcome, not silence
The system SHALL make the outcome of default-model resolution observable and
SHALL NOT silently fall back to an arbitrary variant when a recognized family
has no fitting variant for the current VRAM tier.

#### Scenario: Resolution is logged at startup
- WHEN a family name is resolved to a concrete variant
- THEN the system logs, at INFO level, the family name, the resolved variant
  name, and the current tier name

#### Scenario: Missing-family-fit is a hard startup error
- WHEN `INFERENCE_X_DEFAULT_MODEL` matches a registered family but no variant
  in that family fits the current VRAM tier's available budget
- THEN startup raises an error naming the family, the available variants, and
  the current tier's VRAM budget
- AND the system does not silently start with a different, unrequested variant

### Requirement: Changes are verified automatically before merge
The project SHALL verify every proposed change against the unit suite, the
configured lint rules, and the configured type baseline automatically, without
relying on a contributor or reviewer to run those checks by hand.

#### Scenario: A proposed change breaks an existing test
- WHEN a change is proposed that causes any test in the unit suite to fail
- THEN the automated verification reports failure
- AND the change cannot be merged into a long-lived branch until it passes

#### Scenario: A proposed change violates lint or type rules
- WHEN a change is proposed that violates a configured lint rule, or introduces a
  type error in a module outside the recorded type baseline
- THEN the automated verification reports failure
- AND the change cannot be merged into a long-lived branch until it passes

#### Scenario: Verification requires no accelerator
- WHEN automated verification runs
- THEN it completes on a standard hosted runner with no GPU present
- AND no test is skipped solely because verification ran without a GPU

#### Scenario: The type baseline is explicit and bounded
- WHEN a module is exempted from type checking
- THEN that module is named individually in the recorded baseline
- AND no repository-wide or wildcard exemption is configured, so that a module
  added later is checked by default rather than silently exempted

#### Scenario: Change is applied
- WHEN this change is archived
- THEN automated verification runs on pull requests and on pushes to long-lived
  branches
- AND the required checks are enforced at merge time, not merely reported
- AND the type baseline and the policy governing it are recorded in
  `docs/DECISIONS.md`

### Requirement: Reproducible benchmark results
The benchmark runner SHALL use a fixed, versioned prompt suite so results are
comparable across runs only when they share verified suite identity. The suite's
`suite_version` SHALL be the pinned content hash of the parsed prompt collection
(DEC-054), verified at load, and used as a necessary selection key before
latest-per-model consumption (DEC-055).

#### Scenario: Benchmark is run twice on the same hardware
- **WHEN** the same model is benchmarked twice with the standard prompt suite
- **THEN** results are stored separately with timestamps
- **AND** the advisor uses the most recent result per model among results that match
  the expected `suite_version`

#### Scenario: Unverified suite cannot produce results
- **WHEN** the prompt suite fails identity verification at load
- **THEN** the benchmark run does not proceed
- **AND** no new result file is written from that failed load

### Requirement: Hardware-aware model recommendation
The system SHALL measure model performance on the operator's hardware and produce a
ranked recommendation with plain-language reasoning. Ranking SHALL use only measured
score components under the exact frozen weights, and SHALL expose viability solely
through `viable`. score is a within-report ordinal used only to rank viable models
produced from the same benchmark suite.

#### Scenario: Advisor is run after benchmarking
- **WHEN** `make advise` is run after at least one benchmark result exists for the
  current suite
- **THEN** the advisor produces a ranked list of models
- **AND** each entry includes: throughput (tok/s), TTFT (ms), device VRAM occupancy
  (GiB), a score that is a within-report ordinal used only to rank viable models
  produced from the same benchmark suite, and a one-line recommendation string
- **AND** models that exceed available VRAM are flagged as not viable via `viable`
  rather than by score threshold alone

