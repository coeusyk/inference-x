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

#### Scenario: Sequence-concurrency ceiling is enforced with a priority-differentiated bounded wait
- WHEN the number of in-flight requests against a model is at or above its
  resolved `max_num_seqs`
- THEN a new `interactive`-priority request waits, up to a short, bounded,
  configurable deadline, for an in-flight request to complete and free a slot
- AND a new `batch`-priority request waits, up to a separate and
  substantially longer bounded, configurable deadline, for a slot to free
- AND if a slot frees within the request's applicable deadline, the request
  is admitted and proceeds exactly as it would have if the ceiling had never
  been hit
- AND if no slot frees before the applicable deadline elapses, the request
  is rejected with HTTP 429 and a `Retry-After` header
- AND the engine itself is never invoked for a request that is ultimately
  rejected after its wait elapses

#### Scenario: Batch-priority queue depth is bounded
- WHEN a `batch`-priority request arrives for a model whose count of
  currently-waiting `batch`-priority requests is already at that model's
  configured queue-depth capacity
- THEN the new request is rejected immediately with HTTP 429 and a
  `Retry-After` header, without waiting and without ever counting toward or
  affecting the sequence-concurrency wait
- AND `interactive`-priority requests are never subject to this capacity
  check

#### Scenario: Serving an additional model does not degrade any already-loaded model
- WHEN the system is already serving one model and is asked to also serve
  one or more additional models
- THEN each model is served by its own independent server process — a
  server process is never configured to serve more than one model
- AND no already-loaded model's context-length ceiling, decode
  performance, or GPU-memory budget is reduced below what that same model
  would get if it were the only model being served
- AND no model's admission-control behavior (B4's bounded wait, B5's
  priority-differentiated wait and batch-waiter cap) changes because
  another model is also being served
- AND observability signals (health, native metrics) for one model remain
  attributable to that model alone, without collision against another
  model's signals

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
parameter builder. The platform MAY echo the **requested** seed on the response
as provenance — in the resolved block and in the run manifest — but MUST NOT echo
or fabricate a **backend-derived effective** seed the client did not supply, and
documentation MUST NOT claim end-to-end determinism or that runs are reproducible
from the seed alone.

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
- **WHEN** a client supplies a seed
- **THEN** the platform MAY echo that requested seed in the resolved block and in
  the run manifest as provenance
- **AND** it MUST NOT echo or fabricate a backend-derived effective seed the client
  did not supply, and no other field gains an unreviewed echo as a side effect of
  this change
- **AND** documentation does not claim the server is deterministic or that runs are
  reproducible from the seed alone

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

The system SHALL guarantee that once the engine backend has entered a terminal
failure state, no request submitted afterward is silently enqueued and left
unserved — it SHALL be rejected immediately with that failure, not left to
time out. This requirement is implementation-neutral: it describes observable
caller-facing behavior after terminal failure, not a particular internal class,
step loop, or dead-flag mechanism.

This observable property has been verified against the post-migration engine
backend (Task 1.4, `migrate-async-llm-engine`): a request submitted after the
backend has entered a terminal failure state is rejected immediately and
synchronously — see design.md for the verified evidence chain. Verifying that
one backend satisfies this property does not redefine the requirement in
terms of that backend; the requirement remains implementation-neutral, and a
future backend is bound by the same observable contract, not by the specific
mechanism that happens to satisfy it today.

Every other existing requirement (streaming event order, DEC-050 usage-source,
admission reservation lifecycle, the generate_stream declaration fix) is
preserved unchanged by this migration per Compatibility Invariants 1–3 in
design.md.

#### Scenario: Submission after terminal failure is rejected, not orphaned
- **WHEN** a request is submitted after the engine backend has already entered
  a terminal failure state
- **THEN** the submission fails observably with that failure
- **AND** the caller does not wait for a completion or streaming timeout to
  learn the engine has failed

#### Scenario: Submission during the failure transition is not silently orphaned
- **WHEN** a request is submitted concurrently with the engine backend entering
  a terminal failure state
- **THEN** the request is either included in in-flight failure propagation or
  rejected immediately by the submission path — never silently enqueued with
  no engine path left to serve it

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

### Requirement: Admission reservation lifetime invariant
The system SHALL uphold the following reservation lifetime invariant: from the
instant `admit()` successfully returns until the stream generator terminates for
any reason (normal completion, timeout, engine failure, cancellation,
`GeneratorExit`, or client disconnect), exactly one matching `release()` MUST
occur. The concrete mechanism that upholds the invariant MAY change across
phases; the invariant MUST survive those refactors.

#### Scenario: Full stream consumption releases the reservation exactly once
- WHEN a streamed chat completion runs to normal completion
- THEN the admission reservation is released exactly once
- AND the released token count equals the token count reserved at admission

#### Scenario: Timeout releases the reservation exactly once
- WHEN a streamed chat completion ends by timeout
- THEN the admission reservation is released exactly once

#### Scenario: Engine exception releases the reservation exactly once
- WHEN the engine raises during streamed generation
- THEN the admission reservation is released exactly once

#### Scenario: Cancellation releases the reservation exactly once
- WHEN the stream is cancelled via `CancelledError`
- THEN the admission reservation is released exactly once

#### Scenario: GeneratorExit before first token releases the reservation exactly once
- WHEN the stream consumer closes the generator (`GeneratorExit`) before any content
  token is produced — including during the pre-generation prologue
- THEN the admission reservation is released exactly once
- AND the per-model sequence-concurrency slot held for that request is released

#### Scenario: GeneratorExit after first token releases the reservation exactly once
- WHEN the stream consumer closes the generator (`GeneratorExit`) after one or more
  content events have been produced but before normal completion
- THEN the admission reservation is released exactly once

#### Scenario: Client disconnect releases the reservation exactly once
- WHEN the stream consumer disconnects at any suspension point of the stream generator
- THEN the admission reservation is released exactly once

### Requirement: Admission reservation MUST NEVER be released more than once
A reservation MUST NEVER be released more than once. The system SHALL treat
double-release as a correctness failure independent of leak prevention.

#### Scenario: Single reservation yields a single release
- WHEN an admission reservation is successfully established for a stream
- THEN `release()` for that reservation executes exactly once across all termination
  paths
- AND validation MUST fail if `release()` executes twice for that reservation

### Requirement: Reservation lifetime ownership is ChatService-exclusive
The system SHALL keep reservation lifetime ownership exclusively in ChatService.
Admission owns reservation accounting. ChatService owns reservation lifetime.
Middleware MUST NOT compensate for ChatService lifetime gaps. Engine code MUST NOT
compensate for ChatService lifetime gaps. Admission MUST NOT compensate for caller
failures. The fix for this requirement SHALL reside exclusively in ChatService.

#### Scenario: Fix is confined to ChatService
- WHEN this change is implemented
- THEN reservation lifetime enforcement is present in ChatService
- AND Admission, middleware, and engine code do not gain compensating release logic

### Requirement: Engine streaming interface is declared as an async generator, not a coroutine returning one
`BaseEngine.generate_stream` SHALL be declared such that calling it directly produces an
async generator, matching how every implementation defines it and how every caller uses
it. It SHALL NOT be declared such that a type checker infers calling it returns a
coroutine that must be awaited before iteration.

#### Scenario: Declared type matches implementation type
- WHEN a concrete engine implements `generate_stream` as an async generator function
  (its body contains `yield`)
- THEN the abstract declaration's return type is satisfied without a type checker
  reporting an invalid-override error

#### Scenario: Callers iterate without awaiting
- WHEN a caller invokes `generate_stream(request)`
- THEN the returned object is immediately usable as an async generator (`__anext__`,
  `aclose`) without first being awaited

#### Scenario: Change is applied
- WHEN this change is applied
- THEN `mypy` reports no errors for `services/chat_service.py`
- AND `inference_x.services.chat_service` is removed from the DEC-048 mypy baseline
- AND no other module's baseline entry grows
- AND no runtime behavior, streaming event order, or `ChatStreamChunk` shape changes

### Requirement: Compatibility — abstract and implementation denote the same callable type
The abstract declaration and every implementation MUST denote the same callable type.
The platform SHALL enforce this as a Compatibility Invariant stronger than
"no invalid override": type agreement MUST hold for the current engine and for every
future `BaseEngine.generate_stream` implementation.

#### Scenario: Declaration and implementation share one callable type
- WHEN a concrete engine implements `generate_stream`
- THEN its callable type is the same type denoted by the abstract declaration
- AND a type checker does not report an invalid-override error for that method

### Requirement: Compatibility — no caller changes are permitted
No caller changes are permitted. The platform SHALL keep ChatService, EngineDriver,
middleware, routing, benchmarks, and tests valid without adaptation after the
declaration correction.

#### Scenario: Callers remain valid without adaptation
- WHEN the abstract declaration is corrected
- THEN ChatService, EngineDriver, middleware, routing, benchmarks, and tests remain
  valid without edits
- AND no caller is adapted to await or otherwise reinterpret `generate_stream`

### Requirement: Future generate_stream implementations type-check without suppression
Future implementations of `BaseEngine.generate_stream` MUST type-check without
requiring suppression. The platform SHALL treat a new implementation that needs a
mypy baseline entry or other suppression for this contract as a regression.

#### Scenario: New implementation needs no generate_stream suppression
- WHEN a future concrete engine implements `generate_stream` against the corrected
  declaration
- THEN that implementation type-checks without a mypy baseline entry, `# type: ignore`,
  or other suppression for the `generate_stream` contract

### Requirement: Engine health signal is sourced from AsyncLLM, not a repo-local dead flag

`VLLMEngine.is_healthy()` SHALL derive its answer from `AsyncLLM`'s own
`errored` property, not from a repo-local driver dead-flag.

#### Scenario: Engine core process dies
- **WHEN** the underlying `AsyncLLM` engine-core process dies or the output
  handler task stops running
- **THEN** `AsyncLLM.errored` becomes `True`
- **AND** `VLLMEngine.is_healthy()` returns `False`
- **AND** no repo-local `EngineDriverDeadError` or equivalent wrapper type is
  introduced — `vllm`'s own `EngineDeadError` is used directly

#### Scenario: Engine core is alive
- **WHEN** the underlying `AsyncLLM` instance is running normally
- **THEN** `VLLMEngine.is_healthy()` returns `True`

### Requirement: Engine-side cancellation is observable, not merely consumer-side abandonment

The system SHALL abort the underlying engine computation when a streamed
generation is cancelled or its consumer disconnects — the request MUST NOT be
merely left unread by an abandoned consumer.

#### Scenario: Client disconnects mid-stream
- **WHEN** a client disconnects while `ChatService.stream_response` is
  consuming `VLLMEngine.generate_stream()`
- **THEN** the async generator receives `GeneratorExit`
- **AND** the engine is signaled to abort that request's in-flight computation
  (an engine-side abort call, not only ceasing to read from a queue)

#### Scenario: A later refactor must not silently regress this
- **WHEN** `VLLMEngine`'s streaming implementation is changed after this
  change lands
- **THEN** `GeneratorExit`/`CancelledError` delivered into the consumption of
  `AsyncLLM.generate()` MUST continue to reach `AsyncLLM`'s own cancellation
  path (propagated, not swallowed) — silently catching and discarding it
  would restore consumer-side-only abandonment, the exact behavior this
  requirement exists to rule out

### Requirement: pool_size > 1 is not guaranteed and not forbidden pending verified multi-instance support

The system SHALL preserve the existing public `pool_size` configuration
surface but MUST NOT claim any architectural guarantee about values greater
than 1 until `AsyncLLM` multi-instance support is explicitly verified. Only a
later phase that performs that verification (most likely B6, per
`PHASE-A-ARCHITECTURE.md` §10) SHALL redesign multi-engine serving.

#### Scenario: pool_size > 1 is configured
- **WHEN** `pool_size` resolves to a value greater than 1 at startup
- **THEN** `VLLMEngine` construction proceeds without a new construction-time
  rejection introduced by this change
- **AND** the platform makes no claim, in either direction, about whether
  co-located `AsyncLLM` instances behave correctly under concurrent load —
  this remains an open question until explicitly verified

#### Scenario: A later phase redesigns multi-engine serving
- **WHEN** a future phase verifies `AsyncLLM`'s multi-instance behavior and
  changes how multiple engines are hosted (e.g. splitting into separate
  processes)
- **THEN** that phase updates this requirement with the verified outcome —
  this requirement is a placeholder for an unresolved question, not a
  permanent architectural position

### Requirement: Non-streaming generation is derived from the streaming path

`VLLMEngine.generate()` SHALL be implemented in terms of
`VLLMEngine.generate_stream()` (or a shared internal primitive both are
derived from), not as an independent call into the underlying engine's
generate entrypoint.

#### Scenario: Exactly one call site reaches the engine's generate entrypoint
- **WHEN** either `VLLMEngine.generate()` or `VLLMEngine.generate_stream()` is
  called
- **THEN** both resolve to the same underlying call into `AsyncLLM.generate()`
- **AND** terminal usage metadata for both is produced by the same
  `derive_terminal_metadata` function (unchanged from its pre-migration
  behavior, relocated per DEC-050 §3)

#### Scenario: Implementation technique may change without breaking this requirement
- **WHEN** a future refactor changes *how* `generate()` derives from
  `generate_stream()` (e.g. draining the async generator vs. calling a shared
  internal primitive)
- **THEN** the requirement still holds as long as exactly one code path calls
  `AsyncLLM.generate()` and both public methods are expressed in terms of it

### Requirement: Native engine metrics are exposed as a separate, additive Prometheus endpoint

The system SHALL expose vLLM's own Prometheus stat logger as `GET /metrics`,
in vLLM's native text exposition format, without modifying or being fed by
`GET /v1/metrics`.

#### Scenario: GET /metrics returns vLLM's own Prometheus text exposition
- **WHEN** a client sends `GET /metrics`
- **THEN** the response has a Prometheus text-exposition `Content-Type`
- **AND** the body includes at least one `vllm:`-prefixed metric series
  sourced from vLLM's own `PrometheusStatLogger`
- **AND** no InferenceX code translates, renames, or reinterprets those
  series — the endpoint is a passthrough mount

#### Scenario: /metrics scrapes are excluded from /v1/metrics aggregation
- **WHEN** `GET /metrics` is scraped one or more times
- **THEN** `GET /v1/metrics`'s `total_requests`, `avg_latency_ms`, and
  `p95_latency_ms` are unaffected by those scrapes
- **AND** no `RequestRecord` is created for a `/metrics` request

#### Scenario: pool_size > 1 engines are not silently mislabeled
- **WHEN** the runtime is configured with `pool_size > 1`
- **THEN** a `WARNING` is logged at startup naming the metric-label
  collision risk between multiple `AsyncLLM` instances' default
  `PrometheusStatLogger`s
- **AND** no construction-time validation rejects or alters the
  configuration — `pool_size > 1` remains not-guaranteed-and-not-forbidden

#### Scenario: GET /v1/metrics is unaffected by this change
- **WHEN** this change is applied
- **THEN** `GET /v1/metrics`'s response schema (`MetricsResponse`) and every
  field's computation are byte-for-byte unchanged from before this change
- **AND** no `vllm:*` data feeds any `/v1/metrics` field

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

### Requirement: Admission control stays independent of observability surfaces
The system SHALL make every admission decision (context length, KV-pool
pressure, or sequence concurrency) using only state InferenceX tracks directly
or reads from the engine's synchronous interface, and SHALL NOT read from or
depend on `GET /metrics` or `GET /v1/metrics` to decide whether to admit,
clamp, or reject a request.

#### Scenario: Metrics scraping never influences admission
- **WHEN** any admission decision is made for any gate
- **THEN** the decision does not query, scrape, or otherwise read the
  Prometheus registry or the `/v1/metrics` in-memory store
- **AND** `GET /metrics` and `GET /v1/metrics` continue to report
  observability data only, unaffected by admission-control behavior

#### Scenario: Observability endpoints remain unaffected by this change
- **WHEN** this change (in any future implementation phase) is applied
- **THEN** `GET /metrics` (vLLM-native Prometheus registry) and
  `GET /v1/metrics` (HTTP-boundary aggregates) continue to report exactly the
  same data they reported before this change
- **AND** neither endpoint gains a new dependency on `AdmissionController` or
  vice versa

### Requirement: Run identity is a content hash of the run manifest
The platform SHALL identify each completed chat completion by a `run_id` that is a content hash of a
canonical serialization of the run manifest's semantics-determining fields, and SHALL treat two runs
as comparable if and only if their `run_id`s are equal. Comparability here means configuration
identity — `engine`, `model`, `runtime`, `sampling`, `request`, and each warning's `type`/`code`/
`field` — not execution-environment identity: `hardware` is bound to the manifest as provenance but
SHALL NOT participate in the `run_id` preimage, and a warning's free-text `message` SHALL NOT
participate either. The `run_id` SHALL be reproducible from the manifest and MUST NOT be a random
identifier. The platform SHALL NOT produce a cryptographic signature for the manifest in this
capability; run identity is by content address only.

#### Scenario: Identical configuration yields an identical run_id
- **WHEN** two requests produce manifests whose `engine`, `model`, `runtime`, `sampling`, `request`,
  and per-warning `type`/`code`/`field` are equal
- **THEN** both manifests have the same `run_id`

#### Scenario: A semantics field difference changes the run_id
- **WHEN** two manifests differ in any field within `engine`, `model`, `runtime`, `sampling`, or
  `request`, or in any warning's `type`, `code`, or `field`
- **THEN** their `run_id`s differ

#### Scenario: Speed, observational, and execution-environment blocks do not affect the run_id
- **WHEN** two manifests differ only in `timing`, in the `batch` block, or in the `hardware` block
- **THEN** their `run_id`s are equal

#### Scenario: A warning's free-text message does not affect the run_id
- **WHEN** two manifests' warnings are equal in `type`, `code`, and `field` for every warning, and
  differ only in a warning's `message` text
- **THEN** their `run_id`s are equal

#### Scenario: run_id is a content hash, not a signature
- **WHEN** a manifest is produced
- **THEN** its `run_id` is recomputable by hashing the same canonical preimage
- **AND** no cryptographic signature or key material is produced

### Requirement: Non-streaming chat completions carry a run identifier header
The platform SHALL emit an `X-Run-Id` HTTP header on every non-streaming
`POST /v1/chat/completions` response, carrying the run manifest's `run_id`. The platform SHALL NOT
be required to emit a manifest surface on the streaming path in this capability.

#### Scenario: Non-streaming response carries X-Run-Id
- **WHEN** a non-streaming chat completion is served
- **THEN** the HTTP response includes an `X-Run-Id` header equal to the manifest's `run_id`

#### Scenario: Streaming response carries no manifest surface in this capability
- **WHEN** a streaming chat completion is served
- **THEN** no `X-Run-Id` header and no manifest event are required of this capability

### Requirement: The run manifest attests the computation from existing signals
The platform SHALL assemble a versioned run manifest that records the provenance of the computation,
and SHALL derive it from the per-request signals the platform already produces — the resolved
request, the typed warnings, and the engine timing — without recomputing token counts, timings, or
warnings. The manifest SHALL carry a `manifest_version`. A provenance value the platform cannot
determine SHALL be omitted or recorded as null, and MUST NOT be fabricated, approximated, or
defaulted to a fictitious value.

#### Scenario: Manifest reuses existing per-request signals
- **WHEN** a chat completion has produced a resolved block, warnings, and timing
- **THEN** the manifest's corresponding fields are populated from those existing signals rather than
  recomputed

#### Scenario: Undeterminable provenance is recorded honestly
- **WHEN** a provenance value (for example a model weight hash or model revision) cannot be
  determined
- **THEN** that field is omitted or null in the manifest
- **AND** the manifest records no fabricated or approximated value in its place

#### Scenario: Manifest is versioned
- **WHEN** a manifest is produced
- **THEN** it carries a `manifest_version`

### Requirement: Batch co-batching identity is reserved but not populated
The platform SHALL define `batch.co_batched_request_ids` in the run manifest schema as the reserved
location for engine co-batch composition, and SHALL NOT populate it with engine-internal composition
data in this capability. The field MUST NOT carry fabricated composition data.

#### Scenario: The field exists in the schema
- **WHEN** the run manifest schema is defined
- **THEN** it includes `batch.co_batched_request_ids`

#### Scenario: No composition data is emitted in this capability
- **WHEN** a manifest is produced
- **THEN** `batch.co_batched_request_ids` carries no engine-derived composition data
- **AND** it carries no fabricated composition data

### Requirement: The manifest asserts comparability, not reproducibility
The platform and its documentation SHALL NOT claim end-to-end determinism or that runs are
reproducible on the basis of the seed or the manifest alone. They MAY state the manifest's
comparability property: that equal `run_id`s denote the same recorded configuration. Hardware is
recorded on the manifest as execution-environment provenance and is not part of the `run_id`
preimage; two runs of the same configuration on different hardware MAY therefore share a `run_id`,
and a consumer SHALL use the manifest's `hardware` block, not the `run_id`, to judge whether a
comparison between them is reproducible-grade or only configuration-comparable. A scoped
determinism contract is out of scope for this capability.

#### Scenario: Comparability is claimed, reproducibility is not
- **WHEN** documentation describes the run manifest
- **THEN** it states that equal `run_id`s denote the same recorded configuration
- **AND** it does not claim the server is deterministic or that runs reproduce end-to-end

#### Scenario: A recorded seed does not imply reproducibility
- **WHEN** the manifest records a `sampling.seed`
- **THEN** neither the manifest nor its documentation claims the run is reproducible from that seed
  alone

#### Scenario: Cross-hardware runs of the same configuration can share a run_id
- **WHEN** two runs share an identical `run_id`
- **THEN** their manifests MAY report different `hardware` blocks
- **AND** a consumer reads the `hardware` block, not the `run_id`, to judge whether comparing them is
  reproducible-grade or only configuration-comparable

