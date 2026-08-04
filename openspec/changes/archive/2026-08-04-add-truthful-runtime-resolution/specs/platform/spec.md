# Platform Specification Delta — add-truthful-runtime-resolution

## ADDED Requirements

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

## MODIFIED Requirements

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
