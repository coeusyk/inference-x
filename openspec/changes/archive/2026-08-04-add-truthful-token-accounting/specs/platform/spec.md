# Platform Specification Delta — add-truthful-token-accounting

## ADDED Requirements

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
