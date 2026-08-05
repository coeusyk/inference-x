# Platform Specification Delta — migrate-async-llm-engine

## MODIFIED Requirements

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

## ADDED Requirements

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
