# Platform Specification Delta — fix-admission-reservation-leak

## ADDED Requirements

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
