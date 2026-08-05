# Platform Specification Delta — fix-engine-driver-late-submit-race

## ADDED Requirements

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
