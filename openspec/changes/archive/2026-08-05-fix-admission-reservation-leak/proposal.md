## Why

`AdmissionController.admit()` documents its own caller contract: *"The caller MUST call
release() with that same reserved_tokens value once the request completes (success or
failure), typically from a try/finally around engine dispatch"* (`routing/admission.py:225-227`).
`release()` decrements both the KV-token tracker and the per-model sequence-concurrency
tracker (`admission.py:357-358`).

`ChatService.stream_response` violates this contract on one path. `admit()` is called at
`services/chat_service.py:126`. The DEC-053 pre-generation prologue event is `yield`-ed at
`:139`, but the `try/finally` that calls `self._admission.release(...)` does not begin
until `:150`. Everything between `admit()` returning and the prologue `yield` completing —
including the suspension point of the `yield` itself — sits outside the guard.

Reproduced directly (stub engine, `aclose()` called immediately after consuming only the
prologue event): full consumption releases both trackers to zero; early disconnect at the
prologue leaves the KV reservation and one sequence-concurrency slot permanently held.
`_seq_tracker` is compared against `max_num_seqs` and never decays on its own — repeated
disconnects at this exact point exhaust a model's concurrency ceiling until the process
restarts.

Traced (this change) at the production call site: `observability/middleware.py:295`'s
`_wrap_and_record_sse` consumes `ChatService.stream_response`'s generator via
`async for chunk in body_iterator`, with no explicit `body_iterator.aclose()` in its own
`finally` (`middleware.py:299-329`). A client disconnect closes the *outer* generator
(Starlette delivers `GeneratorExit` there, per that function's own docstring at
`middleware.py:274-276`), but nothing in the wrapper explicitly propagates that closure to
the inner one — cleanup of `body_iterator` is left to async-generator finalization at
garbage-collection time, which is not deterministic and not guaranteed to run promptly.
This confirms, from source, the propagation path the architecture audit
(`docs/REVIEW-2026-08-04-phase-a-final-audit.md`, finding SEV-A1) flagged as inferred.

## What Changes

- Establish and enforce a reservation lifetime invariant in `ChatService.stream_response`:
  from the instant `admit()` successfully returns until the stream generator terminates
  for any reason, exactly one matching `release()` MUST occur. The concrete mechanism
  that upholds the invariant today (exception-handling region placement) is an
  implementation detail; the invariant is the contract.
- Add regression coverage that the reservation returns to zero on early disconnect after
  the prologue, and that `release()` is never executed more than once for a single
  reservation.

## Capabilities

### Modified Capabilities
- `platform`: admission reservations made by `admit()` are now released exactly once on
  every stream termination path out of `ChatService.stream_response`, including
  disconnect during the pre-generation prologue. No other admission behavior changes.
  Double-release is forbidden.

## Impact

- Code: `src/inference_x/services/chat_service.py` only. No change to
  `routing/admission.py`, `observability/middleware.py`, or any engine.
- Ownership: ChatService owns reservation lifetime; Admission owns reservation
  accounting; middleware and engine code must not compensate; Admission must not
  compensate for caller failures. The fix belongs exclusively in ChatService.
- Tests: regression coverage for leak and double-release properties; no existing test's
  expected behavior changes.
- Docs: none required — this closes a gap between code and an already-documented
  contract (`admit()`'s own docstring); no ADR is added or amended.
- Wire format: none. Event order, event content, and timing are unchanged for every
  client-visible path (normal completion, timeout, engine error). Only the *internal*
  bookkeeping timing of `release()` on the early-disconnect path changes.
