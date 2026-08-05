## 1. Fix (ChatService-exclusive)

- [x] 1.1 In `src/inference_x/services/chat_service.py`, uphold the reservation lifetime
      invariant in `stream_response`: from the instant `admit()` successfully returns
      until the stream generator terminates for any reason (normal completion, timeout,
      engine failure, cancellation, `GeneratorExit`, or client disconnect), exactly one
      matching `release()` MUST occur. Cover every post-`admit()` suspension point,
      including the DEC-053 prologue currently outside the lifetime guard.
- [x] 1.2 Uphold the opposite invariant: a reservation MUST NEVER be released more than
      once. Prefer a structure with a single `release()` call site so double-release is
      structurally impossible.
- [x] 1.3 Do not change the prologue event's content, position, or the order of any
      other event. Do not touch `routing/admission.py`, `observability/middleware.py`,
      or any engine. Middleware and engines must not compensate; Admission must not
      compensate for caller failures. The fix belongs exclusively in ChatService.
- [x] 1.4 Confirm `admit()` itself remains outside any cleanup that calls `release()` —
      a reservation is only made if `admit()` returns successfully, so there is nothing
      to release if it raises.

## 2. Tests

- [x] 2.1 Add a regression test using the existing stub-engine test harness: start
      `stream_response`, consume only the prologue event, call `aclose()`, and assert
      both the KV reservation and the sequence-concurrency slot for the routed model
      return to zero (the exact scenario reproduced during the Phase A audit: 522 KV
      tokens / 1 sequence slot leaked on unpatched code).
- [x] 2.2 Confirm existing tests covering full-consumption, timeout, and engine-error
      paths still pass unchanged — they already release today and must continue to
      release exactly once.
- [x] 2.3 Add validation that detects double-release regressions: for a single
      reservation, observe that `release()` executes exactly once. Validation MUST fail
      if `release()` executes twice. Do not prescribe the testing technique; require only
      this observable property.

## 3. Validation

- [x] 3.1 Termination matrix (every row: exactly one `release()`):
      - normal completion
      - timeout
      - engine exception
      - `CancelledError`
      - `GeneratorExit` before first token (including prologue)
      - `GeneratorExit` after first token
      - client disconnect
- [x] 3.2 Double-release property: validation fails if `release()` executes twice for
      one reservation.
- [x] 3.3 `ruff check .` clean.
- [x] 3.4 `mypy src/` clean, no new baseline entries.
- [x] 3.5 `pytest tests/unit/` — all existing tests pass unchanged, plus the new
      regression coverage from 2.1 and 2.3.
- [x] 3.6 `openspec validate fix-admission-reservation-leak --strict` passes.

## Out of scope checklist (must remain undone)

- [ ] Confirm no admission gating, KV/context clamping formula, or `AdmissionResult`
      field was changed.
- [ ] Confirm no DEC-053 event order, event schema, or event content was changed.
- [ ] Confirm no Engine Boundary (`engines/base.py`) or engine implementation change
      was made — that is SEV-A2/SEV-A3, tracked separately.
- [ ] Confirm no `observability/middleware.py` change was made.
- [ ] Confirm no new ADR was added or existing ADR amended — this closes a gap between
      code and `admit()`'s own pre-existing documented contract, not a new decision.
- [ ] Confirm Phase B carry-forward language is MUST (preserve the lifetime invariant),
      not SHOULD.
