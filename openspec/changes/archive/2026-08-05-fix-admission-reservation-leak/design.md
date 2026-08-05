## Context

`ChatService.stream_response` (`services/chat_service.py:92-209`) is an async generator
with this shape today:

```
routed_model, engine = self._resolve_engine(request)          # :125
admitted = self._admission.admit(routed_model, request, engine)  # :126
effective_request = request.model_copy(...)                   # :127-129
...
yield _event({...prologue...})                                # :139  <- OUTSIDE LIFETIME GUARD
gen = engine.generate_stream(effective_request)                # :150
try:                                                            # :150
    ...content/terminal/usage events...
finally:                                                        # :205
    self._admission.release(routed_model, admitted.reserved_tokens)  # :207
yield "data: [DONE]\n\n"                                       # :209
```

The prologue `yield` at `:139` is a suspension point: control returns to the caller
(ASGI/Starlette, or a test driving the generator directly) and does not resume until the
caller asks for the next item — or never resumes, if the caller closes the generator
instead. If closure happens at that exact suspension point, `GeneratorExit` is raised
*at* the `yield`, which is textually before the region that calls `release()` — so
`release()` never runs.

This is DEC-053's own construction: the prologue was deliberately placed at the head of
the function ("Event 0 sits at the head rather than before `[DONE]`... a trailer would be
lost on the timeout path", `chat_service.py:117-120`), which is correct for what DEC-053
optimizes for, but it introduced a new suspension point after `admit()` without extending
reservation lifetime coverage to that point.

## Lifecycle invariant (normative)

From the instant `admit()` successfully returns until the stream generator terminates
for any reason (normal completion, timeout, engine failure, cancellation,
`GeneratorExit`, or client disconnect), exactly one matching `release()` MUST occur.

This invariant is the durable contract. The Phase A implementation may uphold it by
restructuring exception-handling boundaries in `ChatService.stream_response`. Phase B
may change that mechanism. The invariant MUST survive those refactors.

### Opposite invariant (no double release)

A reservation MUST NEVER be released more than once.

Leak prevention and double-release prevention are independent Compatibility Invariants.
An implementation that closes the leak by adding a second `release()` call site that can
run after the first has already run is incorrect.

## Termination matrix

Every stream termination path MUST satisfy the lifecycle invariant. Expected reservation
outcome for every row: **exactly one `release()`**.

| Termination path | Expected reservation outcome |
|---|---|
| Normal completion | exactly one `release()` |
| Timeout | exactly one `release()` |
| Engine exception | exactly one `release()` |
| `CancelledError` | exactly one `release()` |
| `GeneratorExit` before first token (including during pre-generation prologue) | exactly one `release()` |
| `GeneratorExit` after first token | exactly one `release()` |
| Client disconnect | exactly one `release()` |

This matrix is part of implementation validation. An implementation is incomplete if any
row can leave a reservation held, or if any row can invoke `release()` more than once.

## Ownership boundaries

| Owner | Responsibility | Must never |
|---|---|---|
| Admission (`routing/admission.py`) | Reservation accounting (`admit` / `release` trackers and policy inputs) | Compensate for caller failures; auto-release on behalf of ChatService |
| ChatService (`services/chat_service.py`) | Reservation **lifetime** — uphold the lifecycle invariant for every stream | Rely on middleware or engine to release |
| Middleware (`observability/middleware.py`) | Observability wrapping of the response stream | Compensate for ChatService lifetime gaps (e.g. by adding release logic) |
| Engine code (`engines/*`) | Inference execution | Compensate for ChatService lifetime gaps; call `release()` |

The fix belongs exclusively in ChatService. Admission, middleware, and engines are out of
bounds for this change.

## Goals / Non-Goals

**Goals:**
- Uphold the lifecycle invariant: exactly one matching `release()` for every successful
  `admit()` until stream-generator termination for any reason.
- Uphold the opposite invariant: a reservation is never released more than once.
- The DEC-053 event order, event content, and event count are byte-identical to today for
  every path that does not involve early disconnect. A client that consumes the full
  stream, or that reads until a normal timeout/error event, observes no difference
  whatsoever.
- The fix is scoped to `ChatService.stream_response`. It does not depend on, and does not
  require, a corresponding change in `observability/middleware.py`.

**Non-Goals:**
- Do not change admission gating logic, the KV/context clamping formulas, or any
  `AdmissionResult`/`AdmittedRequest` field (`routing/admission.py` is not touched).
- Do not change the DEC-053 event order, event schema, or the prologue's content.
- Do not widen or redesign the Engine Boundary (`engines/base.py` is not touched — that is
  SEV-A2/SEV-A3, a separate change).
- Do not add `body_iterator.aclose()` handling to `observability/middleware.py`. The
  middleware's reliance on GC-driven finalization for the *outer* generator is a
  pre-existing, separate characteristic of how Starlette's `StreamingResponse` is
  consumed; it is not this change's concern and is not altered by it. This change closes
  the gap at the one place a fix is both correct and sufficient: the generator that owns
  the admission reservation must release it on its own closure, regardless of how or when
  its caller closes it.
- No AsyncLLM work, no Phase B scope of any kind.

## Decisions

### Lifetime coverage is the requirement; mechanism is secondary

The defect is a lifetime gap: after a successful `admit()`, there exists a stream
suspension point at which termination does not produce a matching `release()`. Closing
that gap is mandatory. How Phase A does so (for example by ensuring a single cleanup
region covers every post-`admit()` suspension point, including the DEC-053 prologue) is
an implementation choice subordinate to the invariant.

Alternative considered: add a second, prologue-local cleanup that also calls `release()`,
leaving the existing mid-stream cleanup untouched. Rejected — that creates two
`release()` call sites for one reservation and risks violating the opposite invariant
(double release). Prefer a structure in which double-release is structurally impossible
rather than merely avoided by careful sequencing. Analogous to DEC-050's "one call site"
discipline for terminal metadata.

### No new exception type, no new warning

`GeneratorExit` at the prologue is not a new failure mode requiring a client-visible
signal — the client that triggered it has already disconnected and cannot receive
anything. `release()`'s existing behavior (decrement both trackers) is sufficient; no
`AdmissionController` change is needed.

## Risks / Trade-offs

- Extending lifetime coverage brings additional statements into the cleanup region
  (building `effective_request`, computing `completion_id`/`timeout_s`/`include_usage`,
  defining `_event`, and the prologue yield). Those statements are either pure/non-raising
  or already execute unconditionally before any of this ships; none of them may newly
  cause `release()` to run when it previously would not have on the success path, and
  none may introduce a second `release()`.
- No change to what happens on the *existing* early-disconnect paths (mid-stream, after
  the terminal event, during timeout handling) — those already release today and must
  continue to release exactly once.

## Migration Plan

1. In `ChatService.stream_response`, ensure reservation lifetime coverage begins
   immediately after a successful `admit()` and continues until the stream generator
   terminates for any reason, such that exactly one matching `release()` occurs.
2. Ensure `admitted = self._admission.admit(...)` itself remains outside any cleanup that
   calls `release()` (consistent with `AdmissionResult`'s existing role — `release()` is
   only ever called with an `admitted` value that successfully returned from `admit()`;
   `admit()` raising means no reservation was made and there is nothing to release).
3. Add regression tests that (a) drive `stream_response` against the existing stub-engine
   harness, consume only the prologue event, call `aclose()`, and assert both the KV
   tracker and the sequence-concurrency tracker for the routed model return to zero; and
   (b) observe that `release()` executes exactly once for a single reservation —
   validation MUST fail if `release()` executes twice.
4. Re-run the full gate (`ruff`, `mypy`, `pytest`) — no existing test's expected output
   should change, since no event content or ordering changes on any already-tested path.

## Phase B carry-forward

Phase B roadmap B4 rescopes admission to what the vLLM scheduler cannot do and will
delete or rewrite the KV gate. After this fix lands, B4 inherits a `stream_response` that
already upholds the reservation lifetime invariant. B4's job is to change *what* is gated
and *how it's computed*, not to re-discover that lifetime coverage.

Phase B admission refactoring MUST preserve the reservation lifetime invariant:

From the instant `admit()` successfully returns until the stream generator terminates
for any reason (normal completion, timeout, engine failure, cancellation,
`GeneratorExit`, or client disconnect), exactly one matching `release()` MUST occur.

A reservation MUST NEVER be released more than once.

The Phase A mechanism may change; these invariants MUST NOT.
