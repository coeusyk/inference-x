# Proposal: rescope-admission-control

## Status

**Design decided (Option C). Investigation and design only — no implementation
in this change.** Round 2 of this investigation presented three options and a
recommendation without authority to decide. The change owner named Option C
the working direction; round 3 fully specified it (semaphore lifecycle,
keying, timeout/429 contract) and closed the three remaining evidence gaps
(model-size generalization, genuine cancellation-while-queued, genuine
engine-failure-while-queued). Option C is now the selected design. No
`src/`, `tests/`, or `pyproject.toml` file is touched by this change — that
remains implementation-phase work, gated on `tasks.md` §2. See `design.md`
for full evidence, including the "Decision" and "Semaphore lifecycle
analysis" sections, and `tasks.md` for exactly what is and is not checked
off.

## Summary

B4 was framed as "re-scope admission control now that B1 (AsyncLLM), B2 (native
metrics), and B3 (per-request timing) have shipped." The investigation confirms
part of that premise — Gate 1 (the sequence-concurrency ceiling, `_seq_tracker`)
does structurally duplicate a capacity invariant vLLM's own scheduler already
enforces via queueing rather than rejection — but it also surfaces a blocking gap
that makes a naive "just narrow Gate 1" change unsafe: `ChatService.complete()`
has no timeout of any kind around `await engine.generate(...)`. Removing or
narrowing Gate 1 without addressing that gap trades a fast, explicit 429 for an
unbounded hang on the non-streaming path.

This proposal does not pick a fix. It documents two viable options and asks the
change owner to choose one before implementation tasks begin.

**Correction from an earlier draft:** an earlier pass claimed
`ChatService.complete()` has no timeout at all. That was wrong — it missed
that `VLLMEngine.generate()`, one layer down, already wraps its body in a
300-second `asyncio.timeout`. Non-streaming is bounded today, just not
cheaply: a timeout there raises a generic `RuntimeError` mapped to **HTTP
500** with no `Retry-After` header, not the fast, typed 429 Gate 1 gives
clients today. This changes what "the companion decision" actually is — see
below and `design.md` → "Central finding."

## Why B4 exists after B1/B2/B3

- **B1** (`migrate-async-llm-engine`) replaced the offline `LLM` + hand-rolled
  `EngineDriver` with vLLM's `AsyncLLM`. `AsyncLLM.generate()` is the same
  target vLLM's scheduler manages internally — it accepts a request unconditionally
  via `add_request()` and only fails on client-initiated cancellation or
  engine-shutdown, never on capacity [source-derived: `vllm/v1/engine/async_llm.py`
  lines 524-598, `vllm/v1/engine/core.py:1329`]. Before B1, InferenceX had no
  engine that could safely queue on its own, so a local "reject fast" gate was
  the only realistic capacity story.
- **B2** (`expose-native-engine-metrics`) exposed vLLM's own `num_requests_running`
  / `num_requests_waiting` via `GET /metrics`, giving external observers the same
  queue-depth signal the scheduler itself uses — a signal that did not exist when
  Gate 1 was designed.
- **B3** (`expose-per-request-engine-timing`) exposed `queue_time_ms` per request,
  making the cost of vLLM-side queueing visible to clients for the first time.

None of B1-B3 changed the fact that `ChatService.complete()` has no bound on how
long it will wait for `engine.generate()` to return. That gap predates B1 and is
untouched by it — it is the central fact this investigation surfaces.

## Current admission behavior (as of this investigation)

`AdmissionController` (`src/inference_x/routing/admission.py`) runs three
independent, synchronous gates before every dispatch to the engine:

1. **Gate 1 — sequence concurrency** (`_seq_tracker`, a `_PerModelCounter`):
   rejects with `EngineSaturatedError` (HTTP 429 + `Retry-After`) when a model's
   in-flight request count is at or above its resolved `max_num_seqs`. No clamp
   path exists for either `interactive` or `batch` priority — introduced by
   DEC-040 (2026-07-01), documented below.
2. **Gate 2 — context length**: rejects with `ContextTooLongError` (HTTP 400)
   when prompt + requested output tokens exceed the resolved context ceiling.
   Not part of this investigation's scope — this enforces a limit vLLM's own
   tokenizer/scheduler would also hard-reject on, and the HTTP class (400, not
   429) reflects a client input error, not a capacity condition.
3. **Gate 3 — KV-pool pressure** (`_tracker`, a separate `_PerModelCounter`):
   clamps `max_tokens` down for `interactive` priority, or rejects with
   `EngineSaturatedError` for `batch` priority, when in-flight token reservations
   approach `engine.kv_capacity_tokens * 0.9`.

## Concrete evidence of duplicated responsibility (Gate 1 only)

vLLM 0.22.1's scheduler (`vllm/v1/core/sched/scheduler.py`) treats
`max_num_seqs` (`self.max_num_running_reqs`) as a **per-tick promotion throttle**
inside `schedule()`, not an admission gate: `add_request()` (lines 1755-1785)
enqueues unconditionally onto `self.waiting`, and a request that can't be
promoted into `self.running` this tick simply waits for the next tick. The only
rejection path in the entire request lifecycle is engine shutdown
(`core.py:1329`). Under KV pressure, vLLM **preempts already-running** requests
back onto the waiting queue rather than rejecting new ones (`scheduler.py:929`,
call sites at lines 483 and 1916).

This means: today, a client whose request arrives when a model is momentarily at
its `max_num_seqs` ceiling gets an immediate 429, even though vLLM itself would
have queued that request safely and served it as soon as a running slot freed —
exactly the scenario the "current vs. potential" flow the investigation task
asked to distinguish. [source-derived, not live-measured — see
`design.md` → "Empirical findings" for why no live experiment was run.]

**However** — and this is the finding that changes the proposed direction — this
duplication was not an oversight. DEC-040 explicitly names it as the reason Gate
1 was added: *"a saturated sequence-concurrency ceiling just queued silently
inside vLLM's scheduler instead of surfacing as an explicit signal."* The
author already knew vLLM would queue; they judged silent, client-invisible
queueing worse than an explicit, actionable 429. That judgment call has not
been invalidated by anything found in this investigation — if anything, the
`ChatService.complete()` timeout gap makes it look more prescient, not less.

## Intended behavioral change — Option C selected, A and B rejected

Narrowing or removing Gate 1 is only safe if something else bounds how long a
non-streaming client can be left waiting once a request reaches vLLM's queue.
Three ways to provide that bound were identified across two investigation
rounds. They are **not equivalent** — Option C is now the selected design;
A and B are documented below with the concrete reasons they were rejected:

**Option A — bound the wait at the HTTP boundary, dispatch to vLLM's queue.**
Narrow or remove Gate 1 so requests reach `AsyncLLM.generate()` and let vLLM's
real scheduler decide running-vs-waiting, relying on `VLLMEngine.generate()`'s
existing 300s `asyncio.timeout` as the backstop rather than adding a new one.
This requires deciding whether 300s and a generic 500 (today's timeout
behavior) is acceptable, or whether the timeout needs to shrink and/or its
failure needs to become a typed, `Retry-After`-bearing response to match what
Gate 1 gives clients today. Tradeoff: the request *does* reach the engine
before a possible failure — "the engine itself is never invoked for a
rejected request" (today's invariant, and the current spec's stated scenario)
would no longer hold, and it occupies a real vLLM queue slot for up to the
timeout window under sustained overload.

**Option B — keep admission-time gating, make it bounded instead of instant.**
Convert Gate 1 from an instant reject into a bounded wait inside
`AdmissionController.admit()` itself (e.g. wait up to N seconds for a slot to
free before raising `EngineSaturatedError`). Preserves today's invariant that a
rejected request never reaches the engine, and keeps `AdmissionController` as
the single synchronous decision point the codebase's ownership boundaries
(`docs/PHASE-A-ARCHITECTURE.md` §9: `routing/` "owns... whether a request may
run") already describe. Tradeoff: InferenceX ends up re-implementing a bounded
version of the exact queueing vLLM's scheduler already does, on stale
information (`_seq_tracker`'s local count vs. vLLM's live running/waiting
state) rather than direct engine state.

**Option C — bounded per-model semaphore. SELECTED.** Replaces
`_seq_tracker`'s counter-plus-instant-reject with a per-model
`asyncio.Semaphore(max_num_seqs)` acquired with a deadline inside
`AdmissionController.admit()`. Preserves "engine never invoked for a rejected
request," preserves the 429 + `Retry-After` response shape DEC-040
established, reuses the existing release/cancellation machinery already
covered by `test_chat_service.py`'s 9 lifecycle tests, and — unlike Option
A — does not require trusting vLLM's queue-depth behavior at any model size,
since the bound is InferenceX's own choice. `design.md` → "Decision: Option C
approved" records the full rationale; "Semaphore lifecycle analysis" gives
the exact design (including a corrected release-path hazard the original
sketch missed); "Acquisition, rejection, and timeout semantics" gives the
429/timeout contract. The one remaining open item is the exact
`admission_wait_s` default for model sizes this investigation's GPU doesn't
represent — a config-surface question with a provisional default, not a
blocker to the design itself; see `design.md` for why.

**Live evidence, briefly.** Across two investigation rounds, live experiments
against the actual installed vLLM 0.22.1 (`opt-125m`, RTX 4060,
`max_num_seqs=4`) confirmed: 48/48 requests submitted at 4x the ceiling all
completed successfully (worst case 1.13s); completion latency stayed flat
across 90 requests arriving steadily over 45s (no starvation, no unbounded
growth); a request genuinely waiting behind the ceiling (verified via
zero-output-tokens-observed) was cleanly cancelled with no engine-error
state and no sibling impact; a fresh request after that cancellation
completed normally. Model-size generalization of the queue-vs-reject
*structural* property, and genuine engine-failure-while-queued, were both
closed by direct vLLM 0.22.1 source tracing rather than a live benchmark —
see `design.md` → "Round 3 — closing the two remaining gaps" for why a live
7B run wasn't needed to answer those two specific questions.

All three options remove the specific false-429 case this investigation found
(instant rejection despite the ceiling being about to free); they differ in
where the wait happens and in whether "engine never invoked for a rejected
request" survives as an invariant. `specs/platform/spec.md` in this change
now carries a `## MODIFIED Requirements` section replacing the
"Sequence-concurrency ceiling" scenario's instant-rejection text with
bounded-wait-then-429 semantics.

## Explicit non-goals

- **Gate 2 (context length)** — untouched. Not a capacity/scheduling question.
- **Gate 3 (KV-pool pressure, `_tracker`)** — retained as-is. It protects a
  tier-aware, priority-differentiated (clamp-vs-reject) policy that has no
  scheduler equivalent, since InferenceX never sets vLLM's `priority` kwarg and
  therefore never activates `SchedulingPolicy.PRIORITY`. Whether Gate 3 itself
  produces false 429s in the same sense as Gate 1 was **not** established this
  round (see `design.md` → "Gate 3 — open question, not resolved") and is
  explicitly out of scope for this change.
- **Same-model horizontal replication / `pool_size > 1` admission** —
  `EnginePool._engines` is a strict `dict[str, BaseEngine]`, one entry per model
  name; same-model replicas are structurally unrepresentable today. This is B6
  territory (process-splitting), not reachable by any change to
  `AdmissionController` alone.
- **Turning `/metrics` or `/v1/metrics` into an admission input.** Both remain
  observability-only; see the ADDED requirement in this change's spec delta.
- **Changing vLLM's `priority` kwarg usage** or adopting `SchedulingPolicy.PRIORITY`.

## Compatibility impact

No compatibility impact in this phase — nothing in `src/`, `tests/`, or
`pyproject.toml` is modified by this change. At implementation time, Option
C preserves the existing HTTP contract (`429` + `Retry-After` remains the
response shape for genuine, sustained saturation) but changes
`AdmissionController.admit()`'s internal signature from synchronous to
`async def` — an internal interface change, not an HTTP-facing one; see
`design.md` → "Semaphore lifecycle analysis" for why that's required and
"Compatibility invariants" for the concurrency-analysis follow-on it implies.

## Affected components (once implemented, either option)

`src/inference_x/routing/admission.py`, `src/inference_x/services/chat_service.py`,
possibly `src/inference_x/schemas/chat.py` (timeout/config surface),
`tests/unit/test_admission.py`, `tests/unit/test_chat_service.py`.

## Relationship to B5/B6

Not blocking either. B6 (process-splitting for same-model replicas) would need
its own admission redesign regardless of which option this change picks, since
per-model counters assume one engine per model name today.
