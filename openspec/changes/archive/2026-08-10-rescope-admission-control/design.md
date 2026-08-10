# Design: rescope-admission-control

## Evidence tagging

Every substantive claim below is tagged:
- **[verified]** — read directly from the currently-installed source (repo or
  `.venv/.../vllm/`) or produced by an actual command run this session.
- **[source-derived]** — a conclusion reasoned from [verified] facts, not itself
  independently executed (e.g. "vLLM would queue this" reasoned from reading
  `add_request()`, not from running a live burst against a GPU).
- **[inference]** — a judgment call synthesizing multiple source-derived facts,
  offered as this investigation's opinion, not a directly observable fact.
- **[unverified]** — explicitly flagged where something could not be checked
  (e.g. behavior on a different vLLM version) so it isn't silently treated as
  settled.

## vLLM version scope — a live compatibility risk, not a footnote

**[verified]** `pyproject.toml` pins `vllm>=0.6.0` — a floor, not an exact
version. **[verified]** The actual installed/tested version this whole
investigation is based on is vLLM **0.22.1** (from `uv.lock`). **[inference]**
Nothing in `pyproject.toml` prevents a future `uv sync`/lockfile update from
resolving to a vLLM version where `add_request()` gains a capacity check, where
`max_num_seqs` becomes a hard admission ceiling instead of a per-tick throttle,
or where the scheduler's queueing behavior otherwise changes. Every scheduler
claim in this document is scoped to 0.22.1 and **[unverified]** for any other
version. If this change is implemented, either pin `vllm` exactly or re-verify
the scheduler-source claims below against whatever version is actually
installed at implementation time — don't carry these findings forward
unchecked across a version bump.

## Gate-by-gate inventory

### Gate 1 — Sequence concurrency (`_seq_tracker`)

- **State source**: local, in-process (`_PerModelCounter`, a `dict[str, int]`
  behind a `threading.Lock`). **[verified]** `admission.py`.
- **Sync/async**: fully synchronous, no `await` anywhere in `admit()`/`release()`.
  **[verified]**
- **Invariant protected**: "no more than `max_num_seqs` requests run
  concurrently against one model." **[source-derived]** — this is *also*
  vLLM's own invariant (`assert len(self.running) <= self.max_num_running_reqs`,
  `scheduler.py:830`), enforced by promotion throttling + preemption, not
  rejection. **[verified]**
- **Does vLLM already enforce it?** Yes, but via queueing + preemption, not
  rejection. **[source-derived]**
- **Unsafe-dispatch risk if simplified/removed**: if Gate 1 disappears with no
  companion change, non-streaming requests that would have been queued by
  vLLM are instead dispatched into `ChatService.complete()`, which **is**
  bounded — by `VLLMEngine.generate()`'s existing 300s `asyncio.timeout` — but
  fails on timeout with a generic 500 and no `Retry-After`, a materially worse
  client experience than today's instant, typed 429. See "Central finding"
  below for the corrected picture (an earlier draft of this document wrongly
  claimed no timeout exists at all). This is the actual risk, not GPU
  overcommitment (vLLM's own scheduler still protects that).
- **False-429 risk under legitimate scheduler queueing**: confirmed structurally.
  A request arriving when `in_flight == max_num_seqs` is rejected instantly by
  Gate 1 even in cases where vLLM would have safely queued and served it within
  a short wait (e.g. the currently-running request at the ceiling is one token
  away from finishing). **[source-derived]**
- **Scope**: per-model (keyed by `routed_model`). **[verified]**
- **Concurrent-request behavior**: see "Concurrency / race analysis" below —
  no exploitable TOCTOU under current single-process, `--workers`-less deployment.
- **Cancellation behavior**: covered by existing regression tests, see below.
- **Generation-failure behavior**: same — `release()` runs in a `finally` on
  every path in `ChatService`. **[verified]** via `test_chat_service.py`'s 9
  lifecycle tests.
- **Multi-model behavior**: correctly isolated — counter keyed per model name,
  single `AdmissionController` instance shared across models on `ChatService`.
  **[verified]**
- **`pool_size > 1` behavior**: `pool_size` counts distinct model *names*
  (`api/deps.py:98`, `len(resolved_models)`), not replicas of the same model.
  Gate 1's per-model counter is meaningless for same-model horizontal scaling
  because that configuration doesn't exist in this codebase —
  `EnginePool._engines: dict[str, BaseEngine]` (`engines/pool.py:23`) is a
  strict 1:1 name→engine map. **[verified]** Out of reach for this change; B6
  territory.
- **Original rationale — DEC-040 (2026-07-01)** **[verified,** quoted verbatim
  from `docs/DECISIONS.md` lines 580-629**]**:
  > "AdmissionController's KV-pressure gate had no visibility into
  > `max_num_seqs`: a saturated sequence-concurrency ceiling just queued
  > silently inside vLLM's scheduler instead of surfacing as an explicit
  > signal."

  Gate 1 (`_InFlightSeqTracker`, later renamed `_PerModelCounter`) was added
  specifically to convert vLLM's silent internal queueing into an explicit,
  client-visible 429. It was **not** an accidental duplication by someone
  unaware vLLM queues — the author already knew, and chose explicit-reject
  over silent-queue. DEC-040 also confirms: no clamp path for either priority
  ("a sequence slot can't be partially granted"), and the increment happens
  only at the very end of `admit()` so an earlier-gate rejection never leaks a
  slot. **[verified]**, matches current code.

  Git history confirms Gate 1 was introduced in commit `ed16710f`
  (2026-07-01), as part of the *vram-tiers-and-observability* change — **not**
  part of the original `add-admission-control` change, whose own `design.md`
  covers only the context-length and KV-token gates and never mentions a
  sequence-count gate. **[verified]** via `git log --follow -p -- admission.py`.

### Gate 2 — Context length

Out of scope for this investigation (different HTTP class, not a
scheduling/capacity question). Documented for completeness only: rejects with
`ContextTooLongError` (400) when prompt + requested output exceeds the resolved
context ceiling (`min(ModelEntry.max_model_len, tier cap, request override)`).
vLLM enforces its own hard context ceiling independently, so this gate is
legitimately necessary regardless of what happens to Gate 1. **[source-derived]**

### Gate 3 — KV-pool pressure (`_tracker`)

- **State source**: local, in-process, separate `_PerModelCounter` instance
  from Gate 1's (`_KVReservationTracker` in the original design, tracks raw
  token counts against `kv_capacity_tokens * 0.9`). **[verified]**
- **Invariant protected**: a tier-aware, priority-differentiated policy —
  `interactive` clamps down to fit, `batch` rejects outright rather than
  silently truncating. **[verified]** `admission.py` + original `design.md`'s
  "Key decisions" section.
- **Does vLLM already enforce it?** Partially and differently. vLLM protects
  against KV exhaustion via **preemption of already-running requests**
  (`scheduler.py:929`, `_preempt_request`), not admission-time refusal of new
  ones. **[verified]** for the preemption path. Whether vLLM's block manager
  has any admission-time refusal when `self.waiting` grows unboundedly against
  a fixed KV pool was **not traced** this round — that's a different code path
  than the one read. **[unverified]**
- **Gate 3 — open question, not resolved.** An earlier draft of this
  investigation asserted a false-429 symmetry between Gate 1 and Gate 3 ("Gate
  3's batch-tier 429 is also not something vLLM would raise"). That claim is
  **withdrawn**: it conflated preemption-of-running-requests with
  admission-time-refusal-of-new-requests, which are different code paths, and
  only the former was actually traced. Whether Gate 3 produces the same class
  of false 429s as Gate 1 is genuinely unknown from this investigation and
  should not be assumed either way. **Recommendation: retain Gate 3 as-is.**
  It protects a policy (clamp-vs-reject by priority tier) that has no
  scheduler equivalent regardless of the unresolved question above, since
  InferenceX never engages vLLM's `SchedulingPolicy.PRIORITY` (the `priority`
  kwarg on `AsyncLLM.generate()` is never set from InferenceX —
  `vllm_engine.py` has no `priority` reference; `schemas/chat.py`'s `priority`
  field is InferenceX's own concept, not forwarded to vLLM). **[verified]**
- **Config-duplication side finding**: `utils/vllm_pool_config.py:478-482`
  (`apply_tier_knobs`, sets vLLM's actual `max_num_seqs` at construction) and
  `routing/admission.py:202-213` (`_effective_max_num_seqs`, used at admission
  time) independently compute the identical `min(tier ceiling, model override)`
  formula in two places. **[verified]** Not a correctness bug (both currently
  agree), but worth flagging as a latent-drift risk unrelated to the false-429
  question — not fixed in this change (no `src/` edits in this phase).

## Central finding: non-streaming *is* bounded today, but by a 300s generic 500, not a fast typed 429

**Correction to an earlier draft of this investigation.** An earlier pass
concluded `ChatService.complete()` has no timeout at all. That was wrong — it
only inspected the `ChatService` layer and missed a timeout one layer down.

**[verified]** `VLLMEngine.generate()` (`engines/vllm_engine.py:680-706`, the
method `ChatService.complete()` calls into at `chat_service.py:81`) wraps its
entire body in `async with asyncio.timeout(_COMPLETION_TIMEOUT_S):` where
`_COMPLETION_TIMEOUT_S = 300.0` (line 34). This predates B1 — the inline
comment cites it as "Decision 7 / Task 2.7" of the `migrate-async-llm-engine`
change, preserving "the pre-migration policy where only the non-streaming path
was time-bounded." **[verified]** `generate_stream()` (used by
`stream_response()`) has no equivalent engine-layer timeout of its own —
`ChatService`'s own per-token `asyncio.wait_for(gen.__anext__(), timeout=
stream_timeout_s)` bounds that path instead, at the service layer rather than
the engine layer.

**So both paths are bounded today, by different mechanisms at different
layers.** The risk this investigation needs to weigh is narrower than
"unbounded hang" and is genuinely three separate, smaller things:

1. **[source-derived]** 300s is a much longer wait than clients get from
   Gate 1 today (an instant, sub-millisecond 429). Even though it's
   technically bounded, it's a real UX regression if a client hits it under
   Option A.
2. **[verified]** `RuntimeError` (what a timeout raises) is mapped by
   `api/errors.py:14-21`'s `runtime_error_handler` to **HTTP 500**, with a
   sanitized generic message (`"Request could not be processed."`,
   `type: "internal_error"`) and **no `Retry-After` header** — the client
   loses the "this is transient, retry after N seconds" signal Gate 1's 429
   gives it today, and receives a response shape that reads as a server
   fault rather than a capacity condition.
3. **[source-derived]** Under Option A, a request that ultimately times out
   *did* reach the engine and occupy a real vLLM queue slot for up to 300s
   before failing — a real scheduler-attention/resource cost under sustained
   overload, not merely a client-experience difference, unlike today where a
   rejected request costs vLLM nothing.

**[inference]** This softens, but does not eliminate, the case for a
companion decision before narrowing Gate 1. The companion decision is no
longer "does a timeout need to be added" — one already exists — but "is a
300s wait ending in a generic 500 acceptable, or does Option A also need a
shorter timeout value and/or a typed, `Retry-After`-bearing error on that
timeout path to match what Gate 1 gives clients today." This is still a real
design decision, just a smaller and more precise one than originally framed.

## Concurrency / race analysis

**[verified]** `scripts/dev.sh:41` runs `uvicorn` with no `--workers` flag —
single process. **[verified]** `AdmissionController.admit()`/`release()` have
zero `async def` / `await` anywhere in `admission.py`. **[source-derived]**
Under Python's single-threaded event loop with no `await` inside these
methods, no other coroutine can interleave mid-`admit()` or mid-`release()` —
there is no TOCTOU window between "check `_seq_tracker.current()`" and
"increment `_seq_tracker`" that another request could race into. The
`threading.Lock` inside `_PerModelCounter` is defensive (protects against a
hypothetical multi-worker or multi-threaded deployment) but not load-bearing
under the actual current deployment shape. **[inference]** A multi-worker
deployment would need cross-process state (Redis, a shared counter service)
that nothing in this codebase provides today — out of this change's reach,
and already implicitly covered by DEC-DEFER-02's "single-user local" framing.

Traced state-transition paths (all pre-existing, all covered by
`test_chat_service.py`'s 9 named lifecycle tests at lines 609-729):
- admit → increment → dispatch → generate → stream → completion → `release()`
- admit → increment → client disconnect → generator `.close()` → `finally` →
  `release()`
- admit → increment → engine raises → `finally` → `release()`

**[verified]** No state-transition was found in this investigation where
`_seq_tracker` can drift from actual in-flight count — every termination path
routes through the same `finally`-guarded `release()`. This was **not**
re-derived from scratch; it was confirmed by locating and reading the existing
regression test suite rather than authoring new experiments, consistent with
the instruction not to leave temporary artifacts in the repo.

## Empirical findings

A live experiment was run this session against the actual installed vLLM
0.22.1 / RTX 4060 environment, using `opt-125m` (the same model DEC-040 used
for its own live verification) constructed the same way
`VLLMEngine.__init__` does (`AsyncLLM.from_engine_args(AsyncEngineArgs(...))`,
`engines/vllm_engine.py:465-472`), at the 6gb tier's `max_num_seqs=4`. Full
methodology, raw results, and log are preserved outside the repo per the
"leave no artifacts" constraint; summarized here.

**[verified, live] Phase 1 — burst, 4x over ceiling.** 48 requests (3 rounds
of 16, 4x `max_num_seqs`) submitted simultaneously. **All 48 completed
successfully — zero rejections, zero errors.** submit→first-token: min
0.015s, median 0.436s, p90 0.806s, max 0.971s. submit→completion: min 0.173s,
median 0.597s, p90 0.964s, max 1.127s. This directly confirms, empirically
rather than only from source, that vLLM 0.22.1 queues past `max_num_seqs`
rather than rejecting.

**[verified, live] Phase 2 — sustained arrival, 90 requests over 45s (1 every
0.5s).** Completion latency stayed flat across the run: first-10 and last-10
both cluster around 0.13-0.14s median; overall {min 0.114s, median 0.135s,
p90 0.140s, max 0.493s, the single outlier occurring early, consistent with
warmup}. **No monotonic growth, no starvation signature** — this is the
tail-under-sustained-arrival measurement the earlier draft of this
investigation flagged as missing and deliberately did not fabricate.
**[inference] Caveat**: opt-125m's per-request service time (~0.13-0.28s once
running) comfortably outpaces the 2 req/s arrival rate used here, so this run
never built genuine backlog — it demonstrates no drift/starvation bug, not
"bounded wait under real saturation for a slower model." Extrapolating this
result to a 7B-class model is **[unverified]** and should not be assumed.

**[verified, live, but inconclusive for its intended purpose] Phase 3 (round
2) — cancellation while queued, first attempt.** Intended to cancel a request
still waiting behind the 4-slot ceiling. In practice, opt-125m's throughput
meant both "queued" requests completed (~0.11s) before the scripted 0.5s
delay-then-cancel fired, so `task.cancel()` hit an already-finished task — a
no-op, not a genuine mid-queue cancellation. Not cited as evidence either way.

**[verified, live, but inconclusive for its intended purpose] Phase 4 (round
2) — generation-failure isolation, first attempt.** The "bad" request failed
at `SamplingParams` construction (`max_tokens must be at least 1, got -5`) —
a client-side validation error that never reached `add_request()`. Confirmed
siblings were unaffected and the engine did not enter an errored state, but
this is pre-flight-validation isolation, not an admitted/queued request
failing *inside* the engine mid-generation.

### Round 3 — closing the two remaining gaps (3a, 3b, 3c)

**3a — does the queue-vs-reject property depend on model size, KV capacity,
`max_model_len`, `max_num_seqs`, quantization, memory pressure, preemption,
or pool configuration? [source-derived, closed]** Direct reads of the
installed vLLM 0.22.1 request-intake path, not a live benchmark (a live
7B-class run would answer a *timing* question, not this *structural* one —
see below):
- `Scheduler.add_request()` (`v1/core/sched/scheduler.py:1755-1774`)
  unconditionally calls `self._enqueue_waiting_request(request)` for any new
  request id. The only branches in that method concern *duplicate* request
  ids (a streaming-session update path, irrelevant here) — no branch reads
  model size, KV capacity, `max_model_len`, quantization, or memory pressure.
- `EngineCore.add_request()` (`v1/engine/core.py:337-368`), the caller one
  layer up, validates only `request_id`'s type and (for pooling requests
  only) `pooling_params.task` membership, then unconditionally calls
  `self.scheduler.add_request(request)`. No capacity or model-size check.
- The **only** conditional rejection of an add anywhere in this path is
  `_reject_add_in_shutdown` (`core.py:1329-1335`), gated purely on
  `EngineShutdownState`, independent of the request or model entirely.
- `Scheduler.schedule()`'s promotion throttle (`len(self.running) ==
  self.max_num_running_reqs`, `scheduler.py:549`) and KV-pressure preemption
  (`_preempt_request`, `scheduler.py:929`) both operate on requests already
  admitted into `self.waiting`/`self.running` — they decide *promotion
  timing*, not *admission*, and neither can turn into a rejection of a new
  request.

**Conclusion:** the reject-vs-queue property is structurally independent of
model size, KV capacity, `max_model_len`, `max_num_seqs`, quantization,
memory pressure, and pool configuration — it is a pure, unconditional
enqueue in every version-0.22.1 code path read. This is a claim about
*control flow*, provable from source without running a 7B model. What source
reading *cannot* answer — and what remains genuinely open — is *how long* a
queued request waits under real backlog on a larger/slower model; that is a
timing question, addressed separately under the acquisition-timeout section
below, not by this structural finding.

**3b — genuine cancellation while queued. [verified, live, closed]** Two
further live attempts were required before this was cleanly exercised (both
against the actual installed vLLM 0.22.1 / RTX 4060, `opt-125m`,
`max_num_seqs=4`):
- **Attempt 2** (larger `max_tokens=1800`, still allowing early EOS): failed
  the same way as round 2 — opt-125m's greedy decoding hit EOS and the 4
  blockers finished in ~3.3s, before the queued requests could be proven
  waiting.
- **Attempt 3** (`ignore_eos=True` to force full-length generation, still
  `max_tokens=1800`): blockers ran to their full length but still finished in
  ~3.1s — small-model throughput on this GPU is simply too high for a
  multi-second stall window at any `max_tokens` this model's 2048-token
  context allows.
- **Attempt 4** (same as attempt 3, but the queued requests were cancelled
  0.5s after submission instead of 3.0s, based on attempt 3's measured ~3.1s
  blocker runtime): **succeeded.** At the moment of cancellation: all 4
  blockers were confirmed still running (`blockers_done_count_at_cancel_time:
  0`), and both queued requests had produced **zero output tokens**
  (`ever_started: false` / confirmed via `first_token_t is None` immediately
  before the cancel call). **Caveat on what this actually shows**: the
  evidence is zero observed output tokens plus all four blockers still
  running at cancel time — strong circumstantial evidence of a request
  still parked in the scheduler's waiting queue, but not a direct read of
  `Scheduler.get_request_counts()` (returns `(len(running), len(waiting))`),
  which this experiment did not call. `max_num_seqs` throttles *promotion
  per scheduling tick*, not queue membership, so "zero tokens produced yet"
  is consistent with either genuinely waiting or having just been promoted
  and not yet through a first prefill step — this experiment cannot fully
  distinguish the two on timing alone. `queued_cancel.cancel()`
  delivered a real `asyncio.CancelledError` to a request that had never
  started (`elapsed_before_cancel_s: 0.4998s`, `cancelled: true`,
  `ever_started: false`). The engine's `errored` flag stayed `False`
  immediately after and at the end of the run. The sibling queued request
  (`queued_keep`) was unaffected and completed normally once a slot freed
  (`finished: true`). All 4 blockers completed normally afterward. A fresh
  probe request submitted after everything else completed in 0.045s,
  confirming the engine's internal state was undisturbed by the cancellation.

**Conclusion:** vLLM 0.22.1's engine-side handling of a genuinely-queued
request's cancellation is clean — no engine error state, no sibling impact,
no lingering effect on subsequent requests. This is the vLLM-side half of
the cancellation story. The InferenceX-side half (does the future
`AdmissionController` semaphore permit get released exactly once when this
exact `CancelledError` propagates through `admit()`/`ChatService`) is
established separately, by source-level trace, in "Semaphore lifecycle
analysis" below — this live result confirms the premise that trace depends
on (a genuinely-queued request really can be cancelled cleanly at the engine
level), it does not by itself exercise `AdmissionController` code that
doesn't exist yet.

**3c — genuine engine failure while queued. [source-derived, closed via
source tracing, not a live experiment]** Per the explicit instruction not to
destabilize the environment to manufacture a real engine crash, this gap was
closed by tracing the failure paths rather than inducing one:
- A **catastrophic** failure (CUDA error, executor crash) is not
  request-scoped in vLLM v1: the model executor runs the whole scheduled
  batch together, and `EngineCoreRequestType.EXECUTOR_FAILED` (`core.py`)
  raises `RuntimeError("Executor failed.")` at the engine-process level, not
  per-request. `AsyncLLM.generate()` surfaces this as `EngineDeadError`
  (`async_llm.py`, `except EngineDeadError: ... raise`, deliberately *not*
  calling `abort()` — "Engine is dead. Do not abort since we shut down").
  This affects **every** in-flight and queued request simultaneously, running
  or not — it is not a scenario where "siblings continue" is even a coherent
  question, because there are no unaffected siblings.
- A **request-scoped** failure of an *already-running* request (e.g. a
  grammar/FSM rejection, `scheduler.py:1424`, `"Unexpected: grammar rejected
  tokens..."`) is isolated without engine-wide impact — but this applies to a
  request that has been promoted into `self.running`, i.e. it is already
  executing, not queued.
- A request that is **still queued** (never promoted to `self.running`) has
  not yet been included in any `execute_model()` step — nothing about *its
  own* generation can fail independently, because nothing has started. The
  only two ways a queued request terminates before promotion are external
  cancellation/abort (3b, above) or the whole engine going down
  (`EngineDeadError`, which is not sibling-isolated by construction).

**Conclusion:** "a queued request fails inside the engine while siblings
continue" does not correspond to a distinct vLLM-internal failure mode —
it collapses into either 3b (cancellation) or a whole-engine failure that
is explicitly *not* isolated. For admission-slot release purposes this is
good news, not a gap: `ChatService.complete()`/`stream_response()`'s
existing `try/finally` around engine dispatch is exception-type-agnostic —
it releases on `EngineDeadError`→`RuntimeError` exactly the same way it
releases on any other exception, already covered by the existing
`test_engine_exception_releases_exactly_once` regression test (a mock
exception, but the code path — `finally` running regardless of exception
type — is identical for a real one). Option C's semaphore release sits in
the same `finally` span and inherits this property unchanged.

**Net assessment [inference]:** All three round-3 gaps are now closed — 3a
and 3c by source tracing (structural questions, correctly answered without a
live run), 3b by a live experiment that took four attempts to correctly
force a genuine stall window on a fast small model, but succeeded cleanly on
the fourth. Combined with round 2's Phases 1-2 (queue-not-reject confirmed
live, wait stays flat under sustained arrival), this investigation now has
no remaining *structural* unknowns about whether Option C is safe to design
in detail. What remains open is a *quantitative* question — the
acquisition-timeout value for models/loads this investigation's GPU and
models don't represent — addressed explicitly, not glossed over, in
"Acquisition, rejection, and timeout semantics" below.

All vLLM scheduler claims not covered by a live run above remain
**[source-derived]** from direct reads of the installed 0.22.1 source, with
the version-scope caveat noted at the top of this document.

## A third option: bounded per-model semaphore (Option C)

Investigated per the change owner's explicit instruction not to constrain the
comparison to A vs. B. **[inference, this investigation's recommendation]** A
per-model `asyncio.Semaphore(max_num_seqs)`, acquired with a deadline
(`asyncio.wait_for(sem.acquire(), timeout=admission_wait_s)`) inside
`AdmissionController.admit()`, replaces `_seq_tracker`'s counter-plus-instant-
reject with counter-plus-bounded-wait:

- Preserves "the engine is never invoked for a rejected request" — the
  semaphore gates entry the same way `_seq_tracker` does today, just with a
  wait instead of an instant check.
- Preserves DEC-040's explicit-signal goal — a timeout on `sem.acquire()`
  still raises `EngineSaturatedError` (429 + `Retry-After`), not a generic
  500; the client keeps the same actionable response shape it gets today.
- Cancellation-aware for free: `asyncio.wait_for` cancels the pending
  `acquire()` cleanly on client disconnect, same mechanism already proven
  safe by the existing 9 `test_chat_service.py` lifecycle tests' disconnect
  paths.
- Release: `sem.release()` in the same `finally` block that already calls
  `_seq_tracker.add(routed_model, -1)` today — no new lifecycle surface.
- Does **not** require trusting vLLM's queue-depth behavior for models this
  investigation didn't test (Phase 1-2's live results were opt-125m-specific)
  — the bound is enforced by InferenceX itself, so it degrades no worse than
  today's behavior regardless of model size, only the wait window changes.

This is materially different from "Option B" as originally scoped in
`proposal.md` — the same underlying idea (bounded wait at the admission
layer), but concretely simpler: no polling, no re-reading engine state, no
second scheduler loop, just a semaphore, which the earlier B-vs-A framing's
"reimplementing the scheduler" objection does not actually apply to.

## What quantity does the semaphore bound?

**[verified/inference — this must be stated precisely, not left implicit.]**
The semaphore bounds **the number of requests InferenceX has admitted past
`AdmissionController.admit()` for a given model name and not yet released** —
exactly the same quantity `_seq_tracker` counts today. It does **not**:

- read or synchronize with vLLM's live `len(self.running)` /
  `len(self.waiting)` at any point;
- know whether an admitted request has been *promoted* into vLLM's running
  set, only that it was *admitted* by InferenceX;
- adapt its own ceiling to real-time scheduler state.

It is a **static, InferenceX-side mirror** of the same tier-resolved
`max_num_seqs` value `_seq_tracker` already enforces, sized once at first use
per model name and never resized at runtime (consistent with §"Semaphore
ownership and keying" below — this codebase has no dynamic
load/unload/reconfiguration to resize against). Calling this design
"scheduler-aware" would be wrong: it is exactly as scheduler-*unaware* as
today's counter, and it stays correct as an InferenceX-side capacity signal
for the same reason `_seq_tracker` is correct today — because InferenceX's
own admission decisions are the only source of truth for the sequence-slot
budget it hands out, independent of what vLLM's internal scheduler happens
to be doing with them. The only behavior change is *what happens when the
bound is hit*: today, instant reject; under Option C, a bounded wait, because
round 2-3's live evidence established that a slot at the ceiling usually
frees within roughly a second on this hardware, and instant rejection in
that case is a false 429 the current code cannot avoid.

## Semaphore lifecycle analysis — the central finding of this round

**Target invariant:** every successfully acquired admission permit is
released exactly once, regardless of how request processing terminates, and
this must hold without depending on garbage collection.

Eight distinct termination paths were traced. Seven of them are **already
safe for free** — inherited either from `ChatService`'s existing
`try`/`finally` structure (already covered by the 9 named lifecycle tests in
`test_chat_service.py`) or from CPython's own `asyncio.Semaphore`
implementation. **One is not safe under the sketch originally written into
`tasks.md` item 2.2c, and requires new code inside `AdmissionController.admit()`
itself.** This is the single most important finding of this investigation
round.

| # | Path | Release mechanism | Exactly-once? |
|---|---|---|---|
| 1 | Normal completion (non-streaming) | `ChatService.complete()`'s existing `try/finally` around `await engine.generate(...)` | **[verified]** Yes — pre-existing, unchanged by Option C |
| 2 | Normal completion (streaming) | `stream_response()`'s single `try/finally` spanning admit()-return to generator termination | **[verified]** Yes — pre-existing |
| 3 | Client disconnect | Generator `.close()` → `GeneratorExit` at the suspended `yield` → same `finally` | **[verified]** Yes — `test_client_disconnect_after_terminal_event_releases_exactly_once`, `test_generator_exit_before_first_token_releases_exactly_once` |
| 4 | `asyncio` cancellation | `CancelledError` propagates through the same `finally` | **[verified]** Yes — `test_cancelled_error_releases_exactly_once` |
| 5 | Engine timeout (300s `_COMPLETION_TIMEOUT_S`, non-streaming only) | `asyncio.timeout()` cancels internally, propagates through the same `finally` | **[verified]** Yes — `test_timeout_releases_exactly_once` |
| 6 | Engine exception (incl. a real `EngineDeadError`, per 3c above) | Propagates through the same `finally`, exception-type-agnostic | **[verified]** Yes — `test_engine_exception_releases_exactly_once` |
| 7 | **Admission rejection *after* the semaphore permit was already acquired** (Gate 2 or Gate 3 raises after Gate 1's `sem.acquire()` succeeded) | **None today.** `ChatService`'s `try/finally` only starts once `admit()` *returns*; if `admit()` itself raises, that `finally` never begins | **NOT safe under the `tasks.md` 2.2c sketch — see below.** |
| 8 | Task cancellation *while blocked inside* `sem.acquire()` (including via `asyncio.wait_for`'s timeout) | CPython's own `Semaphore.acquire()` | **[verified, from source]** Yes — see below |

**Path 8 in detail — no new code needed.** Reading the installed Python
3.13 `asyncio/locks.py` `Semaphore.acquire()` directly: a waiting `acquire()`
call sits in `try: await fut finally: self._waiters.remove(fut)`. On
`CancelledError`, if the future had *not yet* been woken with a permit, no
bookkeeping needs undoing (it was never granted one). If it *had* just been
woken (the narrow race where `_wake_up_next()` fires right as the caller is
cancelled), the `except CancelledError` branch explicitly does
`self._value += 1` before re-raising — handing the permit back. `locked()`
also consults `self._waiters`, not just `self._value` (see "Fairness"
below), so this bookkeeping is self-consistent. `asyncio.wait_for(sem.acquire(),
timeout=...)`'s timeout path cancels the inner coroutine at exactly this
await point, so a timeout hits the identical, already-safe branch — `wait_for`
then converts the resulting `CancelledError` into `TimeoutError`, which
`admit()` maps to `EngineSaturatedError`. No permit is ever leaked on this
path, and no InferenceX code has to implement it — it is a property of the
primitive itself, confirmed by source, not assumed.

**Path 7 in detail — the hazard, and its fix.** Today's `admit()` is safe
by construction because *every* gate's commit (`self._tracker.add(...)`,
`self._seq_tracker.add(routed_model, 1)`) happens on the **last two lines**
of the method, after all three gates have already passed — an earlier-gate
rejection literally cannot leak a later-gate's reservation, because nothing
was reserved yet (this exact property is what
`test_a_request_rejected_by_context_gate_does_not_consume_a_sequence_slot`
already asserts for today's code). A semaphore breaks this pattern: checking
"is a permit available" and "consuming one" are the same atomic operation —
there is no cheap non-consuming pre-check equivalent to `_seq_tracker.current()
< ceiling`. Once `sem.acquire()` succeeds, a permit **is** held immediately,
before Gates 2/3 have run.

Two orderings were considered:

- **Semaphore acquired last** (after Gates 2/3 pass, mirroring today's
  commit-last pattern): avoids re-litigating error precedence, but does not
  avoid the hazard — it moves it. Gate 3 already commits
  `self._tracker.add(routed_model, reserved_tokens)` before Gate 1 would now
  run; if the now-last semaphore wait times out, that KV-budget reservation
  must be rolled back (`self._tracker.add(routed_model, -reserved_tokens)`)
  before re-raising, or KV budget silently leaks for every timed-out
  admission. Still needs new exception handling, and additionally changes
  which HTTP error a doubly-saturated request sees (context/KV error instead
  of today's sequence-ceiling 429) and forces every request to pay the
  prompt-token/KV computation cost even when it's ultimately going to wait or
  time out on Gate 1.
- **Semaphore acquired first** (recommended — preserves today's exact gate
  order, error precedence, and fail-cheap-before-computing-prompt-tokens
  behavior): requires wrapping the remainder of `admit()` — Gates 2/3 and the
  final commit — in a single `try/except` that releases the just-acquired
  permit before re-raising on any exception:

  ```python
  async def admit(self, routed_model, request, engine) -> AdmissionResult:
      warnings: list[ResponseWarning] = []
      effective_max_num_seqs = self._effective_max_num_seqs(routed_model)
      sem: asyncio.Semaphore | None = None
      if effective_max_num_seqs is None:
          _warn(warnings, type="degraded", code="sequence_gate_skipped", ...)
      else:
          sem = self._get_semaphore(routed_model, effective_max_num_seqs)  # BoundedSemaphore
          try:
              await asyncio.wait_for(sem.acquire(), timeout=self._admission_wait_s)
          except TimeoutError:
              raise EngineSaturatedError(
                  f"Model '{routed_model}' is at its concurrent-sequence limit; "
                  f"no slot freed within {self._admission_wait_s}s; retry shortly.",
                  retry_after_s=1.0,
              )
      try:
          # Gate 2 (context) — unchanged, may raise ContextTooLongError /
          # StrictModeViolationError
          # Gate 3 (KV) — unchanged, may raise EngineSaturatedError /
          # StrictModeViolationError
          reserved_tokens = prompt_tokens + effective_output
          self._tracker.add(routed_model, reserved_tokens)
          return AdmissionResult(effective_max_tokens=effective_output,
                                  reserved_tokens=reserved_tokens,
                                  warnings=tuple(warnings))
      except BaseException:
          if sem is not None:
              sem.release()
          raise
  ```

  Gates 2/3's own logic contains no `await` (**[verified]**, unchanged from
  today), so this second `try` block never itself suspends — the only
  suspension point in the whole method is the `sem.acquire()` above it,
  which is already covered by path 8. The `except BaseException` here only
  ever has to handle the well-typed, synchronous exceptions Gates 2/3 already
  raise today (`ContextTooLongError`, `StrictModeViolationError`,
  `EngineSaturatedError`) — not arbitrary cancellation.

  **Standing invariant this fix depends on:** the blanket `except
  BaseException: sem.release(); raise` is correct *only* because Gates 2/3
  contain no `await` today — there is no suspension point inside the second
  `try` block where a cancellation could land ambiguously between "permit
  should be released" and "permit already released by a concurrent path."
  If a future change adds an `await` inside Gates 2/3 (e.g. `Gate 2`'s
  `engine.count_prompt_tokens()` is a synchronous engine-interface call
  today but is exactly the kind of call that could become async later),
  this blanket release becomes unsafe and must be replaced with an
  explicit `acquired = True` / `acquired = False` local flag guarding the
  release, so a second exception on an already-released permit cannot
  double-release. **Recommendation for 2.1/2.2/2.3:** use
  `asyncio.BoundedSemaphore` instead of plain `asyncio.Semaphore` as the
  ceiling primitive — it is otherwise identical (same `acquire()`/
  `release()`/FIFO-waiter behavior, since it subclasses `Semaphore` and
  only overrides `release()`) but raises `ValueError` on any over-release,
  turning a silent ceiling-inflation bug (permits exceeding the configured
  max) into an immediate, loud failure instead of a slow capacity leak.
  This costs nothing today and is strictly safer if the invariant above is
  ever violated by a future edit.

**This is a real, necessary correction to `tasks.md` item 2.2c's original
sketch** ("release in the same `finally` block as today's `_seq_tracker.add(...,
-1)`" — i.e. release only in `ChatService`'s `finally`). That covers paths
1-6 but not path 7, because path 7's failure happens *inside* `admit()`,
before `ChatService`'s `finally` ever starts. `tasks.md` is updated to
reflect the corrected design.

**Interface consequence:** `AdmissionController.admit()` must become
`async def` (it now contains one genuine `await`). `release()` stays
synchronous — `Semaphore.release()` itself is not a coroutine, so releasing
a permit needs no `await`; `release()` only gains one extra line
(`sem.release()` alongside the existing `_tracker.add(routed_model,
-reserved_tokens)`). `ChatService.complete()`/`stream_response()` each need
exactly one call-site change: `admitted = self._admission.admit(...)` →
`admitted = await self._admission.admit(...)`. This is the first `await` in
`AdmissionController`'s public surface — the "Concurrency / race analysis"
section above, which found no TOCTOU window specifically *because* nothing
in `admit()`/`release()` awaited, no longer describes the code once Option C
ships; see "Compatibility invariants" below for the follow-on note.

## Semaphore ownership and keying

**[verified/inference]** No new keying question is introduced. The semaphore
dict (`dict[str, asyncio.Semaphore]`) is keyed by `routed_model` — the exact
same string key space `_seq_tracker` and `_tracker` already use, created
lazily (`dict.setdefault`-style) on first `admit()` call for a given model
name, sized once from `_effective_max_num_seqs(routed_model)` at that first
call. This is safe because:

- **Model identity is resolved before admission.** `routed_model` is the
  post-routing name `ChatService._resolve_engine()` produces; `admit()`
  never sees an alias.
- **No dynamic load/unload exists today.** `EnginePool.__init__` takes a
  fixed `dict[str, BaseEngine]` at construction and has no `add`/`remove`
  method (**[verified]**, `engines/pool.py`) — the set of model names, and
  each one's resolved `max_num_seqs`, is fixed for the process lifetime.
  There is no runtime event that would need to resize or replace an
  already-created semaphore, so no invalidation logic is needed.
- **Single `AdmissionController` instance, single event loop.** No
  cross-process or cross-worker semaphore-sharing question exists —
  `scripts/dev.sh` runs one process with no `--workers`, matching the
  existing "Concurrency / race analysis" finding.
- **`pool_size > 1` limitation is unchanged, not solved.** `pool_size`
  counts distinct *model names* (`api/deps.py:98`), never same-model
  replicas — `EnginePool._engines` is a strict 1:1 name→engine map
  (**[verified]**). A semaphore keyed by `routed_model` is exactly as
  meaningless for same-model horizontal replication as today's counter is.
  This remains explicitly B6 territory, not solved or extended by Option C.

## Primitive comparison — is `asyncio.Semaphore` the right tool?

| Primitive | Advantage | Problem |
|---|---|---|
| `asyncio.BoundedSemaphore` (chosen — subclass of `asyncio.Semaphore`) | Stdlib, ~10 lines of integration, exactly-once release provable by source (paths 7-8 above), FIFO in this Python's implementation (see "Fairness"), identical `acquire()`/waiter behavior to plain `Semaphore` since it only overrides `release()`, and raises `ValueError` on over-release instead of silently inflating the ceiling if the "no `await` in Gates 2/3" invariant is ever broken by a future edit (see "Semaphore lifecycle analysis") | Permits are bounded, **waiters are not** — see "B4/B5 boundary" below |
| Bounded `asyncio.Queue` (put a token in, take one out) | Same bounded-permit idea, marginally more explicit "waiting" semantics | No behavioral advantage over `Semaphore` for this use case; strictly more code to get the same guarantee, and still has the identical unbounded-waiters property (a `Queue.get()` on an empty queue blocks the same way `Semaphore.acquire()` does) |
| Explicit waiter list + manual wake-up | Full control over ordering/starvation policy | Reimplements what `Semaphore` already does correctly (see "Fairness") — exactly the "second scheduler" risk the decision matrix already penalizes Option B for |
| Retain `_PerModelCounter` + `asyncio.Condition` | Closer to today's existing counter shape | A `Condition` requires the same acquire/release discipline as a `Semaphore` but without the built-in permit bookkeeping — reimplements `Semaphore` with strictly more surface area for the same result |
| Rely entirely on vLLM's queue (no InferenceX-side bound at all) | Zero new state | This *is* Option A, already compared in the decision matrix — changes the "engine never invoked for a rejected request" invariant, which Option C exists specifically to avoid changing |

**Conclusion:** `asyncio.Semaphore` gives the smallest correct primitive for
this specific bound. None of the alternatives buy additional safety or
correctness for the cost of more code — the standard "reimplementing a
second scheduler" risk this document already used to reject Option B applies
equally to hand-rolled alternatives to `Semaphore`.

## Fairness — is `asyncio.Semaphore` actually FIFO?

**[verified, from source]** This must not be assumed without checking —
`Semaphore` does not document a fairness guarantee in the abstract. Reading
the installed Python 3.13 `asyncio/locks.py` directly: `locked()` returns
`True` whenever `self._value == 0` **or** any non-cancelled waiter already
exists in `self._waiters` (a `collections.deque`). This means a *new* caller
to `acquire()` cannot synchronously steal a just-freed permit out from under
an earlier caller that is still queued — even if `self._value > 0` at that
instant, `locked()` forces the newcomer to also enqueue behind the existing
waiters. Waiters are woken strictly from the front of the deque
(`_wake_up_next`'s `for fut in self._waiters: if not fut.done(): ...; break`).
**This is a genuine, source-verified FIFO guarantee in this Python's
implementation** — not something this design has to hedge on, and not
something Option C needs an alternative primitive to obtain.

## Acquisition, rejection, and timeout (429) contract

**[inference — mechanism justified by evidence, exact value explicitly not
invented without evidence, per this investigation's standing instruction.]**
"Bounded semaphore" is ambiguous by itself — bounding *permits* (the
semaphore's job) is orthogonal to bounding *how long a caller waits for one*
(a separate policy decision). The chosen policy:

- **Wait up to a deadline, then 429** — not indefinite wait (would
  reintroduce an unbounded hang, the exact failure mode DEC-040 was written
  to avoid) and not immediate-reject-with-no-wait (would just be today's
  behavior with extra code). `asyncio.wait_for(sem.acquire(), timeout=
  admission_wait_s)`.
- **The deadline starts when `admit()` begins waiting on `sem.acquire()`**,
  i.e. effectively when the request arrives at admission — not after some
  other clock. This is the simplest interpretation and matches how
  `stream_timeout_s` and `_COMPLETION_TIMEOUT_S` are each scoped to their own
  single wait, not stacked with anything upstream.
- **Streaming and non-streaming share the identical admission deadline**,
  because both `complete()` and `stream_response()` call the same
  `admit()` before either path diverges — there is exactly one admission
  wait mechanism, not two. This deadline is separate from and additive to
  whichever of `_COMPLETION_TIMEOUT_S` (300s, non-streaming) or
  `stream_timeout_s` (per-token, streaming) applies afterward — worst-case
  latency for a request that waits the full admission deadline and then
  runs to its own timeout is the sum of the two, which is a real
  consideration for picking the admission deadline's magnitude, not just its
  mechanism.
- **On timeout: `EngineSaturatedError` → HTTP 429 + `Retry-After`** — the
  same response shape Gate 1 already gives clients today, preserving
  DEC-040's explicit-signal goal. No new error type, no new HTTP mapping.
- **Config surface**: a new `admission_wait_s` setting, following the exact
  pattern `core/settings.py`'s existing `stream_timeout_s` already
  establishes (`INFERENCE_X_STREAM_TIMEOUT_S` env var, `float`, a documented
  default) — **[verified]** that pattern exists at `core/settings.py:61`.
  `admission_wait_s` should be tier-configurable, not a single global
  constant, because the one open quantitative question this investigation
  could not close (3a closed the *structural* queue-vs-reject question; it
  did not close the *timing* question) is how wait time scales for
  larger/slower models under genuine backlog. Round 2-3's live numbers
  (worst case 1.13s for a 4x-over-ceiling burst on `opt-125m`; the round-3
  cancellation experiment's blockers ran 3-15s depending on `max_tokens`)
  only characterize a small, fast model. **[unverified]** for a 7B-class
  model — this investigation deliberately does not invent a production
  default for that case. A conservative provisional default (low
  single-digit seconds, e.g. derived with headroom from the `opt-125m`
  worst-case) is reasonable to ship *as a config default*, explicitly
  documented as provisional and overridable, with a live smoke test against
  a larger model recommended as a **follow-up validation step** (`tasks.md`
  §4.3), not a blocker to approving the mechanism itself.

## B4/B5 boundary, made explicit

**[inference]** A bare `asyncio.Semaphore` bounds *permits* but not
*waiters* — nothing caps how many requests can simultaneously be queued
behind `sem.acquire()`. In steady state this is bounded indirectly by
`admission_wait_s` (a waiter that times out removes itself), but a sudden
burst can transiently accumulate many more waiters than `max_num_seqs`
before the timeout window closes. This is a real, documented limitation, not
solved by this change: **B4 (Option C) explicitly does not provide** —
- priority ordering between `interactive` and `batch` (both wait/reject
  identically, exactly as Gate 1 does today — no favoring either priority in
  the FIFO wait);
- starvation prevention across priorities;
- a capped/bounded queue depth (the unbounded-waiters property above);
- persistent or durable queueing (in-memory only, lost on process restart,
  same as every other piece of state in this codebase);
- any general-purpose backpressure infrastructure beyond this one
  per-model, per-Gate-1 bound.

If any of these turn out to be required for Option C's correctness at
production scale, that is **B5's** territory (already the roadmap's home
for priority/queueing policy), not something to silently absorb into B4.
Nothing found in this investigation requires them for Option C's own
correctness — the semaphore is provably exactly-once-safe (paths 1-8 above)
without any of them.

## Decision matrix — Option A vs. B vs. C

| Dimension | A (narrow Gate 1, dispatch to vLLM) | B (poll/re-check inside admission) | C (bounded semaphore) |
|---|---|---|---|
| False-429 reduction | Yes — vLLM's real queue serves the request | Yes, if implemented correctly | Yes |
| Queue ownership | vLLM (real scheduler state) | InferenceX (re-derived, second scheduler) | InferenceX (semaphore, not a scheduler) |
| Timeout mechanism | Existing 300s `VLLMEngine` timeout (needs a value/error-mapping decision) — **[verified]** exists today | New polling loop with its own timeout — not yet designed | `asyncio.wait_for` on `sem.acquire()` — small, well-understood primitive |
| Failure-mode on bound exceeded | Generic 500, no `Retry-After` — **[verified]**, unless changed | Whatever is implemented — undesigned | 429 + `Retry-After` — same shape as today, **[source-derived]** from the pattern |
| "Engine never invoked for rejected request" invariant | **Broken** — request reaches `AsyncLLM.generate()` before any failure | Preserved | Preserved |
| Cancellation semantics | `CancelledError` → `AsyncLLM`'s own abort path — **[verified]** already proven safe for `generate_stream`/`generate` | Needs custom handling — undesigned | `asyncio.wait_for` cancellation — reuses existing, tested disconnect handling |
| Streaming-path interaction | None — Gate 1 change only affects non-streaming/`complete()`'s dispatch decision (streaming already reaches the engine unconditionally today, gated only by Gate 1's counter check, same as C) | Same | Same |
| Implementation complexity | Low (timeout tuning + error mapping) but changes a stated invariant | Higher — a second, hand-rolled queue-aware loop | Low — one primitive, reuses existing release path |
| Model-size generalization risk | **[unverified]** — only tested on opt-125m; relies on vLLM's real queue depth scaling safely for larger/slower models | Same reliance, plus re-derivation risk (stale local state vs. live engine state) | Lower — the bound is InferenceX's own choice, not dependent on assuming vLLM's queue behaves a certain way at any model size |
| Interaction with B2 (native metrics) | None directly; `/metrics`'s queue-depth signals become more meaningful once requests actually reach vLLM's queue | None | None — this change's ADDED requirement keeps admission independent of `/metrics` regardless of option |
| Interaction with B3 (per-request timing) | `queue_time_ms` becomes attributable to real scheduler queueing, more informative | Unaffected | Unaffected |
| Interaction with B5 (documented as queueing-instead-of-429 for priority/batch) | Overlaps in spirit; A effectively does what B5 was scoped for, at the sequence-concurrency layer specifically — risk of scope creep if not kept narrow | Some overlap, less direct | Some overlap (a bounded wait is queueing-adjacent) — must stay scoped to Gate 1 only, not become a general request queue (explicitly out of scope, see proposal.md) |
| Interaction with B6 (same-model replicas) | Unaffected — all three options are per-model-name, single-engine | Unaffected | Unaffected |

Cells left undesigned for B reflect that this investigation did not design B
in the same depth as A/C, because C dominates B on every dimension checked
(same invariant-preservation, simpler mechanism, no re-derivation risk) —
B is documented for completeness, not because it remains equally viable.

## Decision: Option C approved

**[inference — the change owner's explicit direction, recorded here per this
round's instruction; not this investigation inventing authority for itself.]**
The change owner named Option C the working direction after round 2's
report and asked this round to either fully specify or reject it, not to
default to it. Having done so:

- Option C fixes the concrete false-429 problem this investigation found.
- It preserves every invariant DEC-040 and the current spec establish
  ("engine never invoked for a rejected request", 429 + `Retry-After` on
  genuine saturation, no priority-differentiated clamp for Gate 1).
- Its exactly-once release property is now **proven for all 8 termination
  paths** (see "Semaphore lifecycle analysis"), not merely assumed —
  including the one path (#7) that the original `tasks.md` sketch would have
  gotten wrong, now corrected.
- Its three prior evidence gaps (3a/3b/3c) are closed — two by source
  tracing, one by a live experiment that took four attempts but ultimately
  succeeded cleanly.
- Unlike Option A, it does not require trusting vLLM's queue-depth *timing*
  behavior at model sizes this investigation didn't test — the bound is
  InferenceX's own choice, degrading no worse than today regardless of model
  size (only the wait window's *appropriateness* varies by model, a
  config-surface question, not a correctness one).

**What is and is not resolved.** The *mechanism* is fully specified: async
`admit()`, semaphore acquired first (preserving today's gate order and error
precedence), `try/except`-release wrapping Gates 2/3, `release()` staying
synchronous, keying identical to today's counters, FIFO confirmed from
source. The *exact `admission_wait_s` default* for production model sizes
this investigation didn't test remains explicitly **[unverified]** and is
not invented here — a provisional, config-overridable default is
recommended, with a larger-model live smoke test as a follow-up validation
task, not a blocker. This is why `tasks.md` records the design as decided
and specified, while still not marking any *implementation* task complete —
see "Compatibility invariants" and `tasks.md` for the precise line.

Option A is rejected: it breaks the "engine never invoked for a rejected
request" invariant and its failure mode on the timeout path (generic 500, no
`Retry-After`) is worse than Option C's by construction, with no offsetting
advantage once Option C's own live evidence (3a/3b/3c) closes the gaps that
used to make Option C's safety less certain than Option A's. Option B is
rejected as previously documented — Option C achieves the same bounded-wait
goal with a smaller, better-understood primitive and no re-derived scheduler
state.

## Exact proposed responsibility boundary

Unchanged from the codebase's existing framing
(`docs/PHASE-A-ARCHITECTURE.md` §9): `routing/` (admission) decides *whether*
a request may run; the engine decides *when/how* an admitted request
executes. This investigation found no evidence that boundary should be
erased. Option C keeps it exactly as-is (only the instant-reject mechanism
changes shape). Option A would relocate part of the *when* decision from
InferenceX to vLLM's scheduler for the specific case of sequence-concurrency
saturation — a real, evidenced-but-not-yet-chosen boundary shift, not an
accidental one. `AdmissionController` remains the single synchronous decision
point under either A or C; nothing here proposes routing admission logic into
`services/` or the engine layer.

## Rejected alternatives

- **Delete Gate 1 outright, rely entirely on vLLM's scheduler, no companion
  change.** Rejected: reintroduces the exact silent-queueing problem DEC-040
  was written to solve, and — even with the corrected understanding that a
  300s timeout already exists — trades an instant, typed 429 for a
  potentially-300s wait ending in a generic, unhelpful 500. Distinct from
  Option A, which explicitly requires deciding whether that timeout/error
  mapping needs to change before shipping.
- **Use `/metrics`'s `num_requests_waiting` as a live admission signal instead
  of `_seq_tracker`.** Rejected: would make a control-plane decision (admit or
  reject) depend on an observability surface explicitly designed and
  documented (B2) as scrape-only, HTTP-boundary-decoupled. Also introduces a
  cross-cutting dependency from `routing/` on the Prometheus registry that
  `docs/PHASE-A-ARCHITECTURE.md` §9's Engine Boundary table doesn't grant it.
  See this change's ADDED spec requirement guarding against this directly.
- **Adopt vLLM's `priority` kwarg / `SchedulingPolicy.PRIORITY` to let vLLM
  itself differentiate interactive vs. batch.** Not rejected outright, but
  deliberately not investigated in depth — it's a materially larger change
  (changes what "priority" means end-to-end, touches Gate 3's clamp-vs-reject
  policy, needs its own evidence base) and wasn't asked for. Flagged as a
  possible future angle, not folded into this change.

## Compatibility invariants

- No `src/`, `tests/`, or `pyproject.toml` changes in this phase — verified by
  this session touching none of those paths (a temporary experiment harness
  ran outside the repo, in the job's own scratch directory, and is not part
  of this change).
- Under Option C (or B), `429` + `Retry-After` remains the response shape for
  genuine, sustained saturation, unchanged from today. Under Option A, the
  failure mode on saturation changes to a 500 with no `Retry-After` unless the
  timeout/error-mapping is also revised — a compatibility-relevant behavior
  change that must be called out explicitly if A is chosen.
- `ContextTooLongError` → 400 (Gate 2) is entirely unaffected by any option.
- `GET /metrics` and `GET /v1/metrics` are unaffected; see the ADDED
  requirement in `specs/platform/spec.md`.
- **Test-coverage gap found, not fixed in this phase**: no existing test in
  `tests/unit/` exercises `VLLMEngine.generate()`'s `_COMPLETION_TIMEOUT_S`
  path at all (`grep` for `_COMPLETION_TIMEOUT_S`/`"completion timed out"`
  across `tests/` returns nothing). This gap should be closed as part of
  implementation regardless of option.
- **`AdmissionController.admit()` becomes `async def` under Option C** — a
  real interface change, though internal (not an HTTP-facing contract
  change). This is the first `await` `AdmissionController` will have; the
  "Concurrency / race analysis" section's no-TOCTOU finding was specific to
  the current fully-synchronous code and will need re-verification at
  implementation time against the actual committed diff (the "Semaphore
  lifecycle analysis" section's path 7/8 trace already does this for the
  *design*, but a fresh check against the literal diff is still
  implementation-phase due diligence, not something this investigation phase
  can finish on unwritten code).

## Validation strategy (for whichever option is chosen — not run in this phase)

Capacity-at-limit, above-limit, vLLM-queued-and-served-within-bound,
legitimate-policy-rejection (bound exceeded), cancellation while admitted,
generation failure after admission, concurrent admissions, counter cleanup
after every termination path, multi-model isolation, `pool_size > 1`
(document as unsupported rather than test), no regression to existing 429
semantics for genuinely-exhausted capacity, `/v1/metrics` unchanged. Do not add
tests merely to inflate coverage — several of these are already covered by the
existing 6 `TestSequenceConcurrencyGate` tests and the 9 `test_chat_service.py`
lifecycle tests; only the *new* bounded-wait behavior needs new tests, under
whichever option is picked.

## The MODIFIED spec delta, now added

Round 2 of this investigation deliberately withheld a `## MODIFIED
Requirements` section, because the option choice was unconfirmed and Option
C's own design had an unresolved lifecycle hazard (path 7, above) and an
unresolved wait-bound question. Both gates are now cleared differently: the
option is confirmed (Option C, see "Decision" above), the lifecycle hazard
is resolved (a specific, provable design, not a placeholder), and the
wait-bound question is resolved *as a mechanism* (bounded wait, deadline
semantics, config surface) even though its exact numeric default for
untested model sizes stays open and provisional.

`specs/platform/spec.md` in this change now carries a `## MODIFIED
Requirements` section pasting the full current "Incremental architecture"
requirement (all 4 scenarios, per this repo's archiver convention) with only
the "Sequence-concurrency ceiling" scenario's text changed to describe
bounded-wait-then-429 semantics instead of instant rejection — worded around
the *mechanism* (a bounded wait exists, then a 429), not around a specific
numeric deadline, since that number is not yet decided. The ADDED
requirement guarding admission independence from `/metrics` is unchanged
from round 2 and remains true regardless of the wait-bound's exact value.

## Gate 3 — confirmed unchanged, not re-opened

Per the change owner's explicit instruction not to fold Gate 3 into the Gate-1
decision: this investigation's position on Gate 3 (`_tracker`, KV-pool
pressure) is unchanged from the finding earlier in this document — retain
as-is. `_tracker` represents in-flight token reservations against
`kv_capacity_tokens * 0.9`; vLLM cannot enforce its tier-aware,
priority-differentiated clamp-vs-reject policy because InferenceX never
engages `SchedulingPolicy.PRIORITY` (`priority` is never forwarded to vLLM —
**[verified]**, `vllm_engine.py` has no `priority` reference). Whether Gate 3
independently produces false 429s in the Gate-1 sense remains untraced
(vLLM's block-manager admission-time behavior under `waiting`-queue growth
was not read this round) — B2's `/metrics` queue-depth signals do not resolve
this question either, since they report scheduler state, not KV-block
admission logic specifically. This is a distinct, separate investigation, not
something this change should absorb. **Gate 3 stays out of scope for B4.**

**No conflation with Option C's semaphore.** Gate 3's `_tracker` (a
`_PerModelCounter` counting reserved KV tokens) and Gate 1's replacement (a
`dict[str, asyncio.Semaphore]` counting sequence permits) are separate
attributes on `AdmissionController`, touching different units (tokens vs.
permits) and different failure semantics (clamp-or-reject vs. wait-or-reject).
Option C's design changes only the attribute currently named `_seq_tracker`;
`_tracker` and every KV-pressure code path in "Gate-by-gate inventory" above
is untouched by anything in the "Semaphore lifecycle analysis" section.
