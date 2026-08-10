# Design: add-batch-priority-queueing (B5)

## Current admission behavior (as of this investigation)

**[verified]**, `routing/admission.py` (post-B4, `develop` @ `1c67f51`).
`AdmissionController.admit()` runs three gates before every dispatch:

1. **Gate 1 — sequence concurrency** (`admission.py:274-295`). A per-model
   `asyncio.BoundedSemaphore`, sized once from `_effective_max_num_seqs()`
   (tier `max_num_seqs` ∩ per-model override). Acquired via
   `asyncio.wait_for(sem.acquire(), timeout=self._admission_wait_s)`.
   `self._admission_wait_s` is a **single process-wide value**
   (`core/settings.py:73-75`, `INFERENCE_X_ADMISSION_WAIT_S`, default `5.0`)
   — there is no branch on `request.priority` anywhere in Gate 1 today. A
   timeout raises `EngineSaturatedError` (429 + `Retry-After: 1`) for
   **either** priority identically.
2. **Gate 2 — context length** (`admission.py:302-344`). Unaffected by this
   change. `interactive` clamps `max_tokens` to fit; `batch` rejects
   (`ContextTooLongError`, 400) instead of clamping.
3. **Gate 3 — KV-pool pressure** (`admission.py:346-385`). Unaffected by
   this change. Same clamp-vs-reject split as Gate 2, keyed on
   `request.priority == "batch"`.

Gate 1 is wrapped in `try/except BaseException: sem.release(); raise`
(`admission.py:297-397`) so a Gate 2/3 rejection after a Gate 1 acquisition
releases the permit — this is safe today specifically because Gates 2/3
contain no `await` (documented constraint, `admission.py:297-301`).

`_get_semaphore()` (`admission.py:223-237`) is lazy, keyed by `routed_model`,
sized once at first use — no dynamic resize, no cross-process sharing
(single-process, no `--workers`, confirmed by `rescope-admission-control`
investigation task 1.6/1.7, still true, unchanged by anything since).

`priority` (`schemas/chat.py:49-55`) is `Literal["interactive", "batch"]`,
client-supplied, unauthenticated, default `"interactive"`.

**The B4/B5 boundary is already written down** in
`openspec/changes/rescope-admission-control/design.md`
("B4/B5 boundary, made explicit"): B4 explicitly does not provide priority
ordering, starvation prevention, a bounded waiter count, or general
backpressure — all named as B5's territory. This investigation starts from
that boundary rather than re-deriving it.

## 1. Queueing semantics

**"Queueing instead of 429 for `priority: batch`" does not mean batch never
429s.** Read literally it would mean an unbounded wait, which is not a safe
default for an HTTP-connection-holding server (see §3). What the roadmap
line means, read against its own justification (`PHASE-A-EXECUTION-PLAN.md`
§9, Varex §8.4: a batch harness tolerates waiting far better than it
tolerates a hard-fail): **`batch` should wait substantially longer than
`interactive` before receiving the 429 that already exists**, and only
receive it immediately when a bounded amount of queueing capacity is
already exhausted.

- **Which requests may wait:** `batch`-priority requests, on the existing
  per-model semaphore, exactly where they wait today — just for longer.
- **Which requests may still receive immediate 429:** (a) `interactive`
  requests when today's `admission_wait_s` elapses — **unchanged**, `[decision]`
  interactive must not get slower; (b) `batch` requests arriving when the
  new per-model batch-queue-depth cap is already at capacity — a new,
  immediate rejection distinct from the wait-elapsed rejection (§3).
- **Does interactive remain latency-sensitive while batch queues:**
  **[decision]** yes, by construction — Gate 1's `interactive` branch is
  untouched code under the recommended option (§2, Option 1). This is the
  central design constraint the options in §2 are evaluated against.

## 2. Priority policy — options and decision matrix

Three concrete policies were evaluated. FIFO-with-differentiated-timeout is
not "the simplest option chosen by default" — §2.4 states explicitly why
the more powerful options were rejected despite solving a real gap
FIFO leaves open.

### Option 1 — FIFO, differentiated wait bound, bounded batch-waiter cap (RECOMMENDED)

Same per-model `asyncio.BoundedSemaphore`, same natural FIFO waiter order
CPython's `asyncio.Semaphore` already provides. Only the `timeout` value
passed to `wait_for` differs by priority: `interactive` keeps
`admission_wait_s`; `batch` gets a new, larger `batch_admission_wait_s`. A
new per-model counter caps how many `batch` waiters may be queued at once
(§3); over-cap `batch` requests reject immediately, before ever calling
`sem.acquire()`.

### Option 2 — Priority-preemptive dual queue

Two logical wait-queues per model (interactive, batch) sharing one permit
pool. When a permit frees, an interactive waiter is preferred over a batch
waiter regardless of arrival order. Requires replacing
`asyncio.BoundedSemaphore` with a custom wait primitive (stdlib's semaphore
has no priority-aware wakeup), plus an age-based promotion rule to bound
batch starvation under sustained interactive load.

### Option 3 — Reserved capacity split (static partition)

Reserve `R` of each model's `max_num_seqs` permits exclusively for
`interactive` (e.g. 1 of 4 on the 6gb tier); `batch` competes only for the
remaining `max_num_seqs - R`. Two semaphores per model instead of one.

### 2.4 Decision matrix

| Dimension | Option 1 (FIFO + cap) | Option 2 (priority-preemptive) | Option 3 (reserved split) |
|---|---|---|---|
| Correctness (does what B5 asks) | Yes — batch waits longer, still 429s on exhaustion | Yes, and additionally protects interactive under batch load | Partial — protects interactive only up to `R`'s size; no queueing improvement for batch beyond Option 1's |
| Fairness / starvation | **Starvation-free by construction** (FIFO — every waiter reaches the head or times out) | Batch starvation possible without the age-based promotion rule; promotion rule adds its own tunable and failure mode | No batch starvation, but `R` permanently removed from batch's ceiling even when interactive is idle — wastes capacity |
| Boundedness | New waiter cap makes it bounded (§3) | Needs the same waiter cap layered on top — doesn't solve boundedness by itself | Same — needs the same cap layered on top |
| Cancellation safety | **Reuses B4's already-proven 8-path analysis unchanged** (`rescope-admission-control/design.md`) — same primitive, only a different `timeout` value | **New primitive, all 8 paths must be re-proven from scratch** — the exact kind of correctness surface B4's investigation spent a full round on for the *simpler* structure | Reuses B4's analysis per-semaphore (two semaphores, same primitive each) — moderate re-verification, not from scratch |
| Timeout semantics | Direct extension of B4's existing contract | Needs a per-queue timeout plus a promotion-timing decision | Direct extension, per semaphore |
| Implementation complexity | Low — one new `if request.priority == "batch"` branch, one new counter | High — new concurrency primitive, new starvation-prevention logic | Medium — two semaphores, capacity-split config surface |
| Interaction with vLLM | None — vLLM has no concept of `priority`, this stays entirely InferenceX-side pre-dispatch (§4) | Same | Same |
| Observability | One new counter to log/warn on (batch-queue-depth) — fits the existing `ResponseWarning`/log pattern (`admission.py` `_warn`) | Two queues to observe, plus promotion events | Two semaphores to observe (idle-reserved-capacity is itself worth surfacing, adding another signal) |
| Maintainability | Extends a 411-line, already-well-tested file with ~15-20 lines | New primitive class, new tests for a new lifecycle | New semaphore-pair bookkeeping, new config knob with no empirical basis for `R` |
| B6 compatibility | No dependency either direction (§7) | No dependency either direction | No dependency either direction |
| Likely operational failure modes | Sustained batch flood can still delay a late-arriving interactive request up to `admission_wait_s` behind a full batch queue — **named, not solved, by this option** (§2.5) | Promotion-rule mistuning: too aggressive starves batch, too lax fails to protect interactive — an untunable-without-data knob | Reserved slot sits idle under batch-only load, permanently discarding throughput; `R` is unvalidated |

### 2.5 Decision

**[decision, confirmed this round]** Option 1. Confirmed by the change
owner against this design's decision matrix and the round-2 targeted
experiment (§2.6). It is not selected merely because it is smallest: it is
selected because it is the only option that inherits B4's already-proven
cancellation-safety analysis without modification, is starvation-free by
construction (no new tunable to get wrong), and stays inside the exact file
and primitive B4 already spent a full investigation round making provably
safe. Options 2 and 3 solve a real problem Option 1 leaves open — an
interactive request queued behind a batch flood still waits up to
`admission_wait_s` before its own 429 — but §2.6's live experiment found
this effect small at a moderate (2x-ceiling) batch flood, and neither this
investigation nor `rescope-admission-control`'s produced Varex-scale
(~50x-ceiling) evidence that the effect is severe enough to justify a new
concurrency primitive's correctness risk. If production evidence later
shows interactive requests being measurably delayed by batch queueing at
real scale, that is this change's own named, explicit follow-up (§8), not
something to solve speculatively now.

### 2.6 Targeted load-test experiment (this round)

**[verified, live experiment]** Run against a live `opt-125m` server (6gb
tier, `max_num_seqs=4`), the same model `rescope-admission-control`'s own
live experiment used, for direct comparability. Server started via
`scripts/dev.sh`-equivalent `uvicorn` invocation with
`INFERENCE_X_LOADED_MODELS=opt-125m`; no `src/` or `tests/` file was
touched — only environment configuration. Four sub-experiments, scoped to
exactly the five behaviors this change's numeric decisions depend on
(per-instruction: not a broad benchmark, no other model was loaded — model
choice does not affect admission-layer scheduling behavior, only
generation throughput, which is not what is being measured here).

**Queue-drain timing** (informs `batch_admission_wait_s`): 16 concurrent
requests fired at once against `max_num_seqs=4` (a 4x-ceiling burst, the
same multiple `rescope-admission-control` tested at n=48). Baseline solo
latency: 0.592s. Worst-case total latency across the 16: **1.569s**
(fastest 0.061s). This is the same order of magnitude as
`rescope-admission-control`'s own 1.13s worst case at a comparable multiple
— consistent, not contradictory, evidence. Implied per-wave drain rate
(16 requests / 4 permits = 4 waves in 1.569s): **~0.39s/wave**.

**FIFO fairness** (informs whether Option 1's known interactive-behind-batch
gap, §2.5, is severe at a moderate flood): 8 `batch`-priority requests
fired, then 1 `interactive`-priority request 50ms later (a 2x-ceiling
flood — deliberately modest, not Varex's ~50x scale). The interactive
request completed in 1.078s, in the middle of the batch cohort's own
0.095s-1.129s spread — **not** measurably worse off than its batch peers at
this flood size. **This does not extend to Varex's ~200-request scale**,
which was not tested (out of scope for a minimum targeted experiment) and
remains the open question named in §2.5/§8.

**Cancellation / no-leak** (confirms the primitive Option 1 reuses is safe
under a client disconnect while queued, at live-HTTP granularity, not just
unit-test granularity): 4 permits held with long-running requests; a 5th
request queued, then cancelled client-side (`asyncio.CancelledError`) while
still waiting; the 4 holds then released normally; a fresh burst of exactly
4 new requests afterward all admitted promptly (0.104s-0.604s, no stall).
**No evidence of a leaked permit** — consistent with B4's own claimed
cancellation-safety property, now reproduced at the live-HTTP layer.

**Timeout → 429 → `Retry-After` (not reproduced live this round):** two
attempts (holds at `max_tokens=200` and `max_tokens=2000`, against
`INFERENCE_X_ADMISSION_WAIT_S=2`) both failed to trigger the timeout path —
`opt-125m` is a non-instruction-tuned base model that terminates generation
at its own EOS well before any requested `max_tokens` ceiling, regardless
of the value requested, and `ChatCompletionRequest` exposes no
minimum-length knob to force a longer hold. **This is a live-HTTP
reproduction gap, not a mechanism gap**: the `wait_for` → `TimeoutError` →
`EngineSaturatedError` → 429 + `Retry-After` path is unchanged code
(`admission.py:288-295`) already covered by
`rescope-admission-control`'s own unit tests
(`test_batch_rejected_at_tier_max_num_seqs`, `_FAST_WAIT_S`) — B5 does not
alter this mechanism, only the `timeout` value passed to it per priority.
Not reproducing it live here does not weaken the Option 1 decision; it
means the specific 429/`Retry-After` HTTP contract was verified by
inspection and prior unit tests, not independently re-demonstrated over
HTTP in this pass.

**Waiter-cap capacity:** no dedicated capacity test was run — there is no
B5 code yet to cap, so a cap-specific experiment would only be measuring
unimplemented behavior. The queue-drain and FIFO-fairness sub-experiments
did exercise 16 and 9 concurrent in-flight/queued requests respectively
without any error or resource issue, which is a weak sanity signal (not a
capacity measurement) that a cap somewhat above that range is safe to ship
provisionally — see §3.1's resulting default and its explicit scope limit.

## 3. Queue bounds

**[decision]** A bare `asyncio.BoundedSemaphore` bounds permits, not
waiters (B4's own finding). B5 introduces a **per-model batch-waiter
counter** — the same `_PerModelCounter` primitive already used for KV-token
and (formerly) sequence tracking, counting *currently-queued batch
requests* rather than tokens or permits.

- **What happens when the queue is full:** a `batch` request arriving when
  `_queued_batch_counter.current(routed_model) >= cap` raises
  `EngineSaturatedError` (429, `Retry-After: 1`, same shape as today)
  **before calling `sem.acquire()` at all** — it never enters Gate 1's wait
  path, so it cannot affect semaphore lifecycle (§6).
- **Scope: per-model, not global.** Consistent with every existing gate
  (`_tracker`, `_semaphores` are both keyed by `routed_model`); a global cap
  would let one hot model starve queue capacity from every other loaded
  model, which nothing else in this file does.
- **Only `batch` counts against the cap.** `interactive` gains no new
  counter, no new cap, no new code path — this is what keeps the "zero risk
  to interactive" claim in §1 true by construction, not just by intent.
- **Resolution:** mirrors `_effective_max_num_seqs()` — skipped entirely
  (fail-open, no cap enforced) when no VRAM tier is resolved, since there is
  no `max_num_seqs` to derive a sane cap multiple from either. This inherits
  the existing fail-open posture (`admission.py:36-39`) rather than
  inventing a new failure mode for the no-tier case.
- **`[decision, evidence-informed provisional]`** cap formula: **8 ×
  `effective_max_num_seqs(routed_model)`**, resolved the same way
  `_effective_max_num_seqs` already composes tier-ceiling ∩ per-model
  override — a multiplier, not an absolute number, consistent with this
  file's existing composition style (DEC-040). Concretely: 32 on the 6gb
  tier (`max_num_seqs=4`), 64 on 12gb, 128 on 24gb. Config surface:
  `INFERENCE_X_MAX_QUEUED_BATCH_MULTIPLIER`, default `8`. **Basis:** §2.6's
  live experiment exercised 16 and 9 concurrent in-flight/queued requests
  without error — 8x headroom over `max_num_seqs` comfortably exceeds that
  exercised range for every tier. **Explicit scope limit:** this is a
  sanity-checked provisional value, not a measured capacity ceiling — no
  experiment measured actual memory/connection cost at the 32-128 range
  itself, and it was not validated against Varex's full ~200-request scale
  (`tasks.md` §1.12 stays open for that).

## 4. Timeout semantics

- **Admission deadline:** `interactive` keeps `admission_wait_s` (~5s,
  unchanged). `batch` gets a new `batch_admission_wait_s`.
  **`[decision, evidence-informed provisional]`** default **30 seconds**,
  `INFERENCE_X_BATCH_ADMISSION_WAIT_S`. **Basis:** §2.6 measured a 4x-ceiling
  burst (16 requests, `max_num_seqs=4`) draining at ~0.39s/wave, worst-case
  total 1.569s. 30s is roughly 75 wave-equivalents of headroom at that
  measured rate — chosen, not measured, as comfortably above a single-wave
  burst while staying well short of an indefinite hold on an HTTP
  connection. **Explicit scope limit:** this is a linear extrapolation from
  a 4x-ceiling measurement to a value intended to also cover much deeper
  batch floods (Varex's ~200-request scale is ~50x-ceiling) — queueing
  behavior was not measured at that depth, and per-wave rate is not
  guaranteed to stay constant that far past the tested range. `tasks.md`
  §1.11 stays open to replace this with a directly-measured value at
  Varex's actual scale.
- **Cancellation:** identical mechanism to B4 — `asyncio.wait_for` cancels
  cleanly on the caller side (client disconnect, generator `.close()`);
  CPython's `asyncio.Semaphore.acquire()` is cancellation-safe by
  construction (a cancelled-before-woken waiter never holds a permit — this
  is exactly why B4 picked this primitive, `rescope-admission-control/design.md`
  "Cancellation-aware for free"). Since Option 1 reuses the identical
  primitive with only a different `timeout` argument, this property carries
  over **unchanged, not re-derived**.
- **Timeout → 429, not another status.** Same `EngineSaturatedError`
  mapping, same `api/errors.py:36-51` handler, same `Retry-After` header
  shape. No new HTTP status is introduced.
- **`Retry-After` value:** **[decision]** stays `1.0` for both the
  wait-elapsed and the queue-full rejection paths. A computed backoff (e.g.
  scaled to `batch_admission_wait_s`) was considered and rejected — it would
  be unvalidated speculation with no data behind the specific number, and
  `EngineSaturatedError`'s existing constructor already supports a
  per-call `retry_after_s` if evidence later justifies differentiating it
  (named, not solved, here — see `tasks.md` §1 follow-up).
- **Interaction with the existing completion timeout:** **[verified]**,
  `services/chat_service.py:139,146-179`. `INFERENCE_X_STREAM_TIMEOUT_S`
  (per-token stream-stall timeout) and the engine's own completion timeout
  both start only *after* `await self._admission.admit(...)` returns
  (`chat_service.py:139`) — `timeout_s` wraps only `gen.__anext__()` calls,
  which happen strictly after admission. B5's queue wait is entirely
  contained inside `admit()`, before either of those clocks starts. No
  double-counting, no interaction bug.

## 5. Existing vLLM queue interaction

**[verified, reasoning from source]** vLLM's own scheduler queues past
`max_num_seqs` too — this was B4's own justification for a bounded wait
instead of an instant reject (`admission.py:16-18`). But vLLM's scheduler
has **no concept of InferenceX's `priority` field** — once a request is
submitted via `AsyncLLM.generate()`, every in-flight request is scheduled
identically by vLLM regardless of what `priority` InferenceX assigned it
pre-dispatch. Priority differentiation is therefore only possible **before**
dispatch, inside `AdmissionController.admit()` — the same place B4 already
lives, for the same reason B4's investigation rejected Option A (dispatch to
vLLM's queue) in its own round 2: it would require trusting vLLM's internal
queue-depth behavior at model sizes never tested, and it breaks "the engine
is never invoked for a rejected/queued request." B5 does not revisit that
rejection; it inherits it. **Nothing delegates to vLLM's queue under this
design** — InferenceX owns the entire batch-priority queueing/backpressure
decision, exactly as it already owns Gate 1 today.

## 6. Streaming vs non-streaming

**[verified]**, `services/chat_service.py`. `complete()` (line 76) and
`stream_response()` (line 139) both call `await self._admission.admit(...)`
at the identical point — before any generation, before any bytes are sent
to the client. Both already treat `admit()` as an opaque, uniform gate (the
entire point of concentrating admission logic in one method, per DEC-047
boundary discipline). B5's longer batch wait is entirely internal to
`admit()`, so **no distinction is needed between the two call sites** — this
is not a design choice being made here, it is a consequence of B4's
existing structure already being correct for this purpose.

One operational nuance, not a code distinction: for **streaming**, event 0
(the `resolved`/`warnings` prologue SSE event, `chat_service.py:154-162`)
fires only after `admit()` returns — so a `batch` request queued for
minutes produces zero bytes on the wire for that entire wait. This is the
existing behavior today, just with a longer possible silence window.
**`[unverified], deployment-environment-dependent:`** whether a reverse
proxy or load balancer in front of InferenceX has an idle-connection
timeout shorter than `batch_admission_wait_s` is outside this codebase's
control and cannot be verified here — `tasks.md` §1 records this as a
documentation item (operators must confirm their proxy's idle timeout
exceeds the configured `batch_admission_wait_s`), not a code change.

## 7. Cancellation and lifecycle correctness

Every path below is either **inherited unchanged from B4** (same primitive,
same wrapping `try/finally` in `ChatService`) or **new and analyzed here**.

| # | Path | Mechanism | Status |
|---|---|---|---|
| 1 | enqueue → wait → admit (interactive) | `wait_for(sem.acquire(), admission_wait_s)` | **Unchanged from B4** — not touched by this design |
| 2 | enqueue → wait → admit (batch) | Same call, `timeout=batch_admission_wait_s` | New value, same mechanism — inherits B4's proof (§4) |
| 3 | enqueue → cancellation while waiting (either priority) | `asyncio.Semaphore`'s built-in cancel-safety (no permit leak on a cancelled-before-woken waiter) | **Unchanged from B4**, same primitive |
| 4 | enqueue → timeout (either priority) | `TimeoutError` → `EngineSaturatedError`; no permit was ever acquired, so no release needed | **Unchanged from B4** — timeout path never acquired a permit, nothing to leak |
| 5 | enqueue → rejection at Gate 2/3 after Gate 1 acquired (either priority) | Existing `except BaseException: sem.release(); raise` wrapper (`admission.py:297-397`) | **Unchanged from B4** — untouched code |
| 6 | enqueue → engine failure post-admission | `ChatService`'s existing `try/finally` around `generate()`/`generate_stream()` | **Unchanged from B4**, outside `admit()`'s scope entirely |
| 7 | **batch queue-full rejection (new)** | Waiter-count check happens strictly *before* `sem.acquire()` is called — the request never enters Gate 1's wait/acquire path at all | **New, and lifecycle-trivial by design**: since the semaphore is never touched, there is nothing to release and no interaction with paths 1-6 |
| 8 | **batch waiter-counter accounting (new)** | Counter incremented immediately before `wait_for(sem.acquire())`, decremented in a `finally` wrapping that call (not the outer Gate 2/3 `try`) — so the counter reflects "currently waiting on the semaphore," not "currently holding a permit" | **New — must be its own `try/finally`, distinct from the existing Gate-1-acquired-then-Gate-2/3-fails wrapper**, since a request that times out at Gate 1 was never inside that outer wrapper (see `admission.py:286-295`, the `TimeoutError` branch is *before* the `try:` at line 302) |

Path 8 is the one genuinely new correctness obligation this change
introduces (mirroring B4's own finding that exactly one of eight paths
needed new code, not a blanket assumption). It requires its own
`try/finally` around the `wait_for` call specifically for the counter,
independent of the existing Gate-1/Gates-2-3 boundary — this must be
verified with a dedicated test (`tasks.md` §2) analogous to
`test_double_release_raises_value_error_instead_of_inflating_capacity` and
the two `test_gate*_failure_after_semaphore_acquired_releases_permit` tests
B4 added.

## 8. B4 compatibility invariants

1. `AdmissionController.admit()`'s signature, return type (`AdmissionResult`),
   and coroutine nature are unchanged.
2. Semaphore keying (`dict[str, asyncio.BoundedSemaphore]` by
   `routed_model`), lazy creation, and one-time sizing (`_get_semaphore`)
   are unchanged.
3. `interactive` priority's Gate 1 behavior — `admission_wait_s`, 429 shape,
   no clamp path — is **byte-for-byte unchanged**. Every existing
   `TestSequenceConcurrencyGate` test must keep passing unmodified.
4. Gates 2 and 3 (`admission.py:302-385`) are untouched: no new branch, no
   new import, no reordering.
5. `release()` (`admission.py:399-410`) is unchanged — it releases the
   sequence-concurrency permit and decrements `_tracker`; it does not need
   to know about the new batch-waiter counter, since that counter is
   decremented at the wait site itself (§7, path 8), not at release time.
6. The single-process, no-`--workers`, no-dynamic-model-load/unload
   assumptions `_get_semaphore`'s race-safety already depends on
   (`admission.py:223-232`) are unchanged and still hold — this change adds
   no new cross-coroutine mutable state that isn't already the same
   dict-plus-lazy-insert shape already proven race-free.

## 9. B6 boundary

This change touches only `routing/admission.py`, `core/settings.py`, and
`api/deps.py`'s `_build_admission_controller` wiring (plus
`tests/unit/test_admission.py`). It does **not** touch, and must not be
made to touch:

- `EnginePool` (`engines/pool.py`) or its construction in `api/deps.py`'s
  `_build_engine_pool` loop.
- `pool_size` / same-model-replica semantics (`migrate-async-llm-engine`
  Decision 3 — still "not guaranteed, not forbidden," B6's exclusively to
  resolve).
- `utils/vllm_pool_config.py` (VRAM estimation heuristics, DEC-045/046).
- `enforce_eager` coupling or the global `threading.Lock` serializing
  `llm_engine.step()` (`REVIEW-2026-08-03-architecture.md` §3.5).

No dependency was found, in either direction, between this change's
mechanism (a per-model waiter cap and a priority-differentiated timeout
inside `AdmissionController`) and B6's process-split work. B6 splitting a
model into its own process does not change how many `AdmissionController`
instances exist per process (still exactly one, `rescope-admission-control/design.md`
"Semaphore ownership and keying" — "single `AdmissionController` instance,
single event loop" — this remains true per-process after a B6 split, it
just means fewer models per instance). This change's new batch-waiter
counter is likewise process-local `_PerModelCounter` state with the same
property.

## 10. B5 vs B6 ordering

**Confirmed independent, zero file overlap.** B5 (this change) touches
`admission.py`/`settings.py`/`deps.py`'s admission wiring; B6 touches
`EnginePool`/`vllm_pool_config.py`/engine-construction and process launch.
Neither reads nor writes state the other owns. Recommending B5 before B6
(consistent with the prior Phase-B planning report) remains a sequencing
preference for context continuity — the admission-control mental model is
already loaded from B4 — not a technical dependency; B6 could equally be
done first without blocking or invalidating this design.

## Remaining `[unverified]` items

Both numeric defaults are now **decided** (§3, §4), evidence-informed by
§2.6's live experiment — but explicitly not validated at production scale:

1. `batch_admission_wait_s=30` is an extrapolation from a measured
   4x-ceiling burst (~0.39s/wave) to a value meant to also cover Varex's
   ~50x-ceiling scale — the deeper scale itself was not measured.
2. The batch-waiter cap formula (`8 × max_num_seqs`) is sanity-checked
   against 16-request concurrency, not measured against actual
   memory/connection cost at the resulting 32-128 range, nor against
   Varex's ~200-request scale.
3. Whether a production-realistic (Varex-scale) batch flood measurably
   delays a late-arriving `interactive` request under Option 1's FIFO
   ordering — §2.6 found no measurable effect at a 2x-ceiling flood, a much
   smaller flood than Varex's own workload — left unmeasured at real scale.
4. Reverse-proxy/load-balancer idle-timeout interaction with a long-held
   streaming connection during the batch wait (§6) — deployment-environment
   dependent, not verifiable from this repository.
5. The timeout → 429 → `Retry-After` path was not independently reproduced
   over live HTTP this round (§2.6) — `opt-125m`'s early-EOS behavior made
   it impractical to force a hold longer than `admission_wait_s` without an
   API-level minimum-length knob this schema doesn't expose. The mechanism
   itself is unit-test-proven and unchanged by this design.

All five are recorded as follow-up tasks in `tasks.md` §1, not silently
dropped. None of them block implementation-readiness — they are
scale/environment validation gaps on top of an already-decided mechanism
and already-decided (if provisional) numeric defaults, the same posture
`rescope-admission-control` shipped B4 under for its own `admission_wait_s`.
