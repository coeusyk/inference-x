# Proposal: add-batch-priority-queueing

## Status

**Implemented (Option 1). Sections 1-5 of `tasks.md` are complete.** Round 1
covered source investigation, options analysis, and a recommendation. Round
2 confirmed Option 1 as the implementation direction and ran a targeted,
minimum-scope live experiment (design.md §2.6) against `opt-125m` (6gb tier)
to inform the two numeric defaults `batch_admission_wait_s` and the
per-model batch-waiter cap, both set to evidence-informed provisional
values — `batch_admission_wait_s=30s` and a per-model cap of
`8 × max_num_seqs` (design.md §3, §4) — not yet validated at Varex's full
~200-request scale. Implementation now exists on `feat/batch-priority-queueing`:
`src/inference_x/routing/admission.py`, `core/settings.py`, `api/deps.py`,
plus tests in `tests/unit/test_admission.py` and `test_chat_service.py`
(`tasks.md` §3-4). Full unit suite, `ruff`, `mypy`, and
`openspec validate --all --strict` all pass (`tasks.md` §5). `tasks.md` §6
(documentation/archive: `docs/DECISIONS.md`, `docs/PHASE-A-ARCHITECTURE.md`,
archiving this change) has not started. Nothing has been committed, pushed,
merged, or released. See design.md for full evidence, including "Decision"
(§2.5), the experiment itself (§2.6), and "Remaining `[unverified]` items"
for what is still open at production scale versus what is now decided.

## Why

Phase-B roadmap item **B5**, as named in `docs/PHASE-A-EXECUTION-PLAN.md:271`
and the architecture review's roadmap table
(`docs/REVIEW-2026-08-03-architecture.md` §9): **"Queueing instead of 429
for `priority: batch`."** Filed as "Varex blocker #2" — Varex (the
downstream SPRT prompt-evaluation consumer, `docs/REVIEW-2026-08-03-architecture.md`
§8.4) runs batches of up to ~200 trials sharing one system prompt
(`docs/REVIEW-2026-08-03-architecture.md` §8.3). A statistical experiment
harness like Varex's SPRT machinery tolerates a long wait for a slot far
better than it tolerates a hard-fail 429 mid-run, since a 429 either aborts
the experiment or forces Varex to implement its own retry/backoff loop
against a server that already knows the request is admissible, just not yet.

B4 (`openspec/changes/rescope-admission-control`, merged to `develop` at
`1c67f51`) replaced Gate 1's instant-reject with a bounded
`asyncio.wait_for(sem.acquire(), timeout=admission_wait_s)` — but
**uniformly for both priorities** (`admission.py:266-268`: "429 for either
priority — there is no clamp path for a sequence slot"). B4's own
`design.md` ("B4/B5 boundary, made explicit") names this gap explicitly and
assigns it to B5: no priority ordering between `interactive`/`batch`, no
starvation prevention, no bounded waiter count (a bare `Semaphore` bounds
*permits*, not the number of coroutines queued behind `acquire()`), no
queue-depth backpressure. This change investigates and designs the
mechanism that closes that gap.

## What Changes

- `AdmissionController.admit()`'s Gate 1 (`routing/admission.py`) gains a
  priority-differentiated wait: `interactive` keeps today's
  `admission_wait_s` (~5s) unchanged; `batch` waits on the same per-model
  `asyncio.BoundedSemaphore`, in the same FIFO order, but bounded by a new,
  much larger `batch_admission_wait_s`.
- A new per-model **batch-waiter cap** bounds how many `batch` requests may
  be queued (waiting on the semaphore) at once — a request arriving when the
  cap is already reached gets today's `EngineSaturatedError` (429 +
  `Retry-After`) immediately, without ever touching the semaphore. This is
  the backpressure valve for the "bare semaphore has unbounded waiters"
  gap.
- Two new settings (`core/settings.py`): `INFERENCE_X_BATCH_ADMISSION_WAIT_S`
  (default `30`, seconds) and `INFERENCE_X_MAX_QUEUED_BATCH_MULTIPLIER`
  (default `8`, resolved as `multiplier × effective_max_num_seqs` per model
  — see design.md §3 "Queue bounds" for the exact resolution shape). Both
  are evidence-informed provisional defaults from design.md §2.6's targeted
  live experiment, not yet validated at Varex's full ~200-request scale
  (design.md "Remaining `[unverified]` items").
- `interactive` priority gains **zero** new gates, settings, or latency —
  every existing `interactive` test and behavior is unchanged by
  construction (Gate 1's `interactive` branch is untouched code).
- Gates 2 (context) and 3 (KV pressure) are untouched. `release()` is
  untouched. The semaphore-lifecycle machinery B4 proved safe (8 termination
  paths, `design.md` "Semaphore lifecycle analysis") is reused verbatim, not
  redesigned — see this change's design.md "B4 compatibility invariants."

## Impact

- Affected code: `src/inference_x/routing/admission.py`,
  `src/inference_x/core/settings.py`, `src/inference_x/api/deps.py` (wiring
  the two new settings into `_build_admission_controller`), and
  `tests/unit/test_admission.py` (new cases; existing cases must keep
  passing unmodified — see design.md).
- Affected spec: `openspec/specs/platform/spec.md`, requirement
  "Incremental architecture" — see this change's
  `specs/platform/spec.md` delta.
- Explicitly NOT affected: `EnginePool`, `pool_size`/replica semantics,
  `utils/vllm_pool_config.py`, `enforce_eager`, the global step lock — all
  reserved for B6 (`docs/REVIEW-2026-08-03-architecture.md` §3.5). See
  design.md "B6 boundary."
- No public HTTP contract change: `priority: batch` already exists on
  `ChatCompletionRequest` (`schemas/chat.py:49`); this change only alters
  how long a `batch` request may wait before its existing 429 shape fires,
  and adds a new (also-429) rejection path for an already-full batch queue.
