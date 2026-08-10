# Tasks: add-batch-priority-queueing

**Sections 1-5 are complete.** Option 1 is implemented on
`feat/batch-priority-queueing` (branched from `develop` @ `1c67f51`):
`src/inference_x/routing/admission.py`, `core/settings.py`, `api/deps.py`,
plus tests in `tests/unit/test_admission.py` and `test_chat_service.py`.
Full unit suite, `ruff`, `mypy`, and `openspec validate --all --strict` all
pass. **Section 6 (documentation/archive) is not started** — this
implementation pass was not asked to update `docs/DECISIONS.md` or
`docs/PHASE-A-ARCHITECTURE.md`, and this change is explicitly not archived
(nor is B4). Nothing has been committed, pushed, merged, or released.

## 1. Investigation and evidence

- [x] 1.1 Read `routing/admission.py` in full (post-B4); document Gate 1's
      current uniform-timeout behavior and its exact line references.
- [x] 1.2 Read `services/chat_service.py` in full; confirm `complete()` and
      `stream_response()` both call `admit()` identically before any
      generation, and locate the stream-timeout/completion-timeout call
      sites relative to `admit()`.
- [x] 1.3 Read `core/settings.py` in full; confirm `admission_wait_s` is a
      single process-wide value with no priority branch, and its exact
      default/provenance (`INFERENCE_X_ADMISSION_WAIT_S`, default `5.0`,
      itself provisional per its own docstring).
- [x] 1.4 Read `api/deps.py`'s `_build_admission_controller`; confirm how
      settings flow into the single `AdmissionController` instance.
- [x] 1.5 Read `schemas/chat.py`'s `priority` field; confirm no HTTP
      contract change is needed (`Literal["interactive","batch"]` already
      exists).
- [x] 1.6 Read the active `rescope-admission-control` change (`proposal.md`,
      `design.md`, `tasks.md`) in full; extract the "B4/B5 boundary, made
      explicit" section as this change's starting scope statement.
- [x] 1.7 Read the current `openspec/specs/platform/spec.md` "Incremental
      architecture" requirement to identify which scenario this change's
      spec delta must modify vs. add.
- [x] 1.8 Read `docs/DECISIONS.md` DEC-040 (Gate 1's original rationale) in
      full via the DECISIONS.md index.
- [x] 1.9 Confirm Varex's workload shape (`docs/REVIEW-2026-08-03-architecture.md`
      §8.3-8.4) as the concrete motivating use case for tolerating a long
      wait over a hard 429.
- [x] 1.10 Confirm `config/vram_tiers.yaml`'s `max_num_seqs` values (4/8/16
      across 6gb/12gb/24gb tiers) as the scale this change's provisional
      defaults must be sane against.
- [x] 1.11 **Targeted load-test experiment, run this pass, at moderate
      scale only:** live `opt-125m` server (6gb tier, `max_num_seqs=4`,
      config-only, no `src/`/`tests/` change), 16 concurrent requests (a
      4x-ceiling burst). Worst-case total latency 1.569s, ~0.39s/wave
      implied drain rate — used to set `batch_admission_wait_s=30`
      (design.md §2.6, §4). **Not** run at Varex's ~50x-ceiling scale — see
      1.11a.
  - [ ] 1.11a **Follow-up (not run this pass):** repeat 1.11 at
        Varex-representative scale (~150-200 concurrent `batch` requests)
        to replace the linear-extrapolation basis for
        `batch_admission_wait_s=30` with a directly-measured value
        (design.md "Remaining `[unverified]` items" #1).
- [ ] 1.12 **Load-test experiment (not run this pass):** measure
      queued-request memory/connection cost at realistic batch-queue depths,
      to replace the batch-waiter cap's provisional default
      (`8 × max_num_seqs`) with an evidence-based one (design.md §3,
      §"Remaining unverified items" #2).
- [x] 1.13 **Measurement, run this pass, at moderate scale only:** 8
      `batch`-priority requests fired, then 1 `interactive`-priority
      request 50ms later (2x-ceiling flood). Interactive request completed
      in 1.078s, within the batch cohort's own spread — no measurable
      starvation at this scale (design.md §2.6). **Not** measured at
      Varex-representative scale — see 1.13a.
  - [ ] 1.13a **Follow-up (not run this pass):** repeat 1.13 with a
        Varex-scale (~150-200) batch flood ahead of the interactive
        request — the evidence that would actually justify revisiting
        Option 2/3 (design.md "Remaining `[unverified]` items" #3).
- [ ] 1.14 **Deployment note (not verified this pass):** confirm or document
      the reverse-proxy/load-balancer idle-connection-timeout interaction
      for a `batch` request streaming-queued near `batch_admission_wait_s`
      (design.md §6, §"Remaining unverified items" #4).
- [ ] 1.15 **Live-HTTP timeout reproduction (attempted, not achieved, this
      pass):** two attempts to trigger Gate 1's `TimeoutError` → 429 path
      over live HTTP both failed because `opt-125m` (non-instruction-tuned)
      terminates generation at its own EOS before any requested
      `max_tokens` ceiling, regardless of value requested. The mechanism
      itself is unchanged from B4 and already unit-test-proven
      (`test_batch_rejected_at_tier_max_num_seqs`); this task is to either
      find/build a model or fixture that reliably holds a permit past a
      short `admission_wait_s`, or accept unit-level coverage as sufficient
      (design.md §2.6, §"Remaining unverified items" #5).

## 2. Decision

- [x] 2.1 Evaluate three priority-queueing mechanisms (FIFO+cap,
      priority-preemptive dual queue, reserved-capacity split) against a
      decision matrix covering correctness, fairness/starvation,
      boundedness, cancellation safety, timeout semantics, complexity,
      vLLM interaction, observability, maintainability, B6 compatibility,
      and operational failure modes (design.md §2.4).
- [x] 2.2 Recommend Option 1 (FIFO, differentiated wait bound, bounded
      batch-waiter cap) with explicit rejection reasons for Options 2 and 3
      (design.md §2.5).
- [x] 2.3 Trace all cancellation/lifecycle paths (8 total, 6 inherited
      unchanged from B4, 2 new) and identify the one genuinely new
      correctness obligation: the batch-waiter counter needs its own
      `try/finally` independent of the existing Gate-1/Gates-2-3 boundary
      (design.md §7, path 8).
- [x] 2.4 Confirm zero file/state overlap with B6 and record the explicit
      boundary (design.md §9).
- [x] 2.5 Confirm B5/B6 are independently orderable and record the
      sequencing rationale (design.md §10).
- [x] 2.6 **Change-owner sign-off, this pass:** Option 1 confirmed as the
      selected (not merely recommended) design — mirrors
      `rescope-admission-control`'s own round-2-recommends/round-3-confirms
      structure (design.md §2.5). No longer blocks section 3.
- [x] 2.7 **Change-owner decision, this pass:** ship with
      evidence-informed provisional numeric defaults now
      (`batch_admission_wait_s=30`, batch-waiter cap `8 × max_num_seqs`),
      set from the targeted experiment at 1.11/1.13 rather than invented,
      but explicitly not validated at Varex's full scale — 1.11a/1.12/1.13a
      remain open follow-ups, not implementation blockers (design.md §3,
      §4, §"Remaining unverified items"). No longer blocks section 3.

## 3. Implementation (complete)

- [x] 3.1 Added `batch_admission_wait_s` (default `30`,
      `INFERENCE_X_BATCH_ADMISSION_WAIT_S`) and `batch_waiter_multiplier`
      (default `8`, `INFERENCE_X_MAX_QUEUED_BATCH_MULTIPLIER`) to
      `core/settings.py`, following the exact pattern `admission_wait_s`
      already uses.
- [x] 3.2 Wired both settings through `api/deps.py`'s
      `_build_admission_controller` into `AdmissionController.__init__`.
- [x] 3.3 Added `self._batch_waiters = _PerModelCounter()` (reused, not
      reinvented — the same primitive already tracking KV-token
      reservations) to `AdmissionController.__init__`. The per-model cap
      itself is derived inline (`self._batch_waiter_multiplier *
      effective_max_num_seqs`) at the one call site in `admit()` rather than
      a separate named method — `_effective_max_num_seqs` is a real
      resolution (tier ∩ per-model override, two data sources); the
      batch-waiter cap is one multiplication with no second data source, so
      a wrapping method would have no logic of its own. **Deviation from the
      task's literal wording** ("resolution helper"), not from the design's
      substance — the fail-open behavior (cap enforcement skipped when no
      tier is resolved, since `effective_max_num_seqs` is `None`) is
      unchanged and verified in tests.
- [x] 3.4 Implemented Gate 1's three-way branch (`admission.py` `admit()`):
      `effective_max_num_seqs is None` (unchanged fail-open warn),
      `priority == "batch"` (new: waiter-cap check → `_get_semaphore` →
      increment counter → `wait_for(..., batch_admission_wait_s)` in its own
      `try/finally` decrementing the counter → unchanged `TimeoutError`
      mapping), else `interactive` (byte-for-byte the original code, just
      renamed from the removed `else` branch). Waiter-cap check happens
      strictly before `_get_semaphore`/`sem.acquire()` — an over-cap
      rejection never touches the semaphore (verified,
      `test_batch_waiter_cap_overflow_rejects_immediately_without_acquiring_semaphore`).
- [x] 3.5 Updated `admission.py`'s module docstring (Gate 1 description) and
      `admit()`'s `Raises:` section to describe the priority-differentiated
      wait and the batch waiter-cap rejection.

## 4. Tests (complete)

- [x] 4.1 `interactive` regression: all pre-existing `TestSequenceConcurrencyGate`
      and `TestBoundedWaitAndCancellation` cases pass unmodified, plus new
      `test_interactive_keeps_its_own_short_wait_bound_unaffected_by_batch_default`
      proves interactive is not slowed by `batch_admission_wait_s`'s
      existence (design.md §8 invariant 3).
- [x] 4.2 `test_batch_waits_then_succeeds_once_a_slot_frees` — batch waits
      past a deliberately-too-short `admission_wait_s` and is admitted once
      the holder releases, within `batch_admission_wait_s`.
- [x] 4.3 `test_batch_waiter_cap_overflow_rejects_immediately_without_acquiring_semaphore` —
      a request arriving when the cap (`multiplier=1`) is already full 429s
      within 1s against a 5s wait bound, proving no wait occurred; a
      dedicated boundary test
      (`test_batch_waiter_cap_boundary_exact_capacity_admits_the_last_one`)
      additionally verifies the `>=` cap check at `multiplier=2` (2 queue,
      not 1 or 3).
- [x] 4.4 `test_batch_timeout_raises_engine_saturated_with_retry_after` (429 +
      `retry_after_s == 1.0`) and
      `test_batch_waiter_count_decremented_after_timeout` (counter back to 0).
- [x] 4.5 `test_batch_waiter_count_decremented_after_cancellation` — cancels a
      queued batch waiter, asserts the counter returns to 0 and, separately,
      that the held permit is still exactly the original holder's (releasing
      it and re-admitting succeeds — no semaphore leak either).
- [x] 4.6 `test_fifo_order_preserved_not_priority_preemption` — a batch
      waiter and a later-arriving interactive waiter both queue behind one
      held permit; releasing it admits the batch waiter first (FIFO arrival
      order), explicitly not priority preemption, matching the approved
      Option 1 tradeoff (design.md §2.5) rather than inventing a different
      one.
- Additional coverage beyond the original plan:
  `test_batch_waiter_count_decremented_after_successful_admission`,
  `test_no_waiter_count_leak_after_gate2_rejection`,
  `test_no_waiter_count_leak_after_gate3_rejection` (design.md §7 path 8's
  specific new correctness obligation),
  `test_multiple_models_have_independent_batch_waiter_state`, and
  `test_batch_priority_non_streaming_and_streaming_both_time_out_through_the_same_gate`
  in `test_chat_service.py` (batch-priority analog of the existing
  `TestBoundedAdmissionSharedByBothPaths`, proving `complete()` and
  `stream_response()` share the identical batch admission path).
- **Regression found and fixed in an existing B4 test:**
  `test_batch_rejected_at_tier_max_num_seqs` set only `admission_wait_s`
  for a `priority="batch"` scenario that requires a real timeout to elapse;
  under B5's dual-timeout split it silently fell back to the new
  `batch_admission_wait_s` default (30s), turning a ~0.05s test into a
  ~30s one. Fixed by also passing `batch_admission_wait_s=_FAST_WAIT_S`.
  No other existing test was affected — swept every `priority="batch"`
  test that resolves a tier and confirmed each either never contends the
  semaphore (no wait occurs regardless of timeout value) or was this one.

## 5. Validation (complete)

- [x] 5.1 Change owner elected to ship provisional defaults now (§2.7);
      1.11a/1.12/1.13a remain open, non-blocking follow-ups, not re-run in
      this implementation pass.
- [x] 5.2 `openspec validate --all --strict` → all 3 items pass
      (`change/add-batch-priority-queueing`, `spec/platform`,
      `change/rescope-admission-control`).
- [x] 5.3 Full unit suite: 550 passed, 1 pre-existing xfail, 0 regressions
      (11.87s). `TestReservationLifecycle`'s 9 named lifecycle tests pass
      unmodified — untouched by this change (they exercise interactive
      priority only). `ruff check .` and `mypy src/` both clean.

## 6. Documentation / archive (not started)

- [ ] 6.1 New `docs/DECISIONS.md` entry recording the chosen mechanism and
      why (mirrors DEC-040 and the still-pending `rescope-admission-control`
      task 5.1 precedent).
- [ ] 6.2 `docs/PHASE-A-ARCHITECTURE.md` §10 updated to mark B5 complete.
- [ ] 6.3 Archive this change (`openspec archive add-batch-priority-queueing`)
      only after `rescope-admission-control` (B4) has itself been archived
      or a deliberate decision is made to archive out of order — not
      performed by this task.
