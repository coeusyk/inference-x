# Tasks: rescope-admission-control

**Option C is implemented (`proposal.md` → "Status", `design.md` →
"Decision: Option C approved").** Section 1 (investigation/design) and
section 2 (implementation) are complete and merged into `src/`/`tests/` on
branch `feat/admission-control-semaphore`, not yet committed. Section 4's
live GPU smoke tests (4.3, 4.3a) remain unchecked — this implementation pass
ran unit tests only, no real vLLM/GPU session. Section 5 (docs) and section 6
(final review/archive) are for a separate, later pass — this change is not
archived yet.

## 1. Investigation and verification

- [x] 1.1 Read `routing/admission.py` in full; document all three gates
- [x] 1.2 Read `test_admission.py` in full; confirm existing gate-level coverage
- [x] 1.3 Read `test_chat_service.py`'s admit/release lifecycle tests
- [x] 1.4 Read vLLM 0.22.1 scheduler source (`add_request`, `schedule`,
      preemption, `max_num_running_reqs`)
- [x] 1.5 Read vLLM 0.22.1 `AsyncLLM.generate()` and `core.py` rejection paths
- [x] 1.6 Confirm single-process, no-`--workers` deployment (`scripts/dev.sh`)
- [x] 1.7 Confirm `AdmissionController` has no `async`/`await` (race-safety)
- [x] 1.8 Trace `pool_size` / `EnginePool` structure for multi-model and
      same-model-replica behavior
- [x] 1.9 Read `docs/PHASE-A-ARCHITECTURE.md` for current admission framing
- [x] 1.10 Read the archived `add-admission-control` change (`design.md`,
      `proposal.md`, `tasks.md`) — found Gate 1 is *not* part of that change
- [x] 1.11 Locate and read DEC-040 (`docs/DECISIONS.md`) — Gate 1's actual
      documented rationale, via `git log --follow -p -- admission.py`
- [x] 1.12 Confirm current committed `specs/platform/spec.md`'s
      "Sequence-concurrency ceiling" scenario (lines 46-51) and its exact text
- [x] 1.13 **Live sustained-load experiment run** against actual vLLM 0.22.1 /
      RTX 4060 (`opt-125m`, 6gb-tier `max_num_seqs=4`). Confirmed empirically:
      48/48 requests at 4x the ceiling all completed (worst case 1.13s);
      completion latency stayed flat across 90 requests over 45s sustained
      arrival (no starvation/growth signature). See `design.md` → "Empirical
      findings" for full numbers and the caveat that this was one small/fast
      model — not yet verified for a 7B-class model with longer per-request
      latency and deeper realistic queue depth.
- [x] 1.13a **Closed [source-derived], not by live benchmark.** Direct
      reads of `Scheduler.add_request()`, `EngineCore.add_request()`, and
      `_reject_add_in_shutdown` confirm the queue-vs-reject property is
      structurally independent of model size, KV capacity, `max_model_len`,
      quantization, memory pressure, and pool configuration — see
      `design.md` → "Round 3" → "3a". The *timing* question (how long a
      larger/slower model's queue wait actually runs) is separately tracked
      as an open, explicitly-provisional config-default question, not a
      blocker — see `design.md` → "Acquisition, rejection, and timeout
      semantics".
- [x] 1.13b **Closed [verified, live].** A live experiment against opt-125m
      forced a genuine stall (blockers still running, both queued requests
      at zero output tokens) and cleanly cancelled one — `CancelledError`
      delivered to a never-started request, no engine error state, sibling
      unaffected, engine healthy afterward. Took 4 attempts to correctly
      time (see `design.md` → "Round 3" → "3b" for why the first 3 failed
      and what changed).
- [x] 1.13c **Closed [source-derived], not by live experiment** (explicitly
      avoided inducing a real engine crash per instruction). Source tracing
      shows "a queued request fails inside the engine while siblings
      continue" is not a distinct vLLM-internal failure mode — it collapses
      into either cancellation (1.13b) or a whole-engine failure that is
      *not* sibling-isolated by construction. See `design.md` → "Round 3" →
      "3c".
- [x] 1.15 Investigated a third option (bounded per-model semaphore) per
      instruction not to constrain the comparison to A/B — see `design.md` →
      "A third option" and the three-way decision matrix
- [x] 1.14 Change owner decision recorded: **Option C**, per `proposal.md` →
      "Status" and `design.md` → "Decision: Option C approved". A and B are
      rejected with reasons recorded in the same section.
- [x] 1.16 Semaphore lifecycle rigor completed — all 8 termination paths
      traced (`design.md` → "Semaphore lifecycle analysis"); the one unsafe
      path in the original `tasks.md` 2.2c sketch (release-after-a-later-gate-
      fails, once the semaphore is already held) identified and given a
      specific fix (`try/except`-release wrapping Gates 2/3, semaphore
      acquired first).
- [x] 1.17 Semaphore ownership/keying investigated — same key space as
      today's counters, no dynamic load/unload to account for, `pool_size >
      1` limitation restated as unchanged (`design.md` → "Semaphore
      ownership and keying").
- [x] 1.18 Primitive comparison completed (`asyncio.Semaphore` vs. bounded
      `Queue` vs. explicit waiters vs. counter+`Condition` vs. vLLM-only) —
      `design.md` → "Primitive comparison". Fairness verified from CPython
      3.13 `asyncio/locks.py` source, not assumed — `design.md` → "Fairness".
- [x] 1.19 429/timeout contract specified (deadline start point, streaming
      vs. non-streaming sharing one deadline, HTTP mapping, config surface)
      — `design.md` → "Acquisition, rejection, and timeout semantics". The
      exact `admission_wait_s` numeric default for untested model sizes is
      explicitly left open, not invented.
- [x] 1.20 B4/B5 boundary made explicit, including the semaphore's
      unbounded-waiters property as a documented (not solved) limitation —
      `design.md` → "B4/B5 boundary, made explicit".

## 2. Implementation (Option C — done, on branch `feat/admission-control-semaphore`)

- [x] 2.1 Added `_get_semaphore(routed_model, max_num_seqs)` to
      `AdmissionController` (lazy `dict[str, asyncio.BoundedSemaphore]`, same
      key space as `_seq_tracker`/`_tracker` — `_seq_tracker` removed).
      `BoundedSemaphore`, not plain `Semaphore`: verified by
      `test_double_release_raises_value_error_instead_of_inflating_capacity`
      that an over-release raises `ValueError` rather than silently inflating
      the ceiling.
- [x] 2.2 `AdmissionController.admit()` is now `async def`; the semaphore is
      acquired as Gate 1 (first, preserving the prior gate order) via
      `asyncio.wait_for(sem.acquire(), timeout=self._admission_wait_s)`,
      mapping `TimeoutError` to `EngineSaturatedError` (429 + `Retry-After`) —
      matches `design.md`'s code shape exactly.
- [x] 2.3 Gates 2/3 + the final commit are wrapped in `try/except
      BaseException: sem.release(); raise` — the corrected fix for lifecycle
      path 7. Verified by
      `test_gate2_failure_after_semaphore_acquired_releases_permit` and
      `test_gate3_failure_after_semaphore_acquired_releases_permit`.
- [x] 2.4 `release()` stays synchronous; releases the sequence-concurrency
      permit (`self._semaphores.get(routed_model)`, `None` only when the gate
      was skipped entirely — no tier resolved) alongside the existing
      `self._tracker.add(routed_model, -reserved_tokens)`. `_seq_tracker`
      removed.
- [x] 2.5 `ChatService.complete()` and `stream_response()`'s one call site
      each updated: `admitted = await self._admission.admit(...)`.
- [x] 2.6 Added `admission_wait_s` to `core/settings.py`
      (`INFERENCE_X_ADMISSION_WAIT_S`, default `5.0`) mirroring
      `stream_timeout_s`'s pattern; wired into `api/deps.py`'s
      `_build_admission_controller` the same way `tier` already is. Default
      is documented in-line as provisional (~4x headroom over the 1.13s
      opt-125m worst case; not validated for larger models — 4.3a tracks the
      follow-up).
- [ ] 2.7 **Deliberately deferred, not done.** Resolving the
      `apply_tier_knobs` / `_effective_max_num_seqs` duplicate `max_num_seqs`
      computation would touch `utils/vllm_pool_config.py` (engine-construction
      code) for a behavior-preserving refactor Option C does not need —
      `design.md`'s own config-duplication finding already says "not fixed in
      this change." Left as a separate, optional follow-up; not required for
      Option C's correctness.
- [x] 2.8 `specs/platform/spec.md`'s `## MODIFIED Requirements` section
      (written in the investigation phase) was re-checked against the actual
      implementation and matches exactly — no divergence, carried forward
      unchanged as anticipated.
- [x] 2.9 Added `test_completion_timeout_raises_runtime_error` to
      `test_vllm_engine_async.py` — `VLLMEngine.generate()`'s
      `_COMPLETION_TIMEOUT_S` path had zero prior coverage.
- [x] 2.10 Re-verified: the only suspension point in `admit()` is
      `sem.acquire()`; the semaphore-creation check-then-insert in
      `_get_semaphore` has no `await` between them (race-free under the
      single-process deployment); Gates 2/3 remain fully synchronous inside
      the `try` block, so no new interleaving window exists beyond the
      semaphore's own atomic acquire. No new TOCTOU introduced.

## 3. Regression tests (done, except the live smoke tests in §4)

- [x] 3.1 Capacity-at-limit, above-limit, and vLLM-queued-and-served-within-bound
      — `test_concurrent_admission_waits_then_succeeds_once_a_slot_frees`
      (unit-level, real `asyncio.BoundedSemaphore`, not a live vLLM run).
- [x] 3.2 Legitimate-policy-rejection when the bound is genuinely exceeded —
      `test_timeout_after_wait_elapses_raises_engine_saturated`.
- [x] 3.3 Cancellation while admitted and waiting on the new bounded path —
      `test_cancellation_while_waiting_does_not_leak_a_permit`.
- [x] 3.3a **Lifecycle path 7 specifically** —
      `test_gate2_failure_after_semaphore_acquired_releases_permit` and
      `test_gate3_failure_after_semaphore_acquired_releases_permit`, each
      proving the permit is available again immediately (not after a wait)
      following the gate-2/3 rejection.
- [x] 3.4 Generation failure after admission (no counter drift) — pre-existing
      `test_chat_service.py` `test_engine_exception_releases_exactly_once`,
      now asserting against the semaphore's `_value` instead of the removed
      `_seq_tracker`.
- [x] 3.5 Concurrent admissions against the same model —
      `test_concurrent_admission_waits_then_succeeds_once_a_slot_frees` plus
      pre-existing `test_release_receives_the_reservation_admit_produced`
      (two genuinely concurrent streams against one model).
- [x] 3.6 Counter cleanup after every termination path — all 9 existing
      `test_chat_service.py` lifecycle tests updated (not duplicated) to run
      with a real tier/semaphore and assert the semaphore returns to full
      value, not just the KV tracker.
- [x] 3.7 Multi-model isolation unchanged —
      `test_sequence_counts_are_tracked_per_model` (pre-existing, still
      passes against the semaphore).
- [x] 3.8 `pool_size > 1` — documented as unsupported in `design.md`
      ("Semaphore ownership and keying"), not tested, as directed.
- [x] 3.9 No regression to existing 429 semantics for genuinely-exhausted
      capacity — `TestKvSaturation` (pre-existing, unaffected by Option C)
      plus the new sequence-gate timeout test above.
- [x] 3.10 `/v1/metrics` and `/metrics` unaffected — pre-existing
      `test_observability.py`/`test_routes.py` coverage, confirmed still
      green in the full `pytest tests/unit` run; no admission code touches
      either endpoint.
- [x] 3.11 Streaming and non-streaming use the identical bounded admission
      gate, not two separate code paths — added
      `test_non_streaming_and_streaming_both_time_out_through_the_same_gate`
      (`test_chat_service.py`), a single-slot model saturated by a held
      admission, asserting both `ChatService.complete()` and the first
      `stream_response()` chunk raise the same `EngineSaturatedError` after
      the same `admission_wait_s`. Added after review flagged that the
      original 9 lifecycle tests all ran at `max_num_seqs=4` with a single
      request in flight, so none of them ever actually waited or timed out —
      this closes that gap.
- [x] 3.12 Verified — not just assumed — the "semaphore cannot silently
      inflate due to an over-release" claim: added
      `test_stray_release_below_capacity_is_not_caught_by_bounded_semaphore`
      (`test_admission.py`). Finding: `BoundedSemaphore.release()` only
      raises when the release would push its internal count to or past the
      value it was constructed with; a stray release below that boundary
      silently inflates capacity by one. Not a regression — the pre-Option-C
      `_PerModelCounter` had the same contract-reliance (a stray decrement
      under-counted in-flight requests, equally allowing one admission past
      the ceiling) — but the earlier `test_double_release_raises_value_error…`
      test only exercised `max_num_seqs=1`, where the boundary is hit
      immediately and looked like a general guarantee. Not reachable via the
      real call sites: `ChatService.complete()`/`stream_response()` call
      `release()` from exactly one place, only after a successful `admit()`,
      so this is a defense-in-depth gap in the primitive, not a live bug in
      the shipped feature. No lifecycle redesign made for this — closing it
      fully would require per-reservation identity tracking through
      `release()`, which is a new mechanism beyond what `design.md` specifies
      and out of scope for this pass.

## 4. Full validation

- [x] 4.1 `pytest tests/unit -q` — 535 passed, 1 xfailed (pre-existing), no
      regressions.
- [x] 4.2 `mypy src/` — "Success: no issues found in 51 source files."
- [ ] 4.3 Live smoke test: burst requests against a real model at the
      resolved `max_num_seqs` ceiling, confirm Option C's behavior matches
      the design. **Not run this pass** — this implementation pass used unit
      tests only, no live vLLM/GPU session. **Judgment call, closure pass:
      not treated as a blocker for archival.** Distinct from, but informed
      by, the investigation-phase live evidence in `design.md` (48-request
      4x-ceiling burst, zero rejections; 90-request sustained-arrival run,
      no starvation signature; a fourth-attempt cancellation-while-queued
      success) — that evidence exercised a hand-built harness reproducing
      Option C's design, not the literal shipped `admit()`, so it does not
      by itself close this item. The judgment for treating it as non-blocking
      instead rests on: (a) the implementation is a direct, narrow
      translation of that already-proven design onto stdlib
      `asyncio.BoundedSemaphore`/`asyncio.wait_for`, with no new logic of its
      own; (b) 4.1's unit suite includes 15 `test_admission.py` cases plus 2
      route-level 400/429 integration tests exercising the actual shipped
      `admit()`/timeout/`EngineSaturatedError` path directly, not a
      hand-built stand-in; (c) this repository's B6 closure pass separately
      live-verified real concurrent `/v1/chat/completions` traffic through
      this exact admission-gated path without incident, though not as a
      dedicated max_num_seqs-ceiling burst. Recorded as a genuine, open
      evidence gap — same tier as 4.3a, not silently resolved.
- [ ] 4.3a Live smoke test against a larger/slower model than `opt-125m`
      (recommended follow-up, not a blocker — see `design.md` →
      "Acquisition, rejection, and timeout semantics") to inform the
      production `admission_wait_s` default; this investigation's
      structural finding (1.13a) does not by itself justify a specific
      wait-time number for larger models. **Not run this pass.**

## 5. Documentation

- [x] 5.1 New `docs/DECISIONS.md` entry recording which option was chosen and
      why (mirrors DEC-040's own precedent of documenting this exact tradeoff)
      — DEC-060.
- [x] 5.2 `docs/PHASE-A-ARCHITECTURE.md` §10 updated to mark B4 complete.

## 6. Final review

- [x] 6.1 `openspec validate rescope-admission-control --strict` passes with
      the implementation-phase spec delta included — "Change
      'rescope-admission-control' is valid".
- [x] 6.2 Archive via `openspec archive rescope-admission-control --yes`
