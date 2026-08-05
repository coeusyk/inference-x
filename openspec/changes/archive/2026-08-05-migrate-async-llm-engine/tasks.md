# Tasks: migrate-async-llm-engine

## 1. Pre-implementation verification — COMPLETE

- [x] 1.1 VERIFIED: installed vLLM version is `0.22.1` (re-confirmed at
      verification time via `uv run python -c "import vllm; print(vllm.__version__)"`).
- [x] 1.2 VERIFIED: `AsyncLLM.generate()`'s `except (asyncio.CancelledError,
      GeneratorExit)` → `abort()` → `engine_core.abort_requests_async(...)`
      path exists exactly as designed (Decision 4). Confirmed by direct read
      of `async_llm.py`'s `generate()` and `abort()` bodies.
- [x] 1.3 VERIFIED: `AsyncLLM.errored` / `dead_error` / `check_health()` exist
      with the semantics Decision 5 assumes. Confirmed by direct read of
      `async_llm.py:1044-1059,900-903`.
- [x] 1.4 VERIFIED: a request submitted after the engine is already dead is
      **rejected immediately**, not silently accepted or orphaned — via two
      independent synchronous guards (`AsyncLLM.add_request`'s own
      `if self.errored: raise EngineDeadError()` at `async_llm.py:300-301`,
      and the transport client's `ensure_alive()` at `core_client.py:658-660`,
      reached via `add_request_async` → `_send_input_message`). Full evidence
      chain recorded in design.md "Task 1.4 — verified conclusion". The spec
      delta's implementation-neutral MODIFIED requirement is confirmed
      satisfied, not merely assumed.

## 2. Engine implementation

- [x] 2.1 Add `AsyncLLM` construction to `VLLMEngine.__init__` (replacing
      `LLM(**kwargs)` + `EngineDriver(...)`); preserve existing kwargs mapping
      (model path, tensor-parallel size, seed defaults, etc.) — no config
      surface change.
- [x] 2.2 Relocate `derive_terminal_metadata` from `driver.py` into
      `vllm_engine.py` verbatim (Decision 8) — signature and body unchanged.
- [x] 2.3 Rewrite `generate_stream()` to iterate `AsyncLLM.generate(...)`
      directly, translating each yielded `RequestOutput` into a
      `ChatStreamChunk` using the same delta-computation logic
      `EngineDriver._dispatch` used (cumulative-text diffing), terminating
      with `derive_terminal_metadata`'s output.
- [x] 2.4 Rewrite `generate()` to derive from `generate_stream()`
      (Decision 2) — no second, independent `AsyncLLM.generate()` call site.
- [x] 2.5 Rewrite `is_healthy()` to `not self._llm.errored` (Decision 5).
- [x] 2.6 Rewrite `_log_kv_cache_stats()` to read
      `self._llm.vllm_config.cache_config` instead of
      `self._llm.llm_engine.vllm_config.cache_config` (Decision 6);
      keep the existing `getattr`-defensive style since this remains
      non-public vLLM internals.
- [x] 2.7 Add the timeout wrapper (`asyncio.wait_for`-based or equivalent)
      around `generate_stream()`/`generate()` consumption, replacing
      `future.result(timeout=_COMPLETION_TIMEOUT_S)`. Preserve the existing
      `_COMPLETION_TIMEOUT_S` policy/duration — this migration changes
      cancellation mechanics only, not timeout duration. Confirm the wrapper
      now triggers real engine-side abort on timeout (Decision 7), not just a
      local give-up.
- [x] 2.8 Delete `_POOL_STEP_LOCK` (`vllm_engine.py:35`) and its
      `pool_size > 1` branch that selected it (Decision 1) — do not replace
      with a new lock; `pool_size` construction otherwise unchanged
      (Decision 3).
- [x] 2.9 Delete `src/inference_x/engines/driver.py` in full.

## 3. Config and baseline

- [x] 3.1 Remove the `inference_x.engines.driver` entry from
      `pyproject.toml`'s mypy baseline (module deleted).
- [x] 3.2 Attempt `mypy src/` with no new suppression for
      `inference_x.engines.vllm_engine`'s `generate`/`generate_stream`
      override. If the module still needs a baseline entry for unrelated
      vLLM-typing reasons, keep it minimal and re-annotate the reason (do not
      carry forward the old "10 — vLLM's LLM is untyped" comment verbatim if
      the cause has changed).

## 4. Tests

- [x] 4.1 Delete `tests/unit/test_engine_driver.py`.
- [x] 4.2 Add new tests against the AsyncLLM-backed `VLLMEngine` reproducing
      the same properties (not the same mechanism):
      - two concurrent completions return distinct, correct output
      - stream yields incremental deltas
      - two concurrent streams do not cross-contaminate
      - an engine-level exception is observable to a pending
        completion/stream (whatever `AsyncLLM`'s equivalent to "broadcast to
        all pending" is — likely per-request via `EngineDeadError`, not a
        broadcast, since `AsyncLLM` has no shared `_pending` dict; confirm and
        adjust the test's shape rather than forcing a broadcast assertion
        that may not apply)
      - shutdown stops cleanly
      - **new**: mid-stream disconnect results in an observable engine-side
        abort call (Decision 4's compatibility invariant, previously
        untested because it was previously untrue)
      - **new**: submission after engine death raises immediately — asserted
        unconditionally (Task 1.4 verified this true; no longer contingent).
- [x] 4.3 Confirm `tests/unit/test_chat_service.py::TestReservationLifecycle`
      (9 tests) passes unmodified — zero edits to that test file.

## 5. Validation

- [x] 5.1 `ruff check .`, `mypy src/`, `pytest tests/unit` all green.
- [x] 5.2 Manual smoke: `make chat` and `make playground` (or their current
      equivalents) against a real model, confirming streaming and
      non-streaming responses are unchanged from a user perspective.
- [x] 5.3 Confirm no edits landed in `services/`, `routing/`, or `api/` —
      grep the diff for those paths; a hit is a scope break (Compatibility
      Invariant 2).

## 6. Documentation

- [x] 6.1 Update `docs/PHASE-A-ARCHITECTURE.md` §10 (or its Phase B
      successor doc) to mark B1 complete and record the decisions above.
- [x] 6.2 Add a `docs/DECISIONS.md` entry (next DEC number) recording
      Decisions 1–8 above at ADR granularity — this migration changes enough
      real behavior (Decision 4, Decision 7) that it needs its own DEC, not
      just an OpenSpec record.
- [x] 6.3 Update the user-facing changelog / release notes to name the
      cancellation and timeout behavioral improvements explicitly (Decision 4,
      Decision 7) — these are real, observable improvements and should not be
      buried as an internal refactor.

## Constraints

- No implementation work begins until this OpenSpec is reviewed and accepted.
- No edits to `src/inference_x/api/`, `src/inference_x/services/`, or
  `src/inference_x/routing/` under any task above — if a task appears to
  require one, stop and flag it rather than silently expanding scope.
- Task 1.4 is resolved (verified: post-terminal-failure submissions are
  rejected immediately) — Task 4.2's regression test is written as an
  unconditional assertion, not a contingent one.
