# Proposal: migrate-async-llm-engine

## Why

`VLLMEngine` currently wraps vLLM's offline `LLM` class and drives it with a
hand-rolled thread (`EngineDriver`, `engines/driver.py`) that calls
`add_request`/`step()` directly. That driver exists to fix a real bug
(DEC-038/039/043: concurrent requests lost output under vLLM V1's synchronous
step loop) but its cost is now well documented
(`docs/REVIEW-2026-08-03-architecture.md` §3.1,
`docs/REVIEW-2026-08-04-phase-a-final-audit.md` OWN-B5): ~226 LOC reimplementing
what vLLM's own async engine (`AsyncLLM`) already provides — real cancellation,
health signaling, and per-request async delivery — plus a race class
(DEC-038/039/043) that only exists because of the reimplementation.

`docs/PHASE-A-ARCHITECTURE.md` §10 names this migration B1 and scopes it
precisely: replace `LLM` + `EngineDriver` with `AsyncLLM` inside `VLLMEngine`,
behind the unchanged `BaseEngine` contract. It explicitly excludes splitting
multi-model serving into separate processes (that is B6) and excludes
admission/KV-gate rescoping (that is B4).

This change was authored after two review passes, delivered as chat output
during the Phase B kickoff session (not persisted as separate `docs/` files):
an architecture review reconciling Phase A authorities (DEC-047, DEC-049–057,
`PHASE-A-ARCHITECTURE.md`) against the deleted `EngineDriver`'s assumptions,
and an implementation-facts verification against the installed AsyncLLM
(vLLM 0.22.1, `vllm/v1/engine/async_llm.py`, read directly, cited by line
number below and in design.md). The facts and freeze from those reviews are
carried forward into design.md rather than re-cited as external documents.

## What Changes

- Delete `src/inference_x/engines/driver.py` (`EngineDriver`,
  `EngineDriverDeadError`, `derive_terminal_metadata`) and `_POOL_STEP_LOCK`
  (`vllm_engine.py:35`) — both exist only to serialize `EngineDriver`'s
  synchronous `step()` calls, a mechanism `AsyncLLM` does not use (each
  `AsyncLLM` instance owns its own background engine-core process; see design.md
  Decision 1).
- Rewrite `VLLMEngine` (`src/inference_x/engines/vllm_engine.py`) to construct
  and drive `vllm.v1.engine.async_llm.AsyncLLM` instead of `vllm.LLM` +
  `EngineDriver`.
- Relocate `derive_terminal_metadata` out of `driver.py` into `vllm_engine.py`
  (moved, not reimplemented — OWN-B5 / DEC-050 §3 requires the streamed and
  non-streamed paths keep deriving usage from the one function).
- `VLLMEngine.generate()` (non-streaming) becomes a thin consumer of
  `generate_stream()` — it is derived from `generate_stream()`, not a second,
  independent call into `AsyncLLM.generate()` (design.md Decision 2; see also
  the recorded compatibility invariant below).
- `VLLMEngine.is_healthy()` switches from `driver.is_dead` to
  `not self._llm.errored` (AsyncLLM's own health property).
- No change to `BaseEngine`, `ChatService`, `AdmissionController`, or any route
  handler — the Engine Boundary (DEC-047) is the whole point of this migration
  landing without touching them.

## Capabilities

### New Capabilities
None. This change replaces an implementation behind an existing contract; it
does not add a new capability to the platform spec.

### Modified Capabilities
- **platform**: engine execution backend (`VLLMEngine`'s internal driving
  mechanism), engine health signaling, engine cancellation semantics, and the
  `pool_size > 1` compatibility wording. See `specs/platform/spec.md` for the
  exact requirement deltas.

## Impact

- **Affected code**: `src/inference_x/engines/driver.py` (deleted),
  `src/inference_x/engines/vllm_engine.py` (rewritten internals, same public
  shape), `pyproject.toml` mypy baseline (the `inference_x.engines.driver`
  entry is removed because the module is deleted).
- **Unaffected code**: `src/inference_x/engines/base.py`,
  `src/inference_x/services/chat_service.py`,
  `src/inference_x/routing/admission.py`, `src/inference_x/api/*` — zero edits
  expected in any of these; a required edit to any of them is a scope break,
  not a refactoring opportunity.
- **Tests**: `tests/unit/test_engine_driver.py` is deleted (it tests a deleted
  class); its property-level assertions (concurrent completions don't
  cross-contaminate, streams don't cross-contaminate, exceptions broadcast to
  all pending callers, dead-engine submissions raise immediately if verified —
  see design.md's open item) are reproduced as new tests against the
  AsyncLLM-backed `VLLMEngine`. `tests/unit/test_chat_service.py`'s
  `TestReservationLifecycle` (9 tests) must pass unmodified.
- **Rollback**: single-PR revert. No persisted state, no wire-format change,
  no schema change — `BaseEngine`'s contract is unchanged, so reverting this
  change restores `EngineDriver` with zero side effects on callers.
