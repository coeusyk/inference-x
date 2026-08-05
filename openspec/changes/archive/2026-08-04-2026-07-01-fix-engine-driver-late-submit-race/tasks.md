## 1. Driver locking

> Note on scope: the lock covers not just the dead-flag check but the full
> "check dead, and if not dead, enqueue" critical section in `submit()`, and the
> full "set dead, and drain whatever is currently enqueued" critical section in
> `_broadcast_exception`. Locking only the flag check (release, then enqueue
> separately) would reopen the same race one level up — see design.md's "Why the
> queue-drain and the enqueue must share the same lock" section.

- [x] 1.1 Add `self._dead_lock = threading.Lock()` and `self._dead_exception:
  BaseException | None = None` to `EngineDriver.__init__`.
- [x] 1.2 In `_broadcast_exception`: acquire `_dead_lock`, set `_dead = True` and
  `_dead_exception = exc`, drain `_submit_q` (failing anything found via
  `_fail_one`) while still holding the lock, then release. Broadcasting to
  `_pending` stays outside the lock (only ever touched by the driver thread).
- [x] 1.3 In `submit_stream`/`submit_complete` (via a shared `_submit` helper):
  acquire `_dead_lock`; if `_dead`, raise `EngineDriverDeadError` (chaining
  `_dead_exception` as the cause) before releasing the lock and without touching
  `_submit_q`; otherwise put the request onto `_submit_q` before releasing.

## 2. Tests

> Note on 2.1: implemented with a registration-count gate
> (`fake.fail_after_registered`) rather than a `threading.Event` the fake
> engine's `step()` blocks on. An event that `step()` waits on before raising
> risks a self-deadlock here: the driver thread only drains `_submit_q` between
> `step()` calls, so if `step()` blocks *inside* a call waiting for a second
> submission to land, that submission can never be drained while `step()` is
> still blocked. Gating on a plain count check inside `step()` (return an empty,
> legitimate "no output yet" result until the count is reached, otherwise raise)
> lets the driver's own loop naturally re-enter `_drain_submissions()` between
> calls, which is deterministic without any blocking wait and carries no
> deadlock risk. The outcome (both requests guaranteed registered before the
> engine fails) matches what 2.1 asks for.

- [x] 2.1 Rewrite `test_step_exception_is_broadcast_to_pending_completion_futures`
  to force the race window deterministically instead of relying on sleep-based
  timing or scheduling luck.
- [x] 2.2 Add a new test: after the driver is confirmed dead (`driver.is_dead is
  True`), a subsequent `submit_complete()`/`submit_stream()` call raises
  `EngineDriverDeadError` immediately (assert wall-clock time is well under the
  300s completion timeout), not after a timeout.

## 3. Validation

- [x] 3.1 Run `uv run pytest tests/unit/test_engine_driver.py -v --count=20`
  (install `pytest-repeat` if not already a dependency) — all 20 runs of every
  test in the file must pass, including the rewritten race test.
- [x] 3.2 Run `uv run pytest tests/unit/ -v` — full suite passes.

## 4. Documentation

- [x] 4.1 Add DEC-043 to `docs/DECISIONS.md`: the race, the atomic dead-flag +
  stored-exception + queue-drain lock fix, and why the lock must cover the
  enqueue decision itself, not just the flag read.
- [x] 4.2 Update `docs/PHASES.md` Phase 12 entry (or add a note) that this driver
  race fix was applied as a follow-up, discovered while validating the Phase 12
  default-model-resolution change.

## Constraints

- Do not change the driver thread's restart policy — a dead driver stays dead.
- Do not change the streaming path's chunking/delta logic in `_dispatch`.
- Do not change any public `EngineDriver` method signature or any other module
  (`vllm_engine.py`, `api/deps.py`, etc. are untouched).
- Do not hold `_dead_lock` across `self._llm_engine.step()` — only across
  flag read/write and queue drain/put.
