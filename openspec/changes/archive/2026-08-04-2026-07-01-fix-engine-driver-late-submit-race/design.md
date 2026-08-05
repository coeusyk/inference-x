## Context

`EngineDriver._run()` (`engines/driver.py`) loops: drain pending submissions from
`_submit_q` into `_pending`, call `step()`, dispatch outputs by request_id. On a
`step()` exception it calls `_broadcast_exception(exc)`, which sets `self._dead =
True`, fails everything currently in `_pending`, drains and fails anything still
sitting in `_submit_q`, then returns — ending the thread for good (no restart).

`submit_stream`/`submit_complete` each do `if self._dead: raise
EngineDriverDeadError(...)` before putting onto `_submit_q`. That check and the
`_submit_q.put()` are two separate, unsynchronized steps.

## The Race

1. Thread A calls `submit_complete()`. It reads `self._dead` — still `False`. It is
   preempted before calling `_submit_q.put(...)`.
2. The driver thread's `step()` raises. `_broadcast_exception` runs: sets `_dead =
   True`, fails everything in `_pending`, drains `_submit_q` (empty — Thread A
   hasn't put its item yet), and the driver thread returns. Nothing will ever call
   `_drain_submissions()` or `step()` again.
3. Thread A resumes and puts its request onto `_submit_q`. The item sits there
   forever: no thread is left to read it, dispatch it, or fail it. The caller's
   `future.result(timeout=300.0)` (or stream `queue.get()`) blocks for the full
   timeout, or forever, instead of receiving the engine's actual failure.

The window is between `self._dead` being read as `False` in step 1 and the
corresponding `_submit_q.put()` landing before `_broadcast_exception`'s drain in
step 2 — reproduced ~40% of the time in
`test_step_exception_is_broadcast_to_pending_completion_futures` under repeated
runs, confirmed present on unmodified `develop` HEAD (verified via `git stash`),
independent of any other change.

## The Fix

Add one `threading.Lock` (`self._dead_lock`) guarding exactly three things:
`self._dead`, `self._dead_exception`, and the decision of whether a given
`_submit_q.put()` is allowed to happen at all.

`_broadcast_exception` becomes: acquire `_dead_lock`; set `_dead = True` and store
the exception; drain `_submit_q` (failing anything found) *while still holding the
lock*; release; then fail everything already in `_pending` (broadcasting itself
does not need the lock — `_pending` is only ever touched by the driver thread).

`submit_stream`/`submit_complete` become: acquire `_dead_lock`; if `_dead`, raise
`EngineDriverDeadError` immediately (still holding the lock, so no half-enqueued
state is possible) and never touch `_submit_q`; otherwise put the request onto
`_submit_q` *before releasing the lock*, then release.

**Why the queue-drain and the enqueue must share the same lock, not just the flag
check:** a design where `submit()` checks `dead` under a lock, releases it, and
*then* separately calls `_submit_q.put()` reopens exactly the same window one level
up — `_broadcast_exception` could run its own lock-protected set-and-drain in the
gap between submit's release and its put, and the put would again be orphaned.
Making "check dead, and if not dead, enqueue" one atomic critical section (and
"set dead, and drain whatever is currently enqueued" the other) means the two
critical sections cannot interleave: either the enqueue fully happens before the
drain (and gets caught by it), or fully after (and `submit()` never got past the
`dead` check to enqueue in the first place). There is no third case.

**Why the flag and the exception must be set together, under the lock:** a caller
that observes `_dead = True` must always be able to read the *matching* exception
right after — setting them as two separate unguarded writes (or setting `_dead`
first and `_dead_exception` after) leaves a window where a concurrent `submit()`
sees `dead = True` but `_dead_exception` still `None`.

**Lock scope excludes `step()`:** the lock is only ever held across flag
read/write and queue drain/put — both fast, non-blocking operations. It is never
held across `self._llm_engine.step()`, which can block for the duration of a
forward pass; holding it there would serialize submission against every step()
call for no reason and reintroduce a different kind of stall.

## Non-Goals

- No change to the driver thread's restart policy — a dead driver stays dead, same
  as today; this only fixes what happens to requests submitted during the
  transition into that state.
- No change to the streaming path's chunking/delta logic — `submit_stream` gets the
  same atomic dead-check as `submit_complete`, nothing about `_dispatch` changes.
- No change to any public `EngineDriver` method signature, `VLLMEngine`, or any
  other module.

## Risks / Trade-offs

- The fix adds one lock acquisition per `submit()` call (in addition to the
  existing `_step_lock` acquisition per `step()` call, which is unaffected). Both
  are held for O(microseconds); no measurable throughput impact expected given the
  existing per-request overhead of engine dispatch.
- The existing test's sleep-based polling loop for `driver.is_dead` (a courtesy
  wait, not part of the race itself) is replaced with explicit synchronization
  (an `Event` or `Barrier`) to make the race window itself deterministic — this
  changes test internals only, not driver behavior.
