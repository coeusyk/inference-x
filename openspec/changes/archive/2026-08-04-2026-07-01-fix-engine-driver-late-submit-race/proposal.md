## Why

`EngineDriver` (see DEC-038/DEC-039, `engines/driver.py`) has one thread calling
`add_request`/`step()` for a given vLLM engine, and demultiplexes every output back
to the submitting caller by request_id. When `step()` raises, `_broadcast_exception`
marks the driver dead and fails every pending request — but there is a window
between that broadcast and the driver thread actually exiting during which a new
`submit()` call can still enqueue a request. That request is never picked up: the
thread has already returned from `_run()`, so nothing will ever drain the queue or
resolve its future/out_queue. The caller waits the full 300s completion timeout (or
forever, for a stream) instead of getting the engine's actual failure immediately.

`tests/unit/test_engine_driver.py::test_step_exception_is_broadcast_to_pending_completion_futures`
reproduces this ~40% of the time under repeated runs — confirmed present on
unmodified `develop` HEAD via `git stash`, independent of any other in-flight work.
A caller hitting this in production sees a hung request during exactly the moment
the engine is already known to be broken, which is worse than an immediate,
attributable error.

## What Changes

- `EngineDriver` gains a `threading.Lock`-protected dead flag and stored exception.
  `_broadcast_exception` sets both under the lock before broadcasting/draining.
- `submit_stream`/`submit_complete` check the dead flag under the same lock,
  immediately raising `EngineDriverDeadError` if the driver is already dead instead
  of enqueuing a request that will never be served.
- `test_step_exception_is_broadcast_to_pending_completion_futures` is made
  deterministic (no sleep-based timing) using explicit synchronization to force the
  race window reliably instead of relying on scheduling luck.
- A new test asserts `submit()` called after driver death raises immediately, not
  after a timeout.
- No changes to the public engine interface, the driver thread's restart policy, or
  the streaming path's chunking behavior.

## Capabilities

### New Capabilities
(none)

### Modified Capabilities
(none — this is a correctness fix to existing `EngineDriver` behavior, not a new
or changed capability. `specs/platform/spec.md` in this change adds one
requirement documenting the corrected dead-driver rejection behavior, since
`openspec validate` requires at least one delta for any change.)

## Impact

- `src/inference_x/engines/driver.py`: add a lock, dead flag, and stored exception;
  make submit-path dead checks atomic with the broadcast path.
- `tests/unit/test_engine_driver.py`: deterministic reproduction of the race; new
  post-death-submit test.
- No changes to `src/inference_x/engines/vllm_engine.py` or any other module.
