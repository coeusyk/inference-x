## Context

Today, in `vllm_engine.py`:
- `generate_stream` spawns a per-request worker thread that calls
  `llm_engine.add_request()` once (under `self._engine_lock`), then loops
  `llm_engine.step()` (under `step_lock`, which is `_POOL_STEP_LOCK` when
  `pool_size > 1`, else `self._engine_lock`), scanning every step's outputs for its
  own `request_id` and computing a text delta against `previous_text`. Safe under
  concurrency because a missed turn is just caught up on the next call that surfaces
  the request's own id.
- `_run_completion` (non-streaming) holds `step_lock` for the entire blocking
  `.chat()`/`.generate()` call — one request fully occupies the model before the
  next one starts. No race, but no concurrency either.
- DEC-038's reverted attempt made `_run_completion` mirror the streaming loop
  exactly, releasing the lock between `step()` calls. Live testing with 2 concurrent
  non-streaming requests reproducibly lost one: a terminal `RequestOutput` for
  request A can be returned by a `step()` call made on request B's thread, which
  discards it (it's only looking for its own id), and `has_unfinished_requests()`
  can flip `False` globally before request A's own thread calls `step()` again.

## Goals / Non-Goals

**Goals:**
- Give non-streaming completions the same continuous-batching concurrency streaming
  already has, without reintroducing the DEC-038 race.
- Unify streaming and non-streaming onto one internal step loop per engine, so there
  is exactly one place that ever calls `add_request`/`step()` on a given
  `llm_engine`.
- Make driver failure (an exception raised from `step()`, e.g. a mid-batch CUDA
  error) an explicit, observable failure for every affected in-flight request
  instead of a silent hang.

**Non-Goals:**
- No change to `BaseEngine`'s abstract contract, `ChatService`, or any request/
  response schema.
- No change to the `_POOL_STEP_LOCK` cross-engine serialization strategy for
  `pool_size > 1` — still required, just now acquired by the driver thread.
- No new engine knobs, no scheduler changes beyond what vLLM's `step()` already
  does — the driver only demultiplexes existing `step()` output, it does not
  re-implement or second-guess vLLM's internal batching decisions.

## Decisions

**One driver thread per engine, started in `VLLMEngine.__init__`, stopped in
`shutdown()`.** The driver is the only thread that ever calls `llm_engine
.add_request()` or `llm_engine.step()`. This is the structural fix: DEC-038's race
existed because *multiple* threads called `step()` and each only recognized its own
`request_id`, so a terminal output surfacing on the "wrong" thread's call was
discarded. With exactly one thread doing all `step()` calls and all dispatch, there
is no "wrong" thread to discard anything — every output is looked up in a
dict keyed by `request_id` and delivered to whichever channel is registered,
regardless of which logical request "caused" that particular `step()` to return it.

**Submission is queued, not synchronous.** Callers (the async `generate`/
`generate_stream` methods, invoked via `asyncio.to_thread`) call
`driver.submit_complete(prompt, sampling)` or `driver.submit_stream(prompt,
sampling)`, which enqueues a `(prompt, sampling, channel)` tuple and returns
immediately with the channel object (a `Future` or `Queue`) already constructed.
`add_request()` itself is called later, from the driver thread, when it drains the
submission queue — so `add_request` and `step()` are always interleaved by the same
single thread, never called concurrently with each other.

**Two channel types, one dispatch path.** `_PendingRequest.mode` is `"stream"` or
`"complete"`. Both are populated from the same per-output dispatch code
(`_dispatch(pending, output)`); only the terminal delivery differs — `"stream"`
pushes each delta chunk plus a final `None` sentinel onto a `queue.Queue`,
`"complete"` calls `future.set_result(output)` once, on `finished=True`. This is
what "unifies" the two paths: they are two thin adapters over one driver, not two
independent implementations.

**Idle wait uses a blocking queue get with a timeout, not a busy loop or a fixed
sleep.** When there are no pending requests, the driver thread blocks on
`self._submit_q.get(timeout=...)` instead of spinning (wastes CPU) or sleeping a
fixed interval (adds latency to a first request arriving while idle). When there
*are* pending requests, the driver drains any newly-arrived submissions
non-blockingly at the top of each loop iteration, then calls `step()`.

**Driver failure is broadcast, not swallowed.** If `llm_engine.step()` raises, the
driver:
1. Sets an exception on every currently-registered `_PendingRequest` (`future
   .set_exception(exc)` for completions, puts a sentinel error object followed by
   closing the queue for streams).
2. Sets an internal `_dead` flag, checked by `VLLMEngine.is_healthy()`.
3. Exits its loop — it does not try to recover or restart itself. A dead engine
   should be visible as unhealthy (existing `/health`/pool-health machinery already
   handles an unhealthy engine), not silently keep accepting new submissions into a
   thread that has already exited.

**`_POOL_STEP_LOCK` is preserved unchanged in meaning, only in a new caller.** The
comment at the top of `vllm_engine.py` ("Serialize `llm_engine.step()` across
engines in one process — vLLM V1 forward context") is a cross-*engine* constraint,
independent of this change's per-engine driver. `EngineDriver` takes a `step_lock`
argument at construction (still `_POOL_STEP_LOCK` when `pool_size > 1`, else a
private per-engine lock) and acquires it around each `step()` call, exactly as
`_run_completion`/`generate_stream`'s worker do today — only the caller of that
lock moves from "each request's own thread" to "the one driver thread".

## Risks / Trade-offs

- **A slow `step()` call now stalls streaming requests it previously wouldn't
  have.** Streaming used to run its own step loop per request; now all requests
  against one engine — streaming and non-streaming alike — share one thread's
  `step()` calls. A pathologically large prefill batch delays every in-flight
  request's next chunk, not just its own. Non-streaming is no worse off than
  today (it was already fully serialized); streaming's worst case changes from
  "delayed by other streaming requests" to "delayed by other streaming *and*
  non-streaming requests" — a strict superset of contention it already tolerated.
- **Idle-wait tuning matters.** Too short a `submit_q.get(timeout=...)` value
  becomes a disguised busy loop; too long delays picking up a new submission that
  arrives while the driver is blocked. Needs a short, bounded timeout (recommend
  sub-100ms) purely so the loop can also check a shutdown flag promptly — it does
  not gate correctness, only shutdown responsiveness and idle CPU use.
- **`_POOL_STEP_LOCK` is easy to forget on a future refactor.** If someone later
  gives each `EngineDriver` its own private lock even when `pool_size > 1`, the
  original multi-engine race the lock was added to prevent would silently return.
  This is called out explicitly in code comments at the lock's definition and in
  `EngineDriver.__init__`'s docstring, not just here.
