## Why

`VLLMEngine._run_completion` (non-streaming) holds a lock across one blocking
`.chat()`/`.generate()` call — safe, but fully serialized: N non-streaming requests
against one model run one at a time, unlike `generate_stream`, which already releases
its lock between individual `llm_engine.step()` calls and gets real continuous-batching
concurrency.

DEC-038 attempted to close this gap by mirroring the streaming loop inside
`_run_completion` (add_request once, then loop `step()` under the lock until this
request's own `finished=True` output appears). It passed unit tests against a mocked
engine, then failed live: 2 real concurrent non-streaming requests reproducibly lost
one of them ("vLLM produced no output"). Root cause: `generate_stream` tolerates its
own step being "won" by another thread because `output.outputs[0].text` is cumulative —
any later call surfacing your `request_id` lets you compute the missed delta. A
one-shot `finished=True` handoff has no such recovery: if a different thread's `step()`
call is the one that returns your request's terminal output, that thread discards it
(request_id mismatch), and `has_unfinished_requests()` can go globally `False` before
the owning thread ever sees its own result. DEC-038 reverted the change and named the
correct fix: a single shared per-engine thread that is the only caller of
`add_request`/`step()`, demultiplexing every output to the right destination by
`request_id` — eliminating the race by construction instead of working around it.

This change implements that driver thread.

## What Changes

- New `EngineDriver`: one per `VLLMEngine` instance, owns `llm_engine` exclusively.
  Callers submit a request (prompt + `SamplingParams`) and get back a channel — a
  `queue.Queue` for streaming, a `concurrent.futures.Future` for non-streaming. The
  driver thread is the sole caller of `add_request`/`step()`; it dispatches each
  `step()` output to the right channel by `request_id` and never lets a caller's own
  thread touch the engine directly.
- `_run_completion` and `generate_stream`'s worker are rewritten to submit through the
  driver instead of running their own `add_request`/`step()` loop. Both become thin
  consumers of the same step loop — streaming and non-streaming are unified at the
  driver boundary, differing only in whether their channel yields incremental chunks
  or one terminal result.
- Driver failure (an exception from `step()`) is broadcast to every pending
  request's channel and flips the engine unhealthy, instead of silently hanging or
  losing output.
- `BaseEngine`'s abstract contract (`generate`, `generate_stream`, `is_healthy`) is
  unchanged — this is entirely internal to `VLLMEngine`.

## Capabilities

### New Capabilities
(none — this modifies existing engine-dispatch behavior, not a new capability surface)

### Modified Capabilities
- `platform`: non-streaming chat completions gain the same continuous-batching
  concurrency streaming already has, and driver failures now surface as explicit
  errors/unhealthy state instead of silent hangs or dropped output.

## Impact

- `src/inference_x/engines/vllm_engine.py`: `_run_completion`, `generate_stream`,
  `__init__`, `shutdown` all change to go through the new driver.
- New `src/inference_x/engines/driver.py`.
- No change to `engines/base.py`, `services/chat_service.py`, or any request/response
  schema — purely an internal engine-layer change.
- Existing `_POOL_STEP_LOCK` (serializing `step()` across engines sharing one process
  when `pool_size > 1`) is preserved, now acquired by the driver thread instead of by
  each request's own thread.
