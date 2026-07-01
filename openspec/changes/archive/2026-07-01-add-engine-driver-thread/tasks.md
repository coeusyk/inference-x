## 1. EngineDriver module

- [x] 1.1 New `src/inference_x/engines/driver.py`: `_PendingRequest` dataclass
  (`mode`, `out_queue` or `future`, `previous_text`)
- [x] 1.2 `EngineDriver.__init__(llm_engine, step_lock)`: starts the daemon driver
  thread, private submission queue, private `_pending: dict[str, _PendingRequest]`
  (only ever touched on the driver thread)
- [x] 1.3 `submit_stream(prompt, sampling) -> queue.Queue[str | None]`
- [x] 1.4 `submit_complete(prompt, sampling) -> concurrent.futures.Future`
- [x] 1.5 Driver loop: drain submissions (call `add_request` for each), block on
  `submit_q.get(timeout=...)` when idle, `step()` under `step_lock` when requests
  are pending, dispatch each output to its registered channel by `request_id`
- [x] 1.6 Exception handling: `step()` raising broadcasts the exception to every
  pending channel, sets `_dead = True`, thread exits (no restart)
- [x] 1.7 `shutdown()`: signal the thread to stop, join with a timeout

## 2. VLLMEngine wiring

- [x] 2.1 `__init__`: construct one `EngineDriver` after `LLM(**kwargs)` succeeds,
  passing `_POOL_STEP_LOCK` when `pool_size > 1` else a private lock
- [x] 2.2 `_run_completion`: replace the blocking `.chat()`/`.generate()` call with
  `driver.submit_complete(...)` + `future.result(timeout=...)`
- [x] 2.3 `generate_stream`: replace the per-request worker thread's
  `add_request`/`step()` loop with `driver.submit_stream(...)`; keep the existing
  `asyncio.to_thread(sync_queue.get)` consumer loop essentially as-is
- [x] 2.4 `is_healthy()`: also `False` when the driver has flagged itself dead
- [x] 2.5 `shutdown()`: drain/stop the driver before releasing `self._llm`

## 3. Unit tests

- [x] 3.1 New `tests/unit/test_engine_driver.py` against a fake `llm_engine` (mocked
  `add_request`/`step`/`has_unfinished_requests`): single request completes, two
  concurrent requests both complete with correct, non-cross-contaminated content,
  streaming delta computation matches cumulative-text behavior, `step()` exception
  is broadcast to all pending channels, `shutdown()` stops the thread cleanly
- [x] 3.2 Update `tests/unit/test_vllm_engine.py` mocks to exercise the engine
  through the driver instead of asserting directly on lock acquisition order

## 4. Validation

- [x] 4.1 `uv run pytest tests/unit -v` — full suite passes
- [x] 4.2 Live smoke test: 2 concurrent non-streaming `POST /v1/chat/completions`
  against a running server, distinct prompts, both return complete non-truncated
  content matching their own prompt
- [x] 4.3 Live smoke test: streaming chat completions still work end-to-end,
  unaffected by the refactor

## 5. Docs and decisions

- [x] 5.1 Add DEC-039 to `docs/DECISIONS.md`: driver thread shipped, closes the
  DEC-038 follow-up
- [x] 5.2 Archive this change to `openspec/changes/archive/` when complete
