# Tasks — OS-2 truthful token accounting

Implement in the order below. It is a dependency order, not a preference: each
file consumes the contract the previous one establishes. If a dependency proves
the sequence impossible, stop and report rather than reordering silently.

## 0. Preconditions

- [x] 0.1 Set `DEC-049` status from `proposed` to `accepted` in
  `docs/DECISIONS.md`. It is the authority for the Engine Boundary widening; the
  widening must not land ahead of it.
- [x] 0.2 Confirm OS-1's gate is green on the branch point (`pytest`, `ruff`,
  `mypy`) so later failures are attributable to this change.
- [x] 0.3 Re-read the proposal's Compatibility Invariants (C1–C14) and Intentional
  Behavioral Changes (B1–B8). They are the review checklist.

## 1. `schemas/chat.py`

- [x] 1.1 Add `ChatStreamChunk` with exactly three fields: `content: str`,
  `finish_reason: Literal["stop","length","error"] | None = None`,
  `usage: ChatCompletionUsage | None = None`. No timing fields (Phase B3).
- [x] 1.2 Add an optional `stream_options` object to `ChatCompletionRequest` with
  a single `include_usage: bool = False` field. Absent `stream_options` is
  equivalent to `include_usage: false` (R2).
- [x] 1.3 Append the new request field at the end of `ChatCompletionRequest`
  rather than inserting it mid-class, so OS-3's `seed` and OS-4's `strict` append
  cleanly behind it.
- [x] 1.4 Do not modify `ChatCompletionResponse` (C1, R4).

## 2. `engines/base.py`

- [x] 2.1 Widen `generate_stream` to `AsyncGenerator[ChatStreamChunk, None]`.
- [x] 2.2 Correct the class docstring. Its claim that adding a second engine "must
  not require changes here" is now demonstrably false; say what the contract
  actually guarantees and cite DEC-049.
- [x] 2.3 Do **not** declare `count_prompt_tokens` or any other capability method
  — OS-4 (acceptance criterion 11).

## 3. `engines/driver.py`

- [x] 3.1 Change the `submit_stream` queue element type to
  `ChatStreamChunk | BaseException | None` (R6). `None` remains the end-of-stream
  sentinel — do not replace it.
- [x] 3.2 In `_dispatch`: put a content-only `ChatStreamChunk` for each text
  delta, and on `output.finished` put a terminal `ChatStreamChunk` carrying
  `finish_reason` and `usage` **before** putting the existing `None` sentinel.
  Stop discarding the terminal `RequestOutput` metadata.
- [x] 3.3 Do **not** change the dead flag, `_dead_lock`, submission rejection after
  death, `_broadcast_exception`, `_fail_one`'s failure semantics, the thread
  restart policy, or the incremental-text delta logic (C7). This task widens a
  data channel; it does not restructure the driver.
- [x] 3.4 Do not begin any AsyncLLM work (Phase B1).

## 4. `engines/vllm_engine.py`

- [x] 4.1 Yield `ChatStreamChunk` from `generate_stream`.
- [x] 4.2 Build the terminal chunk's `finish_reason` and `usage` from the terminal
  `RequestOutput` using the **same derivation `generate()` already uses**, so
  acceptance criterion 2 holds by construction rather than by coincidence.
- [x] 4.3 Preserve the `_VLLM_AVAILABLE` false path and the `EngineDriverDeadError`
  to `RuntimeError` mapping, adapted to the new chunk type.
- [x] 4.4 Do not touch `_sampling_params` beyond what the chunk change requires —
  `seed` is OS-3.

## 5. `services/chat_service.py`

- [x] 5.1 Consume `ChatStreamChunk` in `stream_response`.
- [x] 5.2 Implement the event ordering in R3 exactly: content events with
  `"finish_reason": null`; then one terminal event with `"delta": {}` and a
  non-null finish reason; then, only when `include_usage` is true and the engine
  supplied usage, one event with `"choices": []` and `usage`; then `[DONE]`.
- [x] 5.3 Do not retroactively mutate the last content event to carry the finish
  reason — emit a separate terminal event (R3).
- [x] 5.4 Leave the timeout path's error event and its `[DONE]` unchanged (C6). No
  terminal or usage event on that path.
- [x] 5.5 Keep the `finally` block's `admission.release` and `gen.aclose()`
  semantics intact (C8).
- [x] 5.6 Do not add `warnings`, `resolved`, or `strict` handling — OS-4 (R4).

## 6. `observability/middleware.py`

- [x] 6.1 Delete `_count_sse_delta_tokens` entirely.
- [x] 6.2 Repoint `_wrap_and_record_sse` to read `usage.completion_tokens` and
  `usage.prompt_tokens` from the usage event when present.
- [x] 6.3 When no usage event is present, record **no** completion-token figure —
  not zero (acceptance criterion 4, B5). Confirm `tokens_per_sec` is likewise
  absent rather than zero.
- [x] 6.4 Update the module docstring: it currently documents the whitespace
  approximation and cites DEC-023 as the reason. Replace with the actual
  behaviour and cite DEC-049.
- [x] 6.5 Preserve the single-record-per-stream guarantee, the TTFT measurement,
  the client-disconnect handling, and the swallow-all-errors contract.
- [x] 6.6 Do not change the `/v1/metrics` response schema (C11).

## 7. `benchmarks/runner.py`

- [x] 7.1 In the streaming loop, read `usage.completion_tokens` from the usage
  event instead of accumulating `len(content.split())`.
- [x] 7.2 Set `stream_options.include_usage: true` on the runner's own requests,
  so the figure it needs is on the wire (R2).
- [x] 7.3 Keep TTFT measured from the first content event.
- [x] 7.4 Touch only the stream measurement path and the `PromptResult`
  construction it feeds. `_load_suite` belongs to OS-5; `peak_vram_delta_gb` and
  the advisor weights belong to OS-6.

## 8. Tests

- [x] 8.1 **Streaming contract test — authoritative.** Assert the complete ordered
  event sequence for three cases: default, `include_usage: true`, and timeout.
  Assert order and event count, not substring presence. Do not weaken this test to
  suit an implementation that is easier to write.
- [x] 8.2 Streamed/non-streamed agreement: same prompt and parameters, equal
  `completion_tokens`.
- [x] 8.3 Absence test: with `include_usage` false, assert the recorded metric's
  completion-token figure is absent — distinguish absent from zero.
- [x] 8.4 Update the 12 `generate_stream` stubs across six files:
  `test_chat_service.py` (2), `test_routes.py` (3), `test_observability.py` (3),
  `test_startup.py` (2), `test_engine_interface.py` (1), `test_engine_pool.py` (1).
- [x] 8.5 Update assertions only in `test_vllm_engine_stream.py` — it drives the
  real `VLLMEngine.generate_stream` and has no stub to change.
- [x] 8.6 Do **not** modify `test_streaming.py`'s existing assertions. It tests
  `playground/streaming.py`, which C4 keeps unmodified; it only gains the
  tolerance test in 8.7.
- [x] 8.7 Assert `playground/streaming.py` parses a stream containing terminal and
  usage events without modification (C4).
- [x] 8.8 `uv run pytest tests/unit/test_engine_driver.py --count=20` — all runs
  pass (C7).
- [x] 8.9 Full suite, `ruff check .`, `mypy src/` all green.
- [x] 8.10 If `services/chat_service` now type-checks, remove it from the DEC-048
  baseline. Never add to that baseline (C13).
  → Condition NOT met: 3 residual errors remain (`Coroutine ... has no attribute
  __anext__`), caused by `BaseEngine.generate_stream` being declared `async def`
  rather than `def` returning an AsyncGenerator. Fixing that is a second contract
  change DEC-049 does not authorize. Baseline left as-is — it did not grow.

## 9. Documentation

- [x] 9.1 Add the superseding ADR to `docs/DECISIONS.md`: previously reported
  streamed token counts, and every throughput figure derived from them, are not
  comparable with figures produced after this change. Claim the next free DEC
  number at PR time; register it in the milestone tracking issue.
- [x] 9.2 Do not edit `article-final.md` as part of the repository diff — it is
  gitignored under a deliberate policy comment (R5). The correction remains an
  author obligation, tracked as a non-diff action alongside OS-1's
  branch-protection precedent.
- [x] 9.3 `README.md` was checked and publishes no throughput figures; no
  correction is required there. Re-confirm before closing.
- [x] 9.4 Do not touch `docs/PHASES.md` (C14).

## 10. Self-review and archive

- [x] 10.1 Walk C1–C14. Every invariant holds.
- [x] 10.2 Walk B1–B8. Nothing outside the list changed.
- [x] 10.3 `git diff --stat` touches no file outside Repository Impact.
- [x] 10.4 `openspec validate --strict` passes.
- [x] 10.5 Archive once merged.

## Constraints

- Do not reopen DEC-047 or DEC-049.
- Do not introduce `inference_x/execution/`, a backend-neutral chunk type, a
  second backend, or optional-extra `vllm`.
- Do not declare capability methods on `BaseEngine` (OS-4).
- Do not add `seed` (OS-3), `strict`/`warnings`/`resolved` (OS-4), computed
  `suite_version` (OS-5), or advisor changes (OS-6).
- Do not touch `api/deps.py`, `engines/registry.py`, `routing/`, or `playground/`.
- Do not restructure `engines/driver.py` beyond widening its stream channel.
- Do not add timing fields to `ChatStreamChunk` (Phase B3).
- Do not grow the DEC-048 mypy baseline.
- Do not reformat or opportunistically clean any file touched.
