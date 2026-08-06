# Tasks: expose-per-request-engine-timing

## 1. Verification (complete — see design.md)
- [x] 1.1 Verified `RequestOutput.metrics: RequestStateStats | None` is the
      per-request timing source, and that it is populated (not `None`) for
      InferenceX's current configuration (`disable_log_stats` never set,
      defaults `False`) — confirmed by live `AsyncLLM.generate()` execution
      against `Qwen/Qwen2.5-0.5B-Instruct`, not source reading alone.
- [x] 1.2 Verified the clock-domain split: `queued_ts`/`scheduled_ts`/
      `first_token_ts`/`last_token_ts` are engine-core-process
      `time.monotonic()`; `arrival_time` is frontend-process `time.time()`.
      Traced each to its exact assignment site.
- [x] 1.3 Verified the four derived-interval formulas against vLLM's own
      `output_processor.py: do_tracing()`, not invented independently.
- [x] 1.4 Determined and recorded which fields are excluded and why
      (design.md §5) — `arrival_time`, `first_token_latency`,
      `num_generation_tokens`, `is_corrupted`, and a non-existent second
      "scheduler delay" field.

## 2. `EngineTiming` schema
- [x] 2.1 Add `EngineTiming` to `src/inference_x/schemas/chat.py`:
      `queue_time_ms`, `prefill_time_ms`, `decode_time_ms`,
      `inference_time_ms` (all `float`), following the existing `_ms`
      convention from `MetricsResponse`.
- [x] 2.2 Add `timing: Optional[EngineTiming] = None` to
      `ChatCompletionResponse`.
- [x] 2.3 Add `timing: Optional[EngineTiming] = None` to `ChatStreamChunk`;
      update its docstring (currently: "Per-request timings are Phase B3
      and are not carried here") to describe the now-carried field and
      that it is set only on the terminal event, mirroring `usage`.

## 3. Derivation
- [x] 3.1 Widen `derive_terminal_metadata()` in
      `src/inference_x/engines/vllm_engine.py` to also derive
      `EngineTiming | None` from `output.metrics`, using
      `getattr(..., default=None)` per design.md Decision 5 — whole-block
      `None` on any missing/unreadable field, never a partial
      `EngineTiming`.
- [x] 3.2 Update the function's return type to
      `tuple[Literal["stop","length"], ChatCompletionUsage, EngineTiming | None]`
      and its one call site in `_stream_chunks`.
- [x] 3.3 Confirm both `generate()` and `generate_stream()` receive the
      widened tuple from the same call site — no second call, no
      duplicated derivation (DEC-050).
- [x] 3.4 Thread `timing` onto the wire in
      `ChatService.stream_response()` (`services/chat_service.py`), which
      hand-builds SSE JSON and does not serialize `ChatStreamChunk`
      directly. Discovered via live smoke test: the engine-level
      `ChatStreamChunk.timing` was populated correctly, but the wire-level
      usage event never read it. Bundled into the existing
      `include_usage`-gated usage event rather than adding a new flag
      (design.md Decision 6).

## 4. Tests
- [x] 4.1 Live-verified test: a real `generate()`/`generate_stream()` call
      against the pinned vLLM populates `timing` with
      `inference_time_ms == prefill_time_ms + decode_time_ms` within
      floating-point tolerance, and all four fields are non-negative.
- [x] 4.2 Non-streaming and streaming paths report identical `EngineTiming`
      for the same request (extends the existing DEC-050 usage-parity
      test).
- [x] 4.3 A mocked `RequestOutput` with `metrics=None` (or a missing
      field) asserts `timing is None` on the response — not a partial or
      zero-filled `EngineTiming`.
- [x] 4.4 Existing `usage`/`finish_reason` tests remain green, unmodified.
- [x] 4.5 `GET /metrics` and `GET /v1/metrics` tests remain green,
      unmodified — confirms no cross-talk with B2.

## 5. Docs
- [x] 5.1 Add a `CHANGELOG.md` `[Unreleased]` entry describing the new
      `timing` field.
- [x] 5.2 Mark B3 complete in `docs/PHASE-A-ARCHITECTURE.md` §10, following
      the pattern of the existing "B1 status: complete" / "B2 status:
      complete" writeups.

## 6. Validation
- [x] 6.1 `ruff check .` green.
- [x] 6.2 `mypy src/` green, with no new baseline entries beyond what the
      defensive `RequestStateStats` access (design.md Decision 5)
      requires.
- [x] 6.3 `pytest tests/unit` green.
- [x] 6.4 `openspec validate expose-per-request-engine-timing --strict`
      passes.
