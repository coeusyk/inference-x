# Tasks: add-admission-control

## 1. Request schema
- 1.1 Add `max_context_tokens`, `max_output_tokens`, `priority` to
  `ChatCompletionRequest` (`schemas/chat.py`) — additive, OpenAI-compatible

## 2. AdmissionController
- 2.1 Add `routing/admission.py`: `ContextTooLongError`, `EngineSaturatedError`,
  `AdmissionResult`, `_KVReservationTracker`, `AdmissionController`
- 2.2 Context-length gate: prompt tokens (via `getattr(engine, "count_prompt_tokens",
  None)`, chars/4 fallback) vs. `min(ModelEntry.max_model_len, tier.max_model_len_cap,
  request.max_context_tokens)`
- 2.3 KV-pressure gate: in-flight reservation tracker vs. `engine.kv_capacity_tokens *
  0.9` (skipped when capacity is unreported)
- 2.4 `priority` semantics: interactive clamps to fit (rejects if clamped room <
  `_MIN_CLAMPED_OUTPUT_TOKENS`); batch always rejects instead of clamping
- 2.5 Unit tests (`test_admission.py`): context ceiling composition, prompt-too-long,
  clamp vs. reject by priority, KV saturation reject/clamp, reservation release,
  per-model isolation, tokenizer-less fallback — 15 tests

## 3. Engine tokenizer count
- 3.1 Add `VLLMEngine.count_prompt_tokens(request)` — real tokenizer via
  `_stream_prompt()` + `get_tokenizer().encode()`, chars/4 fallback on any exception
- 3.2 Unit tests: tokenizer path, fallback path

## 4. Error mapping
- 4.1 Add `engine_saturated_error_handler` (429 + `Retry-After`) to `api/errors.py`
- 4.2 Register `EngineSaturatedError` handler in `api/main.py`
- 4.3 `ContextTooLongError` needs no new registration (subclasses `ValueError`)

## 5. Service wiring
- 5.1 `ChatService.__init__` gains optional `admission: AdmissionController | None`,
  defaulting to `AdmissionController(registry)` (no tier) so existing direct
  `ChatService(...)` test construction is unaffected
- 5.2 `complete()`/`stream_response()`: `admit()` → build effective request via
  `request.model_copy(update={"max_tokens": ...})` → dispatch → `release()` in `finally`
- 5.3 Remove `ChatService._enforce_max_tokens_cap`/`_GLOBAL_MAX_TOKENS` (superseded)
- 5.4 `api/deps.py`: `_build_admission_controller(config_dir)` resolves the VRAM tier
  (fail-open with a warning on failure), wired into `get_chat_service()`; cache-cleared
  in `shutdown_app()`
- 5.5 Route-level integration tests (`test_routes.py`): prompt over `max_context_tokens`
  → 400; saturated batch-tier request → 429 with `Retry-After`

## 6. Non-streaming batching fix (attempted, reverted — see DEC-038)
- 6.1 Refactor `_run_completion` to drive `add_request`/`step()` like `generate_stream`
- 6.2 Unit tests against a mocked `llm_engine` — passed
- 6.3 Live 2-concurrent-request smoke test — **failed** (one request's output lost)
- 6.4 Revert to the original blocking `.chat()`/`.generate()` call; keep the incident
  writeup in DEC-038 and `design.md` for whoever attempts the shared-driver-thread fix

## 7. Validation
- 7.1 `uv run pytest tests/unit -v` — 371/371 pass
- 7.2 Live smoke: prompt over `max_context_tokens` → 400 against a running server
- 7.3 Live smoke: 4 concurrent non-streaming requests all return 200 (regression check
  after reverting task 6 — confirms no request is silently dropped)
- 7.4 Live smoke: streaming chat completions unaffected

## 8. Docs and decisions
- 8.1 Add DEC-038 to `docs/DECISIONS.md` (admission control accepted; batching fix
  reverted-with-findings)
- 8.2 Add Phase 8 to `docs/PHASES.md` with exit criteria and explicit "not delivered"
  notes
- 8.3 Archive this change to `openspec/changes/archive/` when complete
