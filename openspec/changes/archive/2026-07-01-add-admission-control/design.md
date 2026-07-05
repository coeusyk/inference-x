# Design: add-admission-control

## Overview
`AdmissionController` lives under the existing `routing/` boundary (`routing/base.py`'s
own docstring already reserves this role: "the service layer calls the router before
dispatching to an engine"). It never touches the GPU — it only decides admit/clamp/reject
from numbers the engine already exposes (`kv_capacity_tokens`) or can compute cheaply
(tokenizer-based prompt count). All GPU sizing stays in `utils/vllm_pool_config.py` and
the engine, unchanged.

## Module structure

```
src/inference_x/routing/
└── admission.py
    ├── ContextTooLongError(ValueError)   # 400 via existing value_error_handler
    ├── EngineSaturatedError(Exception)   # 429 via new handler, has retry_after_s
    ├── AdmissionResult                    # effective_max_tokens, reserved_tokens
    ├── _KVReservationTracker              # thread-safe per-model token counter
    └── AdmissionController
        ├── admit(routed_model, request, engine) -> AdmissionResult
        └── release(routed_model, reserved_tokens) -> None

src/inference_x/schemas/chat.py
└── ChatCompletionRequest   # +max_context_tokens, +max_output_tokens, +priority

src/inference_x/engines/vllm_engine.py
└── VLLMEngine.count_prompt_tokens(request) -> int   # real tokenizer count, chars/4
                                                       # fallback on any failure

src/inference_x/services/chat_service.py
└── ChatService
    ├── __init__(..., admission: AdmissionController | None = None)
    │   # defaults to AdmissionController(registry) — no tier, 4096-token cap — so
    │   # every existing direct ChatService(...) construction (tests) keeps working
    ├── complete()         # admit() -> model_copy(max_tokens=effective) -> generate() ->
    │                      # release() in finally
    └── stream_response()  # same shape; release() happens in the generator's finally,
                            # before the [DONE] sentinel is yielded

src/inference_x/api/
├── deps.py     # _build_admission_controller(config_dir): resolves the VRAM tier via
│               # settings.get_vram_tier(), falls back to no-tier (4096 cap) with a
│               # warning on failure — same fail-open posture as tier logging elsewhere
├── errors.py   # engine_saturated_error_handler: 429, Retry-After header, message is
│               # NOT sanitized to a generic string (it names no internals — telling the
│               # client to retry is the whole point of a 429)
└── main.py     # registers EngineSaturatedError -> engine_saturated_error_handler
```

## Key decisions

**KV pricing is tracked in tokens, not GiB.** The original plan proposed generalizing
`estimate_kv_cache_gib` into a `kv_gib_for_tokens` helper and comparing GiB budgets.
Since `engine.kv_capacity_tokens` (real, post-load, `num_gpu_blocks * block_size`) is
already in tokens, converting a per-request need into GiB and back would add a unit
round-trip with no accuracy benefit. `_KVReservationTracker` reserves/releases raw token
counts (`prompt_tokens + effective_output`) against `capacity * 0.9`, directly.

**Reservations degrade to "don't block" when data is missing.** No tokenizer on the
engine → chars/4 fallback estimate (never blocks admission entirely). No reported
`kv_capacity_tokens` → the KV gate is skipped. This matches the rest of the codebase's
posture for advisory/best-effort signals (e.g. `initialize_app()` logs a warning and
continues when VRAM tier resolution fails, rather than refusing to start).

**`count_prompt_tokens` is not part of `BaseEngine`'s abstract contract.** Making it
abstract would force every test's stub `BaseEngine` subclass (7 files) to implement it
even though only `AdmissionController` needs it and only when a real controller is
wired in. `_estimate_prompt_tokens()` in `admission.py` uses `getattr(engine,
"count_prompt_tokens", None)` — the same pattern `api/routes/metrics.py` already uses
for `kv_capacity_tokens`. Net effect: zero test files needed modification for this to
work; only `VLLMEngine` (the one real implementation) gained the method.

**A "clamp" below a minimum is treated as a rejection.** `_MIN_CLAMPED_OUTPUT_TOKENS =
16`: clamping `max_tokens` down to, say, 2 tokens produces a response so short it isn't
useful — `admit()` raises instead of returning a near-empty budget, for both the
context gate and the KV-pressure gate.

## What was tried and reverted: non-streaming continuous batching

`_run_completion` originally called `.chat()`/`.generate()`, holding `step_lock` for the
*entire* blocking generation — serializing all non-streaming requests against one model,
unlike `generate_stream`, which releases the lock between individual `llm_engine.step()`
calls (verified in the original VRAM-aware plan by reading `vllm_engine.py`). Mirroring
that pattern for `_run_completion` (`add_request` once, then loop `step()` under the lock
until `finished=True` for this request's own `request_id`) passed unit tests (mocked
`llm_engine`) — and then failed live: **2 real concurrent non-streaming requests against
the dev GPU reproducibly lost one of them**, raising "vLLM produced no output".

Root cause: `generate_stream` tolerates a request's own step being "won" by another
concurrently-looping thread, because `output.outputs[0].text` is *cumulative* — whichever
future `step()` call happens to next surface your `request_id` lets you compute the
missed delta and catch up; no data is lost, only coalesced into a bigger chunk. A
non-streaming completion has no such recovery: it needs exactly one `finished=True`
`RequestOutput` for its own `request_id`. If a *different* thread's `step()` call is the
one that returns that terminal output, it's discarded (request_id mismatch is the whole
point of each thread only claiming its own), and `llm_engine.has_unfinished_requests()`
can go globally `False` (once every other concurrent request finishes) before the owning
thread ever gets to see it — so its own `while has_unfinished_requests():` loop exits
having captured nothing.

A correct fix needs a single shared per-engine "driver" thread that continuously calls
`step()` and dispatches each `RequestOutput` to a per-`request_id` queue that every
in-flight caller (streaming or non-streaming) reads from — not N independent
request-owned loops each racing to be the one whose `step()` call surfaces their own
output. That is a materially larger change (a persistent background thread per engine,
lifecycle-tied to engine startup/shutdown, plus a registration/dispatch table) than the
original plan's "low-risk, self-contained" framing suggested, and was **not** attempted
in this change — `_run_completion` was reverted to the original `.chat()`/`.generate()`
call. See DEC-038 for the full incident writeup.
