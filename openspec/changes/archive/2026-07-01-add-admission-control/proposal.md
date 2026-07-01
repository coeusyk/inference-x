# Proposal: add-admission-control

## Summary
Enforce context-length and KV-pool limits before a chat completion request reaches the
vLLM engine, via a new `AdmissionController` that runs after routing and before dispatch.
Reject or clamp oversized/over-capacity requests with an actionable error instead of
letting them fail unpredictably (or silently truncate) inside the engine.

## Why
Before this change, the only pre-dispatch check was a flat `max_tokens ≤ min(4096,
max_model_len)` cap (`ChatService._enforce_max_tokens_cap`). Nothing checked the *prompt*
length against the model's context window, and nothing tracked how much of the KV pool
was already reserved by in-flight requests — a burst of concurrent requests could only be
told apart from a healthy engine by watching it OOM or emit truncated/empty completions.
This is Phase 2 of the VRAM-aware architecture plan (`docs/DECISIONS.md` DEC-037), which
also planned a non-streaming continuous-batching fix and engine-knob surfacing under the
same phase — both were attempted or scoped and are **not** part of this change; see the
"Out of scope" section and DEC-038 for why.

## In scope
- `routing/admission.py`: `AdmissionController.admit(routed_model, request, engine)` —
  context-length gate (prompt + requested output vs. `ModelEntry.max_model_len` ∩ VRAM
  tier `max_model_len_cap` ∩ request `max_context_tokens`) and a KV-pressure gate
  (in-flight token reservations vs. `engine.kv_capacity_tokens * 0.9`)
- `priority: "interactive" | "batch"` request field: interactive clamps `max_tokens` down
  to what fits; batch rejects outright rather than silently truncating
- `max_context_tokens`, `max_output_tokens` (alias for `max_tokens`, additive/back-compat)
  request fields
- `VLLMEngine.count_prompt_tokens()` — real tokenizer-based prompt token count, accessed
  via `getattr` (not added to `BaseEngine`'s abstract contract — same optional-attribute
  pattern already used for `kv_capacity_tokens`)
- `ContextTooLongError` (subclasses `ValueError`, reuses the existing sanitized 400
  handler) and `EngineSaturatedError` (new, 429 + `Retry-After`)
- `ChatService` wires admission around every `generate()`/`generate_stream()` call,
  releasing the KV reservation in a `finally` block

## Out of scope (see DEC-038)
- **Non-streaming continuous-batching fix.** Implemented, unit-tested, then reverted: a
  live 2-concurrent-request test reproducibly lost one request's output (a `step()`-loop
  race that `generate_stream` tolerates via cumulative text but a one-shot `finished`
  handoff does not). Needs a shared per-engine driver thread, not N independent
  request-owned loops. Non-streaming requests still serialize per model, unchanged.
- **Engine knob surfacing** (`block_size`, `max_num_batched_tokens`, `kv_cache_dtype`,
  `enable_prefix_caching`) — not started this round.
- **`precision` request field / variant selection** — meaningless without model variant
  sets (a separate, unstarted component of the VRAM-aware plan); not added.
- Phase 3 of the VRAM-aware plan (CPU/weight offload, prefix caching).
