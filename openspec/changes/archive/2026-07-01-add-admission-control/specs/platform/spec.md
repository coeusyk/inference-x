# Platform Specification Delta — add-admission-control

## MODIFIED Requirements

**Requirement: Pre-dispatch admission control**
The system SHALL validate a chat completion request's context-length and KV-pool
requirements after model routing and before engine dispatch, rejecting or clamping
requests that cannot be safely served instead of forwarding them to the engine
unconditionally.

Scenario: Prompt exceeds the context window
- WHEN a request's prompt token count exceeds the effective context ceiling
  (`min(ModelEntry.max_model_len, VRAM tier max_model_len_cap, request.max_context_tokens)`)
- THEN the request is rejected with HTTP 400 and a sanitized error message
- AND the engine is never invoked

Scenario: Requested output does not fit remaining context
- WHEN `prompt_tokens + requested_output_tokens` exceeds the context ceiling
- THEN an `interactive`-priority request has `max_tokens` clamped to the remaining room
  (unless that room is below a minimum useful size, in which case it is rejected)
- AND a `batch`-priority request is rejected outright, never silently truncated

Scenario: KV pool is under pressure
- WHEN in-flight reservations plus this request's estimated KV need would exceed the
  engine's reported KV-pool capacity (with a safety margin)
- THEN a `batch`-priority request is rejected with HTTP 429 and a `Retry-After` header
- AND an `interactive`-priority request is clamped to the available room when that room
  is still useful, or rejected otherwise

Scenario: Required signals are unavailable
- WHEN the engine does not expose a tokenizer or a reported KV capacity
- THEN admission does not block the request on that gate (fails open), using a
  chars/4 token estimate as a fallback where a prompt count is otherwise required

**Requirement: Non-invasive integration**
This change SHALL NOT alter the request/response shape of any existing endpoint beyond
strictly additive fields, and SHALL NOT change Phase 1–7 API behavior for requests that
don't use the new fields.

Scenario: Change is applied
- WHEN this change is applied
- THEN all Phase 1–7 tests continue to pass unchanged
- AND `ChatCompletionRequest` gains only additive fields (`max_context_tokens`,
  `max_output_tokens`, `priority`)
- AND a request omitting the new fields behaves exactly as before (default
  `priority: interactive`, `max_output_tokens` falling back to `max_tokens`)
