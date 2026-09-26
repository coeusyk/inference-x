# Proposal: complete-openai-serving-surface (V1-0)

## Why

v1.0 re-centers InferenceX on being a local server a real coding client can use
instead of Ollama (`docs/INFERENCEX-EVOLUTION.md` §9–10). V1-0 proves that with the
current single-model server before anything larger is built. The exit test is a real
Aider session.

Aider's actual traffic was captured on 2026-09-26 (aider 0.86.2, whole/diff/udiff edit
formats, streaming and non-streaming). Every request was `{model, messages, stream?,
temperature: 0}`, with string contents and only the `system`/`user`/`assistant` roles.
It sent no `max_tokens`, `stop`, `stream_options`, or tools. Against that traffic, the
current server fails in these ways:

1. **Aider cannot fit on the 6gb tier at all.** The tier's `max_model_len_cap` is 2048
   tokens. Aider's system prompt plus repo map was about 5k tokens (21k chars) on a
   two-file repo.
2. **Omitted `max_tokens` becomes 512** (schema default). Aider's whole-file edits
   would be truncated.
3. **DEC-025 caps break long sessions.** They allow ≤ 50 messages (each turn adds 2)
   and ≤ 32,000 chars per message (the chat-files message was already about 16k chars
   with two files added).
4. **Context overflow is not actionable.** `ContextTooLongError` is a `ValueError`, so
   the generic handler replaces the message with "Request could not be processed."
   litellm (which Aider uses) recognizes context overflow by the phrase "maximum
   context length", so the user never learns the prompt was too long.
5. **Unknown OpenAI fields (`stop`, `n`, …) are silently dropped** by pydantic. This
   contradicts the never-lie principle. `stop` is also a commonly used parameter in
   its own right, and Continue (the next acceptance client) relies on it.
6. **Validation errors use FastAPI's `422 {"detail": …}`**, not the OpenAI error
   envelope.
7. **Superseded PR #38** would have put the full run manifest in every response. The
   v1.0 direction is that the manifest is opt-in, and the default shape stays exactly
   what `develop` returns today.

## What changes

- **`stop`**: a string or a list of ≤ 4 strings, forwarded to the sampler and echoed in
  `resolved` and in the manifest's `sampling` block.
- **Omitted `max_tokens`** resolves to the remaining context window (prompt tokens
  subtracted), which is OpenAI's semantics. This produces no warning, because nothing
  was substituted. The KV gate still applies and still warns if it clamps. The fixed
  4096 upper bound on `max_tokens`/`max_output_tokens` is removed: the context gate is
  the real bound.
- **Request bounds become token-based.** The context gate enforces them. The
  character and message-count limits remain only as body-size guards, set well above
  any supported context: ≤ 2048 messages, ≤ 1,000,000 chars per message.
- **Unknown fields are rejected** with 400 and a named `param`, on the request and on
  each message. Nothing is silently dropped.
- **Validation errors use the OpenAI error envelope**: HTTP 400, `type:
  invalid_request_error`, with `param` set.
- **Context overflow** returns 400 with `code: context_length_exceeded` and an OpenAI
  style message ("This model's maximum context length is N tokens…"). Strict-mode
  rejections return their real message with `code: strict_violation`.
- **Full manifest is opt-in.** The new request field `include_manifest: true` is
  allowed on non-streaming requests only (400 if combined with `stream: true`). It adds
  the `manifest` body field. The default response omits the key entirely. `timing`,
  `resolved`, `warnings`, and `run_id` are unchanged, and so is `X-Run-Id`. This
  reuses #38's schema field and reconstruction test.
- **VRAM tier context cap composes as a KV-budget envelope** (owner decision,
  2026-09-26). A model entry may exceed the tier's `max_model_len_cap` only if its
  effective `max_num_seqs` is low enough that `max_model_len × max_num_seqs ≤
  max_model_len_cap × tier.max_num_seqs`. Otherwise the tier cap applies, as today,
  but is now reported rather than silent. The effective context and concurrency are
  reported by `/v1/models` and `/v1/plan`. This is a policy envelope. Physical safety
  still comes from vLLM's startup KV check and the admission KV gate's measured
  `kv_capacity_tokens`.
- **New model entry `qwen2.5-coder-1.5b`**: Qwen2.5-Coder-1.5B-Instruct, 8192 context,
  1 sequence. It is the Aider acceptance model on the 6 GiB laptop.

## Not in scope

- **Tools and function calling.** Aider did not send them (its default edit formats
  don't use them).
- **`n`, penalties, logprobs, `response_format`, and array-form message content.** They
  are added only if an acceptance client proves it needs them. Until then, sending
  them gets a clear 400.
- **Making `timing`, `resolved`, `warnings`, or `run_id` opt-in.** Owner decision: keep
  them in the default response.
- **Passing `priority` to vLLM, and admission-restoration instrumentation.** Separate,
  non-blocking investigation after V1-0.
- **llama.cpp, the lifecycle supervisor, and the CLI.** V1-1 to V1-3.

## Impact

- **Code:**
  - `schemas/chat.py`: request fields, bounds, `extra="forbid"`, `manifest`.
  - `schemas/common.py`: `param` on the error body.
  - `api/errors.py` and `api/main.py`: new handlers.
  - `api/routes/chat_completions.py`: omit an absent manifest.
  - `routing/admission.py`: default output, context ceiling via the tier composition,
    error text.
  - `utils/vram_tiers.py`: composition helper.
  - `api/routes/models.py` and `api/routes/plan.py`: effective context and sequences.
  - `engines/vllm_engine.py`: `stop` into `SamplingParams`.
  - `services/chat_service.py`: manifest `stop`, opt-in.
  - `config/models.yaml`: new entry.
- **Contract:**
  - Validation status goes from 422 to 400; the envelope changes from `detail` to
    `error`.
  - Previously ignored unknown fields now return 400.
  - Omitted `max_tokens` is no longer 512.
  - All of these are recorded in a DEC.
- **Varex:** sends only `model`/`messages`/`max_tokens`
  (`varex/src/models/openai_compat.py`), so it is unaffected by rejecting unknown
  fields.
- **run_id:** the manifest `sampling` block gains `stop`, so `run_id` values change.
  They already change on every commit, because `engine.git_sha` is in the preimage.
