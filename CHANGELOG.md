# Changelog

All notable user-facing changes to InferenceX are recorded here. Internal
refactors with no observable behavior change are not listed — see
`docs/DECISIONS.md` for the full architectural decision record.

## [Unreleased]

### Added

- **`stop` sequences** on `POST /v1/chat/completions` (a string or up to 4
  strings), honored by the engine and echoed in `resolved`.
- **Opt-in full run manifest**: `include_manifest: true` on a non-streaming
  request returns the complete manifest in a `manifest` field. The default
  response is unchanged and has no `manifest` key; `run_id` and `X-Run-Id` are
  still always present.
- **`/v1/models` and `/v1/plan` report each model's effective context window**
  (`context_window`, `max_num_seqs`, `context_composed`,
  `context_tier_limited`). A model may exceed its VRAM tier's per-sequence
  context cap by running fewer concurrent sequences. The new
  `qwen2.5-coder-1.5b` entry uses this to give 8k context on a 6 GB GPU.
- **`GET /metrics`: vLLM's own native Prometheus stats are now exposed
  directly**, alongside the existing `GET /v1/metrics` JSON summary.
  `/metrics` returns vLLM's `vllm:`-prefixed series (request queue depth, KV
  cache utilization, per-phase timing, and more) in standard Prometheus
  text-exposition format. `/v1/metrics` is unchanged — it continues to
  report only HTTP-boundary metrics (latency, TTFT, tokens/sec) — and
  scraping `/metrics` never affects `/v1/metrics`'s figures.
- **Chat completion responses now include per-request engine timing.** Both
  `POST /v1/chat/completions` (non-streaming) and the streaming terminal
  event carry an optional `timing` field: `queue_time_ms`, `prefill_time_ms`,
  `decode_time_ms`, and `inference_time_ms`, sourced directly from the
  engine's own internal clock. This decomposes a request's latency into
  queueing versus prefill versus decode — a breakdown no client could
  reconstruct from the outside. `timing` is `None` when the engine cannot
  supply it, never an estimate.

### Changed

- **Omitting `max_tokens` now means "up to the context window"**, as in
  OpenAI's API, instead of a fixed 512 tokens.
- **A model's `max_completion_tokens` is now applied before generation and
  reported.** A larger `max_tokens` is clamped with a `substituted` warning
  (`max_tokens_clamped_to_model_cap`), or rejected under `strict`, and
  `resolved.max_tokens` shows the value that ran. Previously the engine
  replaced the requested value after the fact, so `resolved` could disagree
  with the actual generation length in either direction.
- **Request size limits are now token-based.** The old caps of 4096
  `max_tokens`, 50 messages, and 32,000 characters per message are gone; the
  model's context window is the limit.
- **Unknown request fields are now rejected with 400** naming the field.
  Previously they were silently ignored.
- **Validation errors now return 400 in the OpenAI error shape**
  (`{"error": {"message", "type", "param", "code"}}`), not 422 `{"detail": …}`.
- **Context-overflow errors now say what went wrong**: "This model's maximum
  context length is N tokens…", with code `context_length_exceeded`. Strict
  mode rejections return their real reason with code `strict_violation`.
- **Cancellation now actually stops the model computing for a disconnected
  client.** Previously, if a client disconnected mid-stream, the server
  stopped *reading* the response but the underlying generation kept running
  to completion on the GPU regardless — wasted compute with no observable
  effect. Disconnecting a streaming request now sends a real abort signal
  into the inference engine, which stops the in-flight computation.
- **Request timeouts now actually stop the model computing, not just the
  wait.** Previously, a completion request timing out only stopped the
  server from waiting on it — the engine kept computing that request in the
  background regardless. A timeout now aborts the underlying computation
  the same way a client disconnect does. The timeout duration itself is
  unchanged.
- Internally, `VLLMEngine` now runs on vLLM's `AsyncLLM` engine client
  instead of the previous synchronous engine + custom driver thread. This
  has no effect on the request/response format, streaming protocol, or
  reported token usage — see DEC-058 for the full record if you're
  curious about the internals.
- The free-VRAM probe (`GET /v1/metrics`'s VRAM breakdown, and the
  preflight check run before a model loads) now queries `nvidia-smi`
  instead of reading `torch.cuda.mem_get_info()`, which reported memory as
  seen by this process's own CUDA context and went stale across sibling
  processes — see DEC-059.

### Removed

- **A single server process can no longer load more than one model.**
  `INFERENCE_X_LOADED_MODELS` set to more than one distinct model is now a
  startup error instead of loading every listed model into one process.
  Every model sharing a process paid for it for the whole session — no CUDA
  graphs (`enforce_eager` forced), `max_model_len` silently clamped to
  2048, and gpu_memory_utilization sized by heuristics calibrated against
  past failures rather than measured VRAM. Comparing two models (`make
  playground`, `playground/client.py --compare`) now runs one process per
  model instead — `make playground` and `make playground-compare` do this
  automatically; see `playground/README.md` for the manual dual-process
  pattern. Full rationale: DEC-059.
