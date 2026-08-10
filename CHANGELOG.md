# Changelog

All notable user-facing changes to InferenceX are recorded here. Internal
refactors with no observable behavior change are not listed — see
`docs/DECISIONS.md` for the full architectural decision record.

## [Unreleased]

### Added

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
