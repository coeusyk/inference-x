# Changelog

All notable user-facing changes to InferenceX are recorded here. Internal
refactors with no observable behavior change are not listed — see
`docs/DECISIONS.md` for the full architectural decision record.

## [Unreleased]

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
