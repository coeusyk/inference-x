## Context

`ChatCompletionRequest.stream` already exists, but DEC-010 rejects `stream=true` in `ChatService.complete()`. Phase 5 intentionally lifts that guard and adds a streaming path while preserving the non-streaming JSON contract.

The current observability middleware buffers every successful chat response to extract usage tokens. That is acceptable for JSON responses but would defeat SSE because clients would not receive live chunks until the body iterator is fully consumed. Streaming therefore requires a middleware exemption in addition to route and service changes.

Phase 4 added `playground/client.py` for batch and compare workflows. The interactive Phase 5 playground should be additive and must not alter that client.

## Goals / Non-Goals

**Goals:**
- Return OpenAI-compatible SSE events for `stream=true` chat completion requests.
- Keep `stream=false` and omitted `stream` responses unchanged.
- Extend the engine abstraction with raw text chunk streaming.
- Avoid buffering `text/event-stream` responses in observability middleware.
- Add a Textual TUI that consumes the streaming API with `httpx`.
- Add tests for streaming formatting, route behavior, and playground helper functions.

**Non-Goals:**
- Do not change `GET /health`, `GET /v1/models`, startup behavior, routing policy, or model registry behavior.
- Do not change the non-streaming response schema.
- Do not modify `playground/client.py`.
- Do not build a browser UI or npm-based frontend.
- Do not introduce observability dashboards or storage backends.

## Decisions

### Streaming belongs in the service and engine layers

`BaseEngine.generate_stream()` yields raw text chunks. `ChatService.stream_response()` owns OpenAI-compatible SSE framing. This keeps vLLM-specific code unaware of HTTP wire formatting and keeps the route handler thin.

Alternative considered: format SSE events in the route handler. Rejected because it would duplicate routing and engine orchestration in the API layer.

### Non-streaming uses the existing path

The existing `complete()` method continues returning `ChatCompletionResponse`. The route chooses `StreamingResponse` only when `request.stream is True`.

Alternative considered: make `complete()` return either a typed response or an async iterator. Rejected because it would blur the public service contract and make tests less direct.

### Observability records streamed requests without token extraction

For `text/event-stream` responses, middleware records path, model, status, latency, and error state but skips body buffering and token extraction. Token counts are available after non-streaming JSON completions only.

Alternative considered: parse SSE chunks inside middleware to count tokens. Rejected because middleware should not wrap the streaming iterator in a way that risks changing delivery timing.

### Textual is additive to the rich CLI

The existing rich CLI remains the batch comparison tool. The new Textual app is the live demo artifact with health status, keyboard shortcuts, and token streaming.

Alternative considered: extend `playground/client.py` into an interactive app. Rejected because the current file has stable batch behavior and tests that should remain intact.

## Risks / Trade-offs

- vLLM async streaming API shape can vary by version → keep the implementation localized in `vllm_engine.py` and preserve a dev/test fallback when vLLM is unavailable.
- Streaming responses do not include final token usage from the backend yet → show a client-side token estimate in the playground and document the tradeoff in DEC-023.
- Textual app tests should not start the full app event loop → expose pure parser, health, and model helper functions for unit tests.
- Manual TUI smoke requires a running server → if no server is available, record that automated tests passed and the manual connection check was blocked by environment state.

## Migration Plan

1. Create the streaming engine interface and service method.
2. Add the streaming route branch and middleware streaming exemption.
3. Update tests and docs for DEC-010 supersession.
4. Add Textual dependencies and run `uv sync`.
5. Implement and test playground helper functions and TUI.
6. Run `uv run pytest tests/unit/ -v`.
7. Manually launch `uv run python playground/app.py` against the running server when available.
