## Why

Phase 5 needs streaming responses so the chat API can support OpenAI-compatible clients and make local demos feel responsive instead of waiting for a full completion. The existing DEC-010 `stream=true` rejection was correct for Phase 1, but now blocks the intended streaming feature.

The playground also needs a richer interactive terminal experience for article screenshots and demos while keeping the existing batch comparison client intact.

## What Changes

- Enable `stream=true` on `POST /v1/chat/completions` and return Server-Sent Events with OpenAI-compatible chat completion chunks.
- Preserve the existing non-streaming JSON response path for `stream=false` and omitted `stream`.
- Extend the engine interface with raw text chunk streaming and implement it for the vLLM engine.
- Ensure observability middleware does not buffer `text/event-stream` responses, so streamed tokens reach clients live.
- Add a Textual-based interactive playground that consumes the streaming API with `httpx`.
- Add focused tests for streaming SSE formatting, route behavior, playground parsing helpers, model listing, and health checks.
- Record DEC-023 for SSE streaming and DEC-024 for the Textual playground; mark DEC-010 superseded.

## Capabilities

### New Capabilities
- `interactive-playground`: Textual terminal playground with model selection, compare mode, live streaming, health status, and keyboard controls.

### Modified Capabilities
- `platform`: `POST /v1/chat/completions` accepts `stream=true` and returns OpenAI-compatible SSE chunks instead of rejecting the request.

## Impact

- API: `POST /v1/chat/completions` gains a streaming response mode for `stream=true`; `GET /health` and `GET /v1/models` remain unchanged.
- Services and engines: `ChatService` gets a streaming orchestration method, and `BaseEngine` implementations provide raw text chunk streaming.
- Observability: streaming chat responses are timed and recorded without response-body token extraction.
- Playground: new `playground/app.py` and `playground/app.css`; existing `playground/client.py` remains unchanged.
- Dependencies: `textual>=0.60` and `httpx>=0.27` become runtime dependencies.
- Tests and docs: new unit tests, updated streaming expectations, decision records, and article notes.
