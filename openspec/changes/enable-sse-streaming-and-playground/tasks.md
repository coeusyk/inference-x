## 1. OpenSpec and Backend Streaming

- [x] 1.1 Add `generate_stream()` to `BaseEngine`, `VLLMEngine`, and test stub engines.
- [x] 1.2 Implement `ChatService.stream_response()` and remove the DEC-010 stream guard from the non-streaming path.
- [x] 1.3 Branch the chat completions route to return `StreamingResponse` for `stream=true`.
- [x] 1.4 Skip observability response-body buffering for SSE chat responses.

## 2. Backend Tests and Documentation

- [x] 2.1 Add `tests/unit/test_streaming.py` for SSE formatting, `[DONE]`, streaming route behavior, and non-streaming JSON behavior.
- [x] 2.2 Update existing stream guard tests to assert the new streaming behavior.
- [x] 2.3 Mark DEC-010 superseded and add DEC-023 for SSE streaming.
- [x] 2.4 Add Phase 5 streaming notes to `docs/ARTICLE_NOTES.md`.

## 3. Textual Playground

- [x] 3.1 Add `textual>=0.60` and `httpx>=0.27` runtime dependencies and run `uv sync`.
- [x] 3.2 Add testable playground helper functions for SSE parsing, health checks, and model listing.
- [x] 3.3 Implement `playground/app.py` with single-model and compare-mode streaming panels.
- [x] 3.4 Implement `playground/app.css` for the dark two-panel layout.

## 4. Playground Tests, Docs, and Validation

- [x] 4.1 Add `tests/unit/test_app.py` for helper functions without launching the full Textual app.
- [x] 4.2 Update `playground/README.md` with interactive launch instructions and shortcuts.
- [x] 4.3 Add DEC-024 for Textual + httpx interactive playground and update `docs/ARTICLE_NOTES.md`.
- [x] 4.4 Run `uv run pytest tests/unit/ -v`.
- [x] 4.5 Launch `uv run python playground/app.py` and confirm startup behavior against the available server.
