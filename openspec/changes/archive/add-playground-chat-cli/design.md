## Design

### Chat CLI layout

```
Header (title · URL · health · model)
VerticalScroll #history  (1fr, scrollable turn log)
Input #input             (docked bottom)
```

- **User turns**: `Static` widget, right-aligned, dim styling.
- **Assistant turns**: `Markdown` widget; tokens appended via `Markdown.get_stream()` + `await stream.write(token)`.
- **Scroll pinning**: `history.anchor()` before stream; `history.release_anchor()` after `await stream.stop()`.
- **Multi-turn**: `self._messages` holds full OpenAI-style message list; every request sends the accumulated history.

### Shared streaming module

`playground/streaming.py` re-exports `parse_sse_line`, `parse_sse_stream`, `fetch_health`, `fetch_models` from `app.py` (no duplication). Adds `stream_chat_tokens()` using the same httpx SSE loop as `app._stream_model`.

### TUI log isolation

Root cause: Makefile starts uvicorn in the background on the same TTY (`&`). Textual captures the foreground process streams but not the sibling server process.

Fix: redirect server stdout/stderr to `logs/playground-server.log` in Makefile only. No server-side `INFERENCE_X_TUI_MODE` env hook — keeps production `./scripts/dev.sh serve` console logging unchanged.

### Textual version

`Markdown.get_stream()` / `MarkdownStream` exist in Textual 8.x only. Pin `textual>=8.2,<9` before landing chat code.
