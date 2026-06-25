## Why

Phase 4 delivered a multi-tab Textual playground (`playground/app.py`) for compare and benchmark workflows. Daily chat use needs a focused, Claude-style persistent conversation UI with scrollable history, bottom input, and efficient token streaming — without replacing the existing benchmark/compare tool.

Background server logs from `make playground` also corrupt the alternate-screen TUI when uvicorn writes to the same TTY.

## What Changes

- Add `playground/chat.py` — single-purpose Textual chat CLI with multi-turn history and `Markdown.get_stream()` rendering.
- Add `playground/streaming.py` — thin module re-exporting SSE helpers from `app.py` plus `stream_chat_tokens()` async generator.
- Add `make chat` — starts server (logs redirected) + launches `chat.py`.
- Redirect background uvicorn output to `logs/playground-server.log` for `playground`, `playground-compare`, and `chat` Makefile targets.
- Pin `textual>=8.2,<9` in `pyproject.toml` (`Markdown.get_stream()` requires Textual 8.x).
- Unit tests for streaming and chat payload helpers.

## Capabilities

### New Capabilities
- `playground-chat-cli`: Claude-style terminal chat with SSE streaming, model picker, and keyboard shortcuts.

### Modified Capabilities
- None (no API or server contract changes).

## Impact

- Playground: new `chat.py`, `chat.css`, `streaming.py`; `app.py` unchanged.
- Makefile: new `chat` target; log redirect on TUI server bootstrap targets.
- Dependencies: `textual` pin tightened to 8.x.
- Docs: README, playground README, PHASES.md, DECISIONS.md.
