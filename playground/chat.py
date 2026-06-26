#!/usr/bin/env python3
"""Claude-style multi-turn chat CLI for InferenceX.

Daily-driver interface: scrollable turn history, bottom input bar, SSE
streaming via Markdown.get_stream().  Sends full conversation history on
every request (true multi-turn). Compare and daily chat are separate entry points:
`make chat` for single-model use, `make playground` for side-by-side evaluation.

Launch via:
    make chat
    uv run python playground/chat.py --allow-internal
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

try:
    from streaming import stream_chat_tokens
    from startup_screen import ModelSelectScreen
    from url_validation import validate_base_url
    from loading_screen import LoadingScreen
    from server_control import ensure_models_loaded, cleanup_playground_server_if_started_sync
except ImportError:
    from playground.streaming import stream_chat_tokens  # type: ignore[no-redef]
    from playground.startup_screen import ModelSelectScreen  # type: ignore[no-redef]
    from playground.url_validation import validate_base_url  # type: ignore[no-redef]
    from playground.loading_screen import LoadingScreen  # type: ignore[no-redef]
    from playground.server_control import (  # type: ignore[no-redef]
        ensure_models_loaded,
        cleanup_playground_server_if_started_sync,
    )

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Input, Label, Markdown, Static

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512


def build_chat_payload(
    messages: list[dict[str, str]],
    model: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Build an OpenAI-compatible streaming chat completion request body."""
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }


def should_submit_prompt(prompt: str) -> bool:
    """Return True when *prompt* is non-empty after stripping."""
    return bool(prompt.strip())


def append_turn(messages: list[dict[str, str]], role: str, content: str) -> list[dict[str, str]]:
    """Return a new message list with one turn appended."""
    updated = list(messages)
    updated.append({"role": role, "content": content})
    return updated


# ---------------------------------------------------------------------------
# MessageBubble widget
# ---------------------------------------------------------------------------

class MessageBubble(Vertical):
    """One turn in the conversation history.

    Role "user"  → right-aligned plain Static, dim styling.
    Role "assistant" → left-aligned Markdown widget for rendered output.
    """

    def __init__(self, role: str, content: str = "") -> None:
        super().__init__(classes=f"bubble bubble-{role}")
        self._role = role
        self._content = content

    def compose(self) -> ComposeResult:
        if self._role == "user":
            yield Static(self._content, classes="bubble-text-user")
        else:
            yield Markdown(self._content, classes="bubble-markdown")

    @property
    def markdown_widget(self) -> Markdown | None:
        """Return the Markdown child for assistant bubbles, else None."""
        if self._role != "assistant":
            return None
        try:
            return self.query_one(Markdown)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# ChatApp
# ---------------------------------------------------------------------------

class ChatApp(App[None]):
    """Multi-turn chat TUI for InferenceX."""

    CSS_PATH = Path(__file__).with_name("chat.css")

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+n", "new_conversation", "New chat"),
    ]

    def __init__(
        self,
        base_url: str,
        *,
        model: str | None = None,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self._model: str | None = model
        self._messages: list[dict[str, str]] = []
        self._healthy: bool = False
        self._in_flight: bool = False

    # ── Layout ────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Label("", id="header-bar")
        yield VerticalScroll(id="history")
        yield Input(placeholder="Message… (Enter to send, Ctrl+N new chat, Ctrl+C quit)", id="input")

    # ── Startup ───────────────────────────────────────────────────────────

    async def on_mount(self) -> None:
        self.query_one("#input", Input).disabled = True
        if self._model is None:
            # push_screen_wait must run inside a worker (Textual requirement).
            self.run_worker(self._startup_flow(), exclusive=True)
        else:
            self.run_worker(self._post_select_init(), exclusive=True)

    async def _startup_flow(self) -> None:
        """Show model picker then proceed to health sync."""
        selected = await self.push_screen_wait(
            ModelSelectScreen(base_url=self.base_url, initial_model=None)
        )
        if selected:
            self._model = selected
        await self._post_select_init()

    async def _post_select_init(self) -> None:
        """Ensure server loads the chosen model, then enable input."""
        input_widget = self.query_one("#input", Input)
        if not self._model:
            self.exit()
            return

        loading = LoadingScreen("Preparing your model…")
        await self.push_screen(loading)
        painted = asyncio.Event()
        loading.call_after_refresh(painted.set)
        await painted.wait()
        ok = await ensure_models_loaded(
            self.base_url,
            [self._model],
            on_status=loading.set_message,
            on_log=loading.append_log,
        )
        if ok:
            self.pop_screen()
        else:
            try:
                from log_feed import extract_error_summary
            except ImportError:
                from playground.log_feed import extract_error_summary

            loading.finish_failed(extract_error_summary())
            await loading.wait_for_ack()

        await self._sync_health()
        if not ok:
            self._render_header(status="Failed to load model — see logs/playground-server.log")
            return

        self._render_header()
        input_widget.disabled = False
        input_widget.focus()

    async def _sync_health(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/health")
                self._healthy = (
                    resp.status_code == 200
                    and resp.json().get("status") == "healthy"
                )
        except Exception:
            self._healthy = False

    # ── Header ────────────────────────────────────────────────────────────

    def _render_header(self, *, status: str | None = None) -> None:
        model_str = self._model or "—"
        if status:
            health_str = f"● {status}"
            health_class = "offline" if "not available" in status.lower() else ""
        else:
            health_str = "● healthy" if self._healthy else "● offline"
            health_class = "healthy" if self._healthy else "offline"
        label = self.query_one("#header-bar", Label)
        label.update(
            f"[bold #7aa884]InferenceX Chat[/]  "
            f"[#7f8795]{self.base_url}[/]  "
            f"[#596275]{model_str}[/]  "
            f"[{('#7aa884' if self._healthy and not status else '#d06c75')}]{health_str}[/]"
        )
        label.remove_class("healthy", "offline")
        if health_class:
            label.add_class(health_class)

    # ── Send flow ─────────────────────────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value
        if not should_submit_prompt(prompt) or self._in_flight:
            return
        prompt = prompt.strip()

        self._in_flight = True
        input_widget = self.query_one("#input", Input)
        input_widget.disabled = True
        input_widget.clear()

        history = self.query_one("#history", VerticalScroll)

        # 1. User bubble
        user_bubble = MessageBubble("user", prompt)
        await history.mount(user_bubble)
        self._messages = append_turn(self._messages, "user", prompt)

        # 2. Empty assistant bubble — will be streamed into
        assistant_bubble = MessageBubble("assistant", "")
        await history.mount(assistant_bubble)

        # 3. Anchor scroll to bottom before streaming begins
        history.anchor()

        # 4. Stream tokens into the assistant bubble's Markdown widget
        md = assistant_bubble.markdown_widget
        collected = ""
        error_msg: str | None = None

        if md is not None:
            payload = build_chat_payload(self._messages, self._model or "")
            stream = Markdown.get_stream(md)
            try:
                async for token in stream_chat_tokens(self.base_url, payload):
                    await stream.write(token)
                    collected += token
            except httpx.HTTPStatusError as exc:
                error_msg = f"HTTP {exc.response.status_code}: {exc.response.text[:120]}"
            except httpx.ReadTimeout:
                error_msg = "Read timeout while streaming."
            except httpx.HTTPError as exc:
                error_msg = f"Request failed: {exc}"
            finally:
                await stream.stop()
        else:
            error_msg = "Could not find Markdown widget in assistant bubble."

        # 5. Record assistant turn (or error placeholder)
        if error_msg:
            # Show error inline without polluting the messages history
            try:
                err_md = assistant_bubble.query_one(Markdown)
                err_md.update(f"**Error:** {error_msg}")
            except Exception:
                pass
        else:
            self._messages = append_turn(self._messages, "assistant", collected)

        # 6. Release anchor, re-enable input
        history.release_anchor()
        self._in_flight = False
        input_widget.disabled = False
        input_widget.focus()

    # ── Actions ───────────────────────────────────────────────────────────

    async def action_new_conversation(self) -> None:
        """Clear history and start a fresh conversation."""
        if self._in_flight:
            return
        self._messages = []
        history = self.query_one("#history", VerticalScroll)
        await history.remove_children()

    def action_quit(self) -> None:
        self.exit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InferenceX Claude-style chat CLI")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Server base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Skip model picker and use this model directly.",
    )
    parser.add_argument(
        "--allow-internal",
        action="store_true",
        help="Allow loopback and private-network base URLs (local dev).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_url = args.base_url.rstrip("/")
    url_error = validate_base_url(base_url, allow_internal=args.allow_internal)
    if url_error:
        print(f"Error: {url_error}", file=sys.stderr)
        sys.exit(1)
    app = ChatApp(base_url=base_url, model=args.model)
    try:
        app.run()
    finally:
        cleanup_playground_server_if_started_sync()


if __name__ == "__main__":
    main()
