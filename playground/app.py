#!/usr/bin/env python3
"""Interactive Textual playground for InferenceX."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx

try:
    from url_validation import validate_base_url
    from startup_screen import ModelSelectScreen
    from loading_screen import LoadingScreen
    from server_control import ensure_models_loaded
except ImportError:
    from playground.url_validation import validate_base_url
    from playground.startup_screen import ModelSelectScreen
    from playground.loading_screen import LoadingScreen
    from playground.server_control import ensure_models_loaded

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, Label, Markdown, Rule, Static

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "qwen2.5-0.5b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Pure helper functions (used by tests and the TUI)
# ---------------------------------------------------------------------------

def parse_sse_line(line: str) -> str | None:
    """Extract a token from one OpenAI-compatible SSE data line."""
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None

    data = line.removeprefix("data:").strip()
    if not data or data == "[DONE]":
        return None

    try:
        payload = json.loads(data)
        choices = payload.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        return content if isinstance(content, str) else None
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None


def parse_sse_stream(lines: Iterable[str]) -> list[str]:
    """Return all token chunks found in an iterable of SSE lines."""
    return [token for line in lines if (token := parse_sse_line(line)) is not None]


def fetch_health(base_url: str, client: httpx.Client) -> bool:
    """Return True when the InferenceX health endpoint is reachable and healthy."""
    try:
        response = client.get(f"{base_url}/health", timeout=5.0)
        if response.status_code != 200:
            return False
        payload = response.json()
        return payload.get("status") == "healthy"
    except (httpx.HTTPError, ValueError):
        return False


def fetch_models(base_url: str, client: httpx.Client) -> list[str]:
    """Return model ids from GET /v1/models."""
    try:
        response = client.get(f"{base_url}/v1/models", timeout=10.0)
        response.raise_for_status()
        payload = response.json()
        return [
            item["id"]
            for item in payload.get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
    except (httpx.HTTPError, KeyError, ValueError, TypeError):
        return []


# ---------------------------------------------------------------------------
# ResponsePanel widget
# ---------------------------------------------------------------------------

class ResponsePanel(Vertical):
    """A titled response pane with markdown content, empty-state, and a usage footer."""

    def __init__(self, model: str, *, panel_id: str) -> None:
        super().__init__(id=panel_id, classes="response-panel")
        self.model = model
        self.content = ""
        self.started_at: float | None = None
        self.completed = False

    def compose(self) -> ComposeResult:
        yield Static(self._title_text(), classes="panel-title")
        with VerticalScroll(classes="response-scroll"):
            yield Static(
                "Send a message to begin",
                classes="empty-state",
            )
            yield Markdown("", classes="response-markdown")
        yield Static("", classes="usage")

    def clear_response(self) -> None:
        self.content = ""
        self.started_at = None
        self.completed = False
        try:
            self.query_one(".empty-state", Static).display = True
            self.query_one(Markdown).update("")
            self.query_one(".usage", Static).update("")
        except Exception:
            pass
        self.refresh_title()

    def start(self) -> None:
        self.content = ""
        self.started_at = time.perf_counter()
        self.completed = False
        try:
            self.query_one(".empty-state", Static).display = False
            self.query_one(Markdown).update("")
            self.query_one(".usage", Static).update("")
        except Exception:
            pass
        self.refresh_title()

    def append_token(self, token: str) -> None:
        self.content += token
        try:
            self.query_one(Markdown).update(self.content)
        except Exception:
            pass
        self.refresh_title()

    def finish(self, prompt: str) -> None:
        self.completed = True
        prompt_tokens = len(prompt.split())
        completion_tokens = len(self.content.split())
        total_tokens = prompt_tokens + completion_tokens
        elapsed = self.elapsed_seconds()
        try:
            self.query_one(".usage", Static).update(
                f"prompt {prompt_tokens} | completion {completion_tokens} | "
                f"total {total_tokens} | {elapsed:.1f}s"
            )
        except Exception:
            pass
        self.refresh_title()

    def set_error(self, message: str) -> None:
        self.completed = True
        self.content = message
        try:
            self.query_one(".empty-state", Static).display = False
            self.query_one(Markdown).update(f"**Error:** {message}")
            self.query_one(".usage", Static).update("")
        except Exception:
            pass
        self.refresh_title()

    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.perf_counter() - self.started_at

    def refresh_title(self) -> None:
        try:
            self.query_one(".panel-title", Static).update(self._title_text())
        except Exception:
            pass

    def _title_text(self) -> str:
        if self.started_at is None:
            return f"[bold #c9d1e3]{self.model}[/]  [#596275]ready[/]"
        state = "done" if self.completed else "streaming…"
        state_color = "#7aa884" if self.completed else "#d1a65a"
        return (
            f"[bold #c9d1e3]{self.model}[/]  "
            f"[#7f8795]{self.elapsed_seconds():.1f}s[/]  "
            f"[{state_color}]{state}[/]"
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class InferenceXApp(App[None]):
    """Side-by-side model compare playground."""

    CSS_PATH = Path(__file__).with_name("app.css")
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("q", "quit", "Quit"),
        ("ctrl+l", "clear_responses", "Clear"),
        ("f1", "toggle_help", "Help"),
    ]

    def __init__(
        self,
        *,
        base_url: str,
        compare: tuple[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.compare = compare
        self.healthy = False
        self.in_flight = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="header-bar"):
            yield Label("InferenceX Compare", id="header-title")
            yield Label(self.base_url, id="header-url")
            yield Label("● offline", id="header-health")
        yield Rule(id="header-rule")

        with Horizontal(id="response-layout"):
            model_a = self.compare[0] if self.compare else "…"
            model_b = self.compare[1] if self.compare else "…"
            yield ResponsePanel(model_a, panel_id="panel-a")
            yield ResponsePanel(model_b, panel_id="panel-b")
        yield Rule(id="content-rule")

        with Horizontal(id="prompt-bar"):
            yield Input(
                placeholder="Enter a prompt and press Enter to compare…",
                id="prompt-input",
            )
        yield Rule(id="status-rule")

        with Horizontal(id="status-bar"):
            yield Static("Ready", id="status-message")

        yield Static(self._help_text(), id="help-overlay", classes="hidden")

    async def on_mount(self) -> None:
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.disabled = True

        self.set_interval(0.2, self._refresh_live_titles)

        self.run_worker(self._startup_flow(), exclusive=True)

    async def _startup_flow(self) -> None:
        """Pick two models (if needed), start server, enable compare UI."""
        if not self.compare:
            selected = await self.push_screen_wait(
                ModelSelectScreen(self.base_url, mode="compare")
            )
            if not selected or not isinstance(selected, tuple):
                self.exit()
                return
            self.compare = selected
            panel_a = self.query_one("#panel-a", ResponsePanel)
            panel_b = self.query_one("#panel-b", ResponsePanel)
            panel_a.model = self.compare[0]
            panel_b.model = self.compare[1]
            panel_a.refresh_title()
            panel_b.refresh_title()

        models_to_load = list(self.compare)

        loading = LoadingScreen("Preparing your model…")
        await self.push_screen(loading)
        painted = asyncio.Event()
        loading.call_after_refresh(painted.set)
        await painted.wait()
        ok = await ensure_models_loaded(
            self.base_url,
            models_to_load,
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

        await self._refresh_server_state()
        self._render_header()

        if not ok:
            self._set_status(
                "Failed to load selected model(s) — see logs/playground-server.log",
                kind="error",
            )
            return

        self.query_one("#prompt-input", Input).disabled = False
        self._set_status("Ready", kind="success")

    async def _refresh_server_state(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                health = await client.get(f"{self.base_url}/health")
                self.healthy = (
                    health.status_code == 200
                    and health.json().get("status") == "healthy"
                )
                models_resp = await client.get(f"{self.base_url}/v1/models")
                if models_resp.status_code == 200:
                    ids = [
                        item["id"]
                        for item in models_resp.json().get("data", [])
                        if isinstance(item, dict) and isinstance(item.get("id"), str)
                    ]
                    if self.compare and ids:
                        missing = [m for m in self.compare if m not in ids]
                        if missing:
                            self.healthy = False
        except (httpx.HTTPError, ValueError):
            self.healthy = False

    # ── Event handlers ─────────────────────────────────────────────────────

    def action_toggle_help(self) -> None:
        self.query_one("#help-overlay", Static).toggle_class("hidden")

    def action_clear_responses(self) -> None:
        self.query_one("#panel-a", ResponsePanel).clear_response()
        self.query_one("#panel-b", ResponsePanel).clear_response()
        self._set_status("Cleared", kind="success")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt-input":
            return
        self._submit_from_input()

    def _submit_from_input(self) -> None:
        if self.in_flight:
            return
        prompt_input = self.query_one("#prompt-input", Input)
        prompt = prompt_input.value.strip()
        if not prompt:
            self._set_status("Enter a prompt first", kind="error")
            return
        self.run_worker(self._submit_prompt(prompt), exclusive=True)

    async def _submit_prompt(self, prompt: str) -> None:
        self.in_flight = True
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.disabled = True
        prompt_input.value = ""
        self._set_status("Streaming…", kind="progress")

        await self._refresh_server_state()
        self._render_header()

        panel_a = self.query_one("#panel-a", ResponsePanel)
        panel_b = self.query_one("#panel-b", ResponsePanel)
        if not self.compare:
            return
        panel_a.model = self.compare[0]
        panel_b.model = self.compare[1]

        try:
            await asyncio.gather(
                self._stream_model(self.compare[0], prompt, panel_a),
                self._stream_model(self.compare[1], prompt, panel_b),
            )

            prompt_tokens = len(prompt.split())
            completion_tokens = len(panel_a.content.split()) + len(panel_b.content.split())
            total_tokens = prompt_tokens + completion_tokens
            elapsed = max(panel_a.elapsed_seconds(), panel_b.elapsed_seconds())

            self._set_status(
                f"Done — {total_tokens} tokens in {elapsed:.1f}s",
                kind="success",
            )
        except httpx.ReadTimeout:
            self._set_status("Read timeout while streaming", kind="error")
        except httpx.HTTPError as exc:
            self._set_status(f"Request failed: {exc}", kind="error")
        finally:
            prompt_input.disabled = False
            self.in_flight = False

    async def _stream_model(
        self,
        model: str,
        prompt: str,
        panel: ResponsePanel,
    ) -> None:
        panel.start()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "stream": True,
        }
        timeout = httpx.Timeout(10.0, read=120.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip() == "data: [DONE]":
                        break
                    token = parse_sse_line(line)
                    if token:
                        panel.append_token(token)
        panel.finish(prompt)

    # ── Internal render helpers ────────────────────────────────────────────

    def _render_header(self) -> None:
        healthy = self.healthy
        health_text = "● healthy" if healthy else "● offline"
        try:
            self.query_one("#header-health", Label).update(health_text)
            self.query_one("#header-health", Label).set_class(healthy, "healthy")
            self.query_one("#header-health", Label).set_class(not healthy, "offline")
        except Exception:
            pass

    def _refresh_live_titles(self) -> None:
        if not self.in_flight:
            return
        try:
            self.query_one("#panel-a", ResponsePanel).refresh_title()
            self.query_one("#panel-b", ResponsePanel).refresh_title()
        except Exception:
            pass

    def _set_status(self, message: str, *, kind: str) -> None:
        try:
            msg = self.query_one("#status-message", Static)
            msg.update(message)
            msg.remove_class("success", "error", "progress")
            msg.add_class(kind)
        except Exception:
            pass

    @staticmethod
    def _help_text() -> str:
        return (
            "Shortcuts\n"
            "Enter       compare both models\n"
            "Ctrl+L      clear panels\n"
            "F1          toggle help\n"
            "q / Ctrl+C  quit"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="InferenceX compare playground (two models side-by-side)"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Server base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("MODEL_A", "MODEL_B"),
        help="Pre-select two models; skips the model selection screen.",
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
    compare = tuple(args.compare) if args.compare else None
    app = InferenceXApp(base_url=base_url, compare=compare)
    app.run()


if __name__ == "__main__":
    main()
