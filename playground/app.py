#!/usr/bin/env python3
"""Interactive Textual playground for InferenceX."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Markdown, Static, TextArea

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "qwen2.5-0.5b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512


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


class ResponsePanel(Vertical):
    """A titled response pane with markdown content and a usage footer."""

    def __init__(self, model: str, *, panel_id: str) -> None:
        super().__init__(id=panel_id, classes="response-panel")
        self.model = model
        self.content = ""
        self.started_at: float | None = None
        self.completed = False

    def compose(self) -> ComposeResult:
        yield Static(self._title_text(), classes="panel-title")
        with VerticalScroll(classes="response-scroll"):
            yield Markdown("", classes="response-markdown")
        yield Static("", classes="usage")

    def clear_response(self) -> None:
        self.content = ""
        self.started_at = None
        self.completed = False
        self.query_one(Markdown).update("")
        self.query_one(".usage", Static).update("")
        self.refresh_title()

    def start(self) -> None:
        self.content = ""
        self.started_at = time.perf_counter()
        self.completed = False
        self.query_one(Markdown).update("")
        self.query_one(".usage", Static).update("")
        self.refresh_title()

    def append_token(self, token: str) -> None:
        self.content += token
        self.query_one(Markdown).update(self.content)
        self.refresh_title()

    def finish(self, prompt: str) -> None:
        self.completed = True
        prompt_tokens = len(prompt.split())
        completion_tokens = len(self.content.split())
        total_tokens = prompt_tokens + completion_tokens
        elapsed = self.elapsed_seconds()
        self.query_one(".usage", Static).update(
            f"prompt {prompt_tokens} | completion {completion_tokens} | "
            f"total {total_tokens} | {elapsed:.1f}s"
        )
        self.refresh_title()

    def set_error(self, message: str) -> None:
        self.completed = True
        self.content = message
        self.query_one(Markdown).update(f"**Error:** {message}")
        self.query_one(".usage", Static).update("")
        self.refresh_title()

    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.perf_counter() - self.started_at

    def refresh_title(self) -> None:
        self.query_one(".panel-title", Static).update(self._title_text())

    def _title_text(self) -> str:
        if self.started_at is None:
            return f"{self.model}  ready"
        state = "done" if self.completed else "streaming"
        return f"{self.model}  {self.elapsed_seconds():.1f}s  {state}"


class InferenceXApp(App[None]):
    """Interactive terminal playground for live streaming completions."""

    CSS_PATH = Path(__file__).with_name("app.css")
    BINDINGS = [
        ("ctrl+enter", "submit_prompt", "Send"),
        ("ctrl+c", "quit", "Quit"),
        ("q", "quit", "Quit"),
        ("ctrl+l", "clear_responses", "Clear"),
        ("ctrl+m", "cycle_model", "Next model"),
        ("tab", "cycle_model", "Next model"),
        ("f1", "toggle_help", "Help"),
    ]

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        compare: tuple[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.compare = compare
        self.models = [model]
        if compare:
            self.models = [compare[0], compare[1]]
        self.current_model_index = 0
        self.healthy = False
        self.in_flight = False

    def compose(self) -> ComposeResult:
        yield Static("", id="header-bar")
        with Horizontal(id="response-layout"):
            yield ResponsePanel(self.models[0], panel_id="panel-a")
            yield ResponsePanel(
                self.models[1] if self.compare else "",
                panel_id="panel-b",
            )
        with Horizontal(id="prompt-bar"):
            yield Static("", id="model-selector")
            yield TextArea.code_editor(
                "",
                id="prompt-input",
                show_line_numbers=False,
            )
        yield Static("", id="status-bar")
        yield Static(self._help_text(), id="help-overlay", classes="hidden")

    async def on_mount(self) -> None:
        self.query_one("#prompt-input", TextArea).placeholder = (
            "Enter a prompt and press Ctrl+Enter to send..."
        )
        if not self.compare:
            self.query_one("#panel-b", ResponsePanel).display = False
        self.set_interval(0.2, self._refresh_live_titles)
        await self._refresh_server_state()
        self._render_header()
        self._render_model_selector()
        self._set_status("Ready", kind="success")

    async def _refresh_server_state(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                health = await client.get(f"{self.base_url}/health")
                self.healthy = (
                    health.status_code == 200
                    and health.json().get("status") == "healthy"
                )
                models = await client.get(f"{self.base_url}/v1/models")
                if models.status_code == 200:
                    ids = [
                        item["id"]
                        for item in models.json().get("data", [])
                        if isinstance(item, dict) and isinstance(item.get("id"), str)
                    ]
                    if self.compare:
                        if ids:
                            missing = [
                                m for m in self.compare if m not in ids
                            ]
                            if missing:
                                self.healthy = False
                        self.models = list(self.compare)
                        self.current_model_index = 0
                        return
                    if ids:
                        selected_model = self._active_model()
                        self.models = ids
                        if selected_model in ids:
                            self.current_model_index = ids.index(selected_model)
                        else:
                            self.current_model_index = 0
        except (httpx.HTTPError, ValueError):
            self.healthy = False

    def action_toggle_help(self) -> None:
        help_overlay = self.query_one("#help-overlay", Static)
        help_overlay.toggle_class("hidden")

    def action_clear_responses(self) -> None:
        self.query_one("#panel-a", ResponsePanel).clear_response()
        self.query_one("#panel-b", ResponsePanel).clear_response()
        self._set_status("Cleared", kind="success")

    def action_cycle_model(self) -> None:
        if self.compare or self.in_flight or not self.models:
            return
        self.current_model_index = (self.current_model_index + 1) % len(self.models)
        panel = self.query_one("#panel-a", ResponsePanel)
        panel.model = self.models[self.current_model_index]
        panel.refresh_title()
        self._render_model_selector()

    def action_submit_prompt(self) -> None:
        if self.in_flight:
            return
        prompt_input = self.query_one("#prompt-input", TextArea)
        prompt = prompt_input.text.strip()
        if not prompt:
            self._set_status("Enter a prompt first", kind="error")
            return
        self.run_worker(self._submit_prompt(prompt), exclusive=True)

    async def _submit_prompt(self, prompt: str) -> None:
        self.in_flight = True
        prompt_input = self.query_one("#prompt-input", TextArea)
        prompt_input.disabled = True
        self._set_status("Streaming...", kind="progress")

        await self._refresh_server_state()
        self._render_header()

        panel_a = self.query_one("#panel-a", ResponsePanel)
        panel_b = self.query_one("#panel-b", ResponsePanel)
        panel_a.model = self._active_model()
        if self.compare:
            panel_b.model = self.compare[1]

        try:
            if self.compare:
                await asyncio.gather(
                    self._stream_model(self.compare[0], prompt, panel_a),
                    self._stream_model(self.compare[1], prompt, panel_b),
                )
            else:
                await self._stream_model(self._active_model(), prompt, panel_a)

            total_tokens = sum(
                len(panel.content.split())
                for panel in (panel_a, panel_b)
                if panel.display is not False
            )
            elapsed = max(panel_a.elapsed_seconds(), panel_b.elapsed_seconds())
            self._set_status(
                f"Done - {total_tokens} tokens in {elapsed:.1f}s",
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

    def _active_model(self) -> str:
        if self.compare:
            return self.compare[0]
        if not self.models:
            return DEFAULT_MODEL
        if self.current_model_index >= len(self.models):
            self.current_model_index = 0
        return self.models[self.current_model_index]

    def _refresh_live_titles(self) -> None:
        if not self.in_flight:
            return
        self.query_one("#panel-a", ResponsePanel).refresh_title()
        self.query_one("#panel-b", ResponsePanel).refresh_title()

    def _render_header(self) -> None:
        dot = "●" if self.healthy else "●"
        state = "healthy" if self.healthy else "unreachable"
        self.query_one("#header-bar", Static).update(
            f"InferenceX Playground    {self.base_url} {dot} {state}"
        )
        self.query_one("#header-bar", Static).set_class(self.healthy, "healthy")

    def _render_model_selector(self) -> None:
        if self.compare:
            text = f"{self.compare[0]} vs {self.compare[1]}"
        else:
            text = self._active_model()
        self.query_one("#model-selector", Static).update(text)

    def _set_status(self, message: str, *, kind: str) -> None:
        status = self.query_one("#status-bar", Static)
        status.update(message)
        status.remove_class("success", "error", "progress")
        status.add_class(kind)

    @staticmethod
    def _help_text() -> str:
        return (
            "Shortcuts\n"
            "Ctrl+Enter  submit prompt\n"
            "Enter       add newline\n"
            "Ctrl+M/Tab  cycle model\n"
            "Ctrl+L      clear panels\n"
            "F1          toggle help\n"
            "q/Ctrl+C    quit"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InferenceX Textual playground")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--compare", nargs=2, metavar=("MODEL_A", "MODEL_B"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    compare = tuple(args.compare) if args.compare else None
    app = InferenceXApp(base_url=args.base_url, model=args.model, compare=compare)
    app.run()


if __name__ == "__main__":
    main()
