"""Full-screen loading overlay while the inference server starts."""
from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, RichLog, Static


class LoadingScreen(ModalScreen[None]):
    """Blocking overlay with status, live log feed, and graceful errors."""

    DEFAULT_CSS = """
    LoadingScreen {
        align: center middle;
        background: #0f1117 90%;
    }

    LoadingScreen.failed #loading-panel {
        border: round #d06c75;
    }

    LoadingScreen.failed #loading-title {
        color: #d06c75;
    }

    #loading-panel {
        width: 74;
        height: auto;
        max-height: 90%;
        border: round #596275;
        background: #161a23;
        padding: 1 2;
    }

    #loading-title {
        text-style: bold;
        color: #7aa884;
        width: 1fr;
        content-align: center middle;
        padding-bottom: 1;
    }

    #loading-message {
        color: #c9d1e3;
        width: 1fr;
        content-align: center middle;
        padding: 0 1 1 1;
    }

    #loading-log-scroll {
        height: 10;
        width: 1fr;
        border: round #343b49;
        background: #0f1117;
        margin-bottom: 1;
    }

    #loading-log {
        width: 1fr;
        height: auto;
        padding: 0 1;
    }

    #loading-hint {
        color: #7f8795;
        width: 1fr;
        content-align: center middle;
        padding-top: 0;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
    ]

    def __init__(self, message: str = "Preparing server…") -> None:
        super().__init__()
        self._message = message
        self._failed = False
        self._ack = asyncio.Event()
        self._has_logs = False

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="loading-panel"):
                yield Label("Loading", id="loading-title")
                yield Static(self._message, id="loading-message")
                with VerticalScroll(id="loading-log-scroll"):
                    yield RichLog(id="loading-log", markup=False, highlight=False, wrap=True)
                yield Static(
                    "Live server output · full log: logs/playground-server.log",
                    id="loading-hint",
                )

    def on_mount(self) -> None:
        self.append_log("Waiting for server output…")

    def set_message(self, message: str) -> None:
        self._message = message
        try:
            self.query_one("#loading-message", Static).update(message)
        except Exception:
            pass

    def append_log(self, line: str) -> None:
        try:
            log = self.query_one("#loading-log", RichLog)
            if line == "Waiting for server output…" and self._has_logs:
                return
            log.write(line)
            self._has_logs = True
        except Exception:
            pass

    def set_error(self, message: str) -> None:
        self._failed = True
        self.add_class("failed")
        try:
            self.query_one("#loading-title", Label).update("Failed to load")
            self.query_one("#loading-message", Static).update(message)
            hint = self.query_one("#loading-hint", Static)
            hint.update("Press Escape to close · see logs/playground-server.log for details")
        except Exception:
            pass
        self.append_log(f"✗ {message[:66]}")

    def finish_failed(self, message: str) -> None:
        """Mark the screen as failed and allow the user to dismiss it."""
        self.set_error(message)

    async def wait_for_ack(self) -> None:
        """Block until the user dismisses a failed loading screen."""
        if not self._failed:
            return
        await self._ack.wait()

    def action_close(self) -> None:
        self._ack.set()
        self.dismiss(None)
