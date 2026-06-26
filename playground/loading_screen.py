"""Full-screen loading overlay while the inference server starts."""
from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, RichLog, Rule, Static

try:
    from scroll_utils import scroll_to_end
except ImportError:
    from playground.scroll_utils import scroll_to_end

_STEP_NAMES = ("Server", "Model", "Ready")


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

    LoadingScreen.failed #loading-hint {
        color: #c9d1e3;
    }

    #loading-panel {
        width: 80%;
        min-width: 60;
        max-width: 90;
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

    #loading-steps {
        width: 1fr;
        height: 1;
        content-align: center middle;
        padding-bottom: 1;
    }

    #loading-steps .step-sep {
        width: auto;
        color: #343b49;
        padding: 0 1;
    }

    #error-banner {
        display: none;
        color: #d06c75;
        width: 1fr;
        content-align: center middle;
        padding: 0 1 1 1;
        text-style: bold;
    }

    LoadingScreen.failed #error-banner {
        display: block;
    }

    #loading-log-rule {
        height: 1;
        margin: 0 0 1 0;
        color: #343b49;
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
        self._current_step = 0

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="loading-panel"):
                yield Label("Loading", id="loading-title")
                yield Static(self._message, id="loading-message")
                with Horizontal(id="loading-steps"):
                    yield Label("[ Server ]", id="step-server")
                    yield Static("────", classes="step-sep")
                    yield Label("[ Model ]", id="step-model")
                    yield Static("────", classes="step-sep")
                    yield Label("[ Ready ]", id="step-ready")
                yield Static("", id="error-banner")
                yield Rule(id="loading-log-rule")
                with VerticalScroll(id="loading-log-scroll"):
                    yield RichLog(
                        id="loading-log",
                        markup=True,
                        highlight=False,
                        wrap=True,
                    )
                yield Static(
                    "Live server output · full log: logs/playground-server.log",
                    id="loading-hint",
                )

    def on_mount(self) -> None:
        try:
            log = self.query_one("#loading-log", RichLog)
            log.write("[dim]Waiting for server output…[/dim]")
        except Exception:
            pass
        self.set_message(self._message)

    def set_message(self, message: str) -> None:
        self._message = message
        title, step = self._phase_from_message(message)
        try:
            self.query_one("#loading-message", Static).update(message)
            self.query_one("#loading-title", Label).update(title)
            self.set_step(step)
        except Exception:
            pass

    @staticmethod
    def _phase_from_message(message: str) -> tuple[str, int]:
        lower = message.lower()
        if "failed" in lower:
            return "Startup failed", 0

        if "starting" in lower:
            return "Starting server", 0

        if "waiting for server" in lower or "server to respond" in lower:
            return "Waiting for server", 0

        if lower.startswith("loading ") or (
            lower.startswith("waiting for ") and "server" not in lower
        ):
            return "Loading model", 1

        return "Loading", 0

    def set_step(self, step: int, *, error: bool = False) -> None:
        self._current_step = step
        step_ids = ("step-server", "step-model", "step-ready")
        try:
            for idx, step_id in enumerate(step_ids):
                label = self.query_one(f"#{step_id}", Label)
                name = _STEP_NAMES[idx]
                if error and idx == step:
                    label.update(f"[#d06c75][ {name} ]")
                elif idx < step:
                    label.update(f"[#7aa884]✓ [ {name} ][/]")
                elif idx == step:
                    label.update(f"[bold #7aa884][ {name} ]")
                else:
                    label.update(f"[#596275][ {name} ]")
        except Exception:
            pass

    def append_log(self, line: str) -> None:
        try:
            log = self.query_one("#loading-log", RichLog)
            if not self._has_logs:
                log.clear()
                self._has_logs = True
            if line.startswith("✗"):
                log.write(f"[#d06c75]{line}[/]")
            else:
                log.write(f"[#7f8795]{line}[/]")
            scroll_to_end(self.query_one("#loading-log-scroll", VerticalScroll))
        except Exception:
            pass

    def set_error(self, message: str) -> None:
        self._failed = True
        self.add_class("failed")
        had_real_logs = self._has_logs
        try:
            self.query_one("#loading-title", Label).update("Startup failed")
            self.query_one("#loading-message", Static).update(message)
            self.query_one("#error-banner", Static).update(message)
            self.query_one("#loading-steps", Horizontal).display = False
            hint = self.query_one("#loading-hint", Static)
            hint.update(
                "Press Escape to close  ·  full log: logs/playground-server.log"
            )
            self.set_step(self._current_step, error=True)
        except Exception:
            pass

        try:
            log_scroll = self.query_one("#loading-log-scroll", VerticalScroll)
            log_scroll.display = had_real_logs
        except Exception:
            pass

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
