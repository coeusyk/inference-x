"""Model selection startup screen for the InferenceX Textual playground."""
from __future__ import annotations

import httpx
from textual.app import ComposeResult
from textual.containers import Center, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static


class ModelSelectScreen(ModalScreen[str]):
    """Modal that fetches /v1/models and lets the operator pick one.

    Dismisses with the selected model name string.
    Exits the app if the user cancels.
    """

    DEFAULT_CSS = """
    ModelSelectScreen {
        align: center middle;
    }

    #select-panel {
        width: 62;
        height: auto;
        border: round #596275;
        background: #161a23;
        padding: 1 2;
    }

    #select-title {
        text-style: bold;
        color: #7aa884;
        width: 1fr;
        content-align: center middle;
        padding-bottom: 1;
    }

    #select-subtitle {
        color: #7f8795;
        width: 1fr;
        padding-bottom: 1;
    }

    #select-error {
        color: #d06c75;
        padding-bottom: 1;
    }

    #model-radio {
        border: none;
        padding: 0;
        background: transparent;
    }

    #model-input {
        margin-top: 1;
    }

    #select-actions {
        margin-top: 1;
        align: right middle;
    }

    #select-actions Button {
        margin-left: 1;
    }

    .hidden {
        display: none;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, base_url: str, initial_model: str | None = None) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.initial_model = initial_model
        self._models: list[str] = []
        self._selected_index: int = 0

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="select-panel"):
                yield Label("InferenceX Playground", id="select-title")
                yield Label(self.base_url, id="select-subtitle")
                yield RadioSet(id="model-radio")
                yield Static("Fetching available models…", id="select-loading")
                yield Label("", id="select-error", classes="hidden")
                yield Input(
                    placeholder="Enter model name (e.g. qwen2.5-0.5b)",
                    id="model-input",
                    classes="hidden",
                )
                with Horizontal(id="select-actions"):
                    yield Button("Connect", variant="primary", id="btn-confirm")
                    yield Button("Cancel", id="btn-cancel")

    async def on_mount(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(f"{self.base_url}/v1/models")
                resp.raise_for_status()
                data = resp.json()
                models = [
                    item["id"]
                    for item in data.get("data", [])
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                ]
        except Exception as exc:
            self._show_fallback(f"Could not reach server: {exc}")
            return

        if not models:
            self._show_fallback("Server returned no models.")
            return

        self._models = models
        loading = self.query_one("#select-loading", Static)
        loading.display = False

        radio = self.query_one("#model-radio", RadioSet)
        for model in models:
            await radio.mount(RadioButton(model))

        # Pre-select initial_model if provided, otherwise first entry.
        if self.initial_model and self.initial_model in models:
            self._selected_index = models.index(self.initial_model)
        else:
            self._selected_index = 0

        radio.focus()

    def _show_fallback(self, error: str) -> None:
        try:
            self.query_one("#select-loading", Static).display = False
            err = self.query_one("#select-error", Label)
            err.update(error)
            err.remove_class("hidden")
            self.query_one("#model-input", Input).remove_class("hidden")
            self.query_one("#model-input", Input).focus()
        except Exception:
            pass

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        self._selected_index = event.index

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-confirm":
            self._confirm()
        elif event.button.id == "btn-cancel":
            self.action_cancel()

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        self._confirm()

    def _confirm(self) -> None:
        if self._models and self._selected_index < len(self._models):
            self.dismiss(self._models[self._selected_index])
            return
        manual = self.query_one("#model-input", Input)
        if manual.value.strip():
            self.dismiss(manual.value.strip())

    def action_cancel(self) -> None:
        self.app.exit()
