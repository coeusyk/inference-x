"""Model selection startup screen for the InferenceX Textual playground."""
from __future__ import annotations

import httpx
from textual.app import ComposeResult
from textual.containers import Center, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static

try:
    from model_registry import list_registered_models
except ImportError:
    from playground.model_registry import list_registered_models


class ModelSelectScreen(ModalScreen[str | tuple[str, str]]):
    """Modal for picking one model (chat) or two models (compare).

    Dismisses with a model name string or ``(model_a, model_b)`` tuple.
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

    ModelSelectScreen.compare #select-panel {
        width: 70;
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

    #model-radio,
    #model-a-radio,
    #model-b-radio {
        border: none;
        padding: 0;
        background: transparent;
    }

    .compare-label {
        color: #7aa884;
        margin-top: 1;
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

    def __init__(
        self,
        base_url: str,
        initial_model: str | None = None,
        *,
        mode: str = "single",
        initial_compare: tuple[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.initial_model = initial_model
        self.mode = mode
        self.initial_compare = initial_compare
        self._models: list[str] = []
        self._selected_index: int = 0

    def compose(self) -> ComposeResult:
        title = "InferenceX Compare" if self.mode == "compare" else "InferenceX Chat"
        if self.mode == "compare":
            self.add_class("compare")
        with Center():
            with Vertical(id="select-panel"):
                yield Label(title, id="select-title")
                subtitle = (
                    "Select two models to compare"
                    if self.mode == "compare"
                    else self.base_url
                )
                yield Label(subtitle, id="select-subtitle")
                if self.mode == "compare":
                    yield Label("Model A", classes="compare-label")
                    yield RadioSet(id="model-a-radio")
                    yield Label("Model B", classes="compare-label")
                    yield RadioSet(id="model-b-radio")
                else:
                    yield RadioSet(id="model-radio")
                yield Static("Fetching available models…", id="select-loading")
                yield Label("", id="select-error", classes="hidden")
                yield Input(
                    placeholder="Enter model name (e.g. qwen2.5-0.5b)",
                    id="model-input",
                    classes="hidden",
                )
                with Horizontal(id="select-actions"):
                    yield Button("Continue", variant="primary", id="btn-confirm")
                    yield Button("Cancel", id="btn-cancel")

    async def on_mount(self) -> None:
        models = list_registered_models()
        server_ok = False
        if models:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{self.base_url}/health")
                    server_ok = resp.status_code == 200
            except Exception:
                server_ok = False
        else:
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
                    server_ok = True
            except Exception as exc:
                self._show_fallback(f"Could not load models: {exc}")
                return

        if not models:
            self._show_fallback("No models found in config/models.yaml.")
            return

        subtitle = self.query_one("#select-subtitle", Label)
        if self.mode == "compare":
            if server_ok:
                subtitle.update(f"{self.base_url}  ·  server running")
            else:
                subtitle.update(
                    f"{self.base_url}  ·  server will start after you choose models"
                )
        elif server_ok:
            subtitle.update(f"{self.base_url}  ·  server running")
        else:
            subtitle.update(
                f"{self.base_url}  ·  server will start after you choose a model"
            )

        self._models = models
        loading = self.query_one("#select-loading", Static)
        loading.display = False

        if self.mode == "compare":
            if len(models) < 2:
                self._show_fallback("Need at least two models in config/models.yaml.")
                return
            radio_a = self.query_one("#model-a-radio", RadioSet)
            radio_b = self.query_one("#model-b-radio", RadioSet)
            for model in models:
                await radio_a.mount(RadioButton(model))
                await radio_b.mount(RadioButton(model))
            idx_a, idx_b = 0, min(1, len(models) - 1)
            if self.initial_compare:
                a, b = self.initial_compare
                if a in models and b in models and a != b:
                    idx_a = models.index(a)
                    idx_b = models.index(b)
            self._select_radio(radio_a, idx_a)
            self._select_radio(radio_b, idx_b)
            radio_a.focus()
            return

        radio = self.query_one("#model-radio", RadioSet)
        for model in models:
            await radio.mount(RadioButton(model))

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

    def _select_radio(self, radio: RadioSet, index: int) -> None:
        for i, node in enumerate(radio.children):
            if isinstance(node, RadioButton):
                node.value = i == index

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "model-radio":
            self._selected_index = event.index

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-confirm":
            self._confirm()
        elif event.button.id == "btn-cancel":
            self.action_cancel()

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        self._confirm()

    def _confirm(self) -> None:
        if self.mode == "compare":
            if len(self._models) < 2:
                return
            radio_a = self.query_one("#model-a-radio", RadioSet)
            radio_b = self.query_one("#model-b-radio", RadioSet)
            idx_a = radio_a.pressed_index
            idx_b = radio_b.pressed_index
            if idx_a < 0 or idx_b < 0:
                return
            model_a = self._models[idx_a]
            model_b = self._models[idx_b]
            if model_a == model_b:
                err = self.query_one("#select-error", Label)
                err.update("Choose two different models.")
                err.remove_class("hidden")
                return
            self.dismiss((model_a, model_b))
            return

        if self._models:
            radio = self.query_one("#model-radio", RadioSet)
            idx = radio.pressed_index
            if 0 <= idx < len(self._models):
                self.dismiss(self._models[idx])
                return
        manual = self.query_one("#model-input", Input)
        if manual.value.strip():
            self.dismiss(manual.value.strip())

    def action_cancel(self) -> None:
        self.app.exit()
