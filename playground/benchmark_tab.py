"""Benchmark tab for the InferenceX Textual playground.

Shows hardware profile, per-model throughput, and advisor recommendations.
A "Run Benchmark" button launches scripts/benchmark.py for the selected
model as an asyncio subprocess.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import ClassVar

import httpx
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button, DataTable, Label, Select, Static


_DEFAULT_BASE_URL = "http://localhost:8000"


def _hw_text(hw: dict) -> str:
    if hw.get("has_gpu"):
        vram_free = hw.get("vram_free_gb", 0)
        vram_total = hw.get("vram_total_gb", 0)
        name = hw.get("gpu_name") or "GPU"
        return f"{name}  {vram_free:.1f} / {vram_total:.1f} GB VRAM  |  CPU: {hw.get('cpu_cores', '?')} cores  RAM: {hw.get('ram_total_gb', 0):.0f} GB"
    return f"CPU only  |  {hw.get('cpu_cores', '?')} cores  |  RAM: {hw.get('ram_total_gb', 0):.0f} GB"


class BenchmarkTab(Widget):
    """Benchmark results and advisor panel for the playground TUI."""

    DEFAULT_CSS: ClassVar[str] = """
    BenchmarkTab {
        padding: 1;
    }
    BenchmarkTab #hw-info {
        color: $text-muted;
        margin-bottom: 1;
    }
    BenchmarkTab #bench-status {
        margin: 1 0;
        color: $success;
    }
    BenchmarkTab #bench-error {
        margin: 1 0;
        color: $error;
    }
    BenchmarkTab #bench-controls {
        height: 3;
        margin-bottom: 1;
    }
    BenchmarkTab #bench-model-select {
        width: 1fr;
        margin-right: 1;
    }
    BenchmarkTab DataTable {
        margin-bottom: 1;
    }
    """

    _base_url: str = _DEFAULT_BASE_URL

    def __init__(self, base_url: str = _DEFAULT_BASE_URL, **kwargs) -> None:
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")
        self._running = False

    def compose(self) -> ComposeResult:
        yield Label("Hardware", id="hw-label", classes="section-title")
        yield Static("Detecting hardware…", id="hw-info")
        with Horizontal(id="bench-controls"):
            yield Select([], id="bench-model-select", prompt="Select model")
            yield Button("Run Benchmark", id="run-benchmark-btn", variant="primary")
        yield Label("Benchmark Results", id="results-label", classes="section-title")
        yield DataTable(id="results-table")
        yield Label("Advisor Ranking", id="advisor-label", classes="section-title")
        yield DataTable(id="advisor-table")
        yield Static("", id="bench-status")
        yield Static("", id="bench-error")

    def on_mount(self) -> None:
        self._setup_tables()
        self.run_worker(self._refresh_data(), exclusive=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "run-benchmark-btn":
            return
        select = self.query_one("#bench-model-select", Select)
        if select.value is Select.BLANK:
            self.query_one("#bench-error", Static).update("Select a model first.")
            return
        self.run_worker(self.run_benchmark(str(select.value)), exclusive=True)

    def _setup_tables(self) -> None:
        results_table = self.query_one("#results-table", DataTable)
        results_table.add_columns("Model", "Throughput (tok/s)", "p50 ms", "p95 ms", "VRAM GB")

        advisor_table = self.query_one("#advisor-table", DataTable)
        advisor_table.add_columns("Rank", "Model", "Score", "Viable", "Recommendation")

    async def _refresh_data(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                results_resp = await client.get(f"{self._base_url}/v1/benchmark/results")
                advise_resp = await client.get(f"{self._base_url}/v1/benchmark/advise")
                models_resp = await client.get(f"{self._base_url}/v1/models")

            if models_resp.status_code == 200:
                payload = models_resp.json()
                model_ids = [
                    item["id"]
                    for item in payload.get("data", [])
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                ]
                select = self.query_one("#bench-model-select", Select)
                current = select.value if select.value is not Select.BLANK else None
                select.set_options([(m, m) for m in model_ids])
                if current and current in model_ids:
                    select.value = current
                elif model_ids and select.value is Select.BLANK:
                    select.value = model_ids[0]

            if results_resp.status_code == 200:
                data = results_resp.json()
                hw = data.get("hardware", {})
                self.query_one("#hw-info", Static).update(_hw_text(hw))

                results_table = self.query_one("#results-table", DataTable)
                results_table.clear()
                for r in data.get("results", []):
                    results_table.add_row(
                        r["model_name"],
                        f"{r['mean_throughput_tps']:.1f}",
                        f"{r['p50_latency_ms']:.0f}",
                        f"{r['p95_latency_ms']:.0f}",
                        f"{r['peak_vram_delta_gb']:.2f}",
                    )

            if advise_resp.status_code == 200:
                data = advise_resp.json()
                advisor_table = self.query_one("#advisor-table", DataTable)
                advisor_table.clear()
                for i, r in enumerate(data.get("ranked", []), 1):
                    advisor_table.add_row(
                        str(i),
                        r["model_name"],
                        f"{r['score']:.1f}",
                        "YES" if r["viable"] else "NO",
                        r["recommendation_str"],
                    )

        except (httpx.HTTPError, KeyError, ValueError):
            self.query_one("#hw-info", Static).update("Server not reachable — start server first")

    async def run_benchmark(self, model_name: str) -> None:
        """Launch benchmark.py for *model_name* as a subprocess and refresh on completion."""
        if self._running:
            return
        self._running = True
        status = self.query_one("#bench-status", Static)
        error = self.query_one("#bench-error", Static)
        status.update(f"Running benchmark for {model_name}…")
        error.update("")
        self.query_one("#run-benchmark-btn", Button).disabled = True

        script = Path(__file__).parent.parent / "scripts" / "benchmark.py"
        cmd = [sys.executable, str(script), "--model", model_name, "--base-url", self._base_url]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                status.update(f"Benchmark for {model_name} complete.")
            else:
                error.update(
                    f"Benchmark failed (exit {proc.returncode}): "
                    f"{stderr.decode(errors='replace').strip()[:200]}"
                )
                status.update("")
            await self._refresh_data()
        except Exception as exc:
            error.update(f"Error launching benchmark: {exc}")
            status.update("")
        finally:
            self._running = False
            self.query_one("#run-benchmark-btn", Button).disabled = False
