"""Shared rich Console instance and styled components for the playground client.

Design palette
--------------
- Model names: bold, prominent
- Latency / token counts: dim (secondary)
- Response text: bright_white (readable at a glance)
- Borders: dim blue (left/single) · dim cyan (right/compare)
- Errors: red bordered panel to stderr
- Header / prompts: dim
"""
from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

_THEME = Theme(
    {
        "header": "dim",
        "prompt.label": "dim bold",
        "model.name": "bold",
        "model.latency": "dim",
        "response.text": "bright_white",
        "token.info": "dim",
    }
)


def stdout_console() -> Console:
    """Return a Console targeting the current sys.stdout.

    Created fresh on each call so tests that patch sys.stdout get the
    patched target (Rich captures the file reference at Console init time).
    """
    return Console(file=sys.stdout, theme=_THEME, highlight=False)


def stderr_console() -> Console:
    """Return a Console targeting the current sys.stderr."""
    return Console(file=sys.stderr, theme=_THEME, highlight=False)


# ---------------------------------------------------------------------------
# Simple top-level print helpers (used by main())
# ---------------------------------------------------------------------------

def print_header(base_url: str) -> None:
    """One-line muted header; shown for prompt modes, not --health/--list-models."""
    stdout_console().print(
        f"[header]InferenceX Playground  ·  {base_url}[/header]"
    )


def print_health(ok: bool, base_url: str) -> None:
    """Single styled health indicator line."""
    c = stdout_console()
    if ok:
        c.print(f"[bold green]● healthy[/bold green]  [dim]{base_url}[/dim]")
    else:
        c.print(f"[bold red]● unreachable[/bold red]  [dim]{base_url}[/dim]")


def print_models_table(models: list[str]) -> None:
    """Render an 'Available Models' table with alternating dim rows."""
    c = stdout_console()
    table = Table(
        title="Available Models",
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 1),
        show_edge=False,
    )
    table.add_column("Name")
    for i, name in enumerate(models):
        table.add_row(name, style="dim" if i % 2 == 0 else "")
    c.print(table)


def print_error(message: str) -> None:
    """Render a bordered error panel to stderr; never plain print() for user errors."""
    stderr_console().print(
        Panel(message, title="[bold red]Error[/bold red]", border_style="red")
    )
