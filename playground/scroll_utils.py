"""Auto-scroll helpers for Textual scroll containers."""
from __future__ import annotations

from textual.containers import ScrollableContainer


def scroll_to_end(container: ScrollableContainer) -> None:
    """Scroll *container* to the latest content without animation."""
    try:
        container.scroll_end(animate=False)
    except Exception:
        pass
