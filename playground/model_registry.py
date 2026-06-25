"""Read registered model names from config/models.yaml (no server required)."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent


def list_registered_models(config_dir: str | Path | None = None) -> list[str]:
    """Return model ``name`` values from ``config/models.yaml``."""
    base = Path(config_dir) if config_dir is not None else _REPO_ROOT / "config"
    path = base / "models.yaml"
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = data.get("models") or []
    names: list[str] = []
    for entry in models:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            names.append(entry["name"])
    return names
