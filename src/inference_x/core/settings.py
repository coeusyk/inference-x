from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class AppSettings:
    """Application settings sourced from environment variables and YAML config files.

    Environment variables take precedence over file defaults.
    All env vars use the prefix INFERENCE_X_.

    Key vars:
        INFERENCE_X_DEFAULT_MODEL  — which model entry in models.yaml to load (default: qwen2.5-0.5b)
        INFERENCE_X_CONFIG_DIR     — directory for config files (default: config)
    """

    def __init__(self) -> None:
        self.default_model: str = os.environ.get(
            "INFERENCE_X_DEFAULT_MODEL", "qwen2.5-0.5b"
        )
        self.config_dir: str = os.environ.get("INFERENCE_X_CONFIG_DIR", "config")

    def get_model_config(self, model_name: str | None = None) -> dict[str, Any]:
        """Return the config block for *model_name* from models.yaml.

        Falls back to *default_model* when *model_name* is None.

        Raises:
            FileNotFoundError: models.yaml does not exist.
            ValueError: requested model is not listed.
        """
        name = model_name or self.default_model
        path = Path(self.config_dir) / "models.yaml"

        if not path.exists():
            raise FileNotFoundError(f"Model config file not found: {path}")

        with path.open() as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}

        for entry in data.get("models", []):
            if entry.get("name") == name:
                return entry

        available = [m.get("name") for m in data.get("models", [])]
        raise ValueError(
            f"Model '{name}' not found in {path}. Available: {available}"
        )

    def get_server_config(self) -> dict[str, Any]:
        """Return the server section from server.yaml, or empty dict if absent."""
        path = Path(self.config_dir) / "server.yaml"
        if not path.exists():
            return {}
        with path.open() as fh:
            data = yaml.safe_load(fh) or {}
        return data.get("server", {})


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the cached application settings singleton."""
    return AppSettings()
