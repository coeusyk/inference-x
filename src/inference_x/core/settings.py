from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_project_env() -> None:
    """Load repo-root ``.env`` into ``os.environ`` (existing shell vars win)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    for candidate in (_repo_root() / ".env", Path.cwd() / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


load_project_env()


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

        # Comma-separated list of models to load at startup.
        # Falls back to default_model when not set.
        # Example: INFERENCE_X_LOADED_MODELS=qwen2.5-0.5b,tinyllama-chat
        _raw = os.environ.get("INFERENCE_X_LOADED_MODELS", "")
        self.loaded_models: list[str] = (
            [m.strip() for m in _raw.split(",") if m.strip()]
            if _raw.strip()
            else [self.default_model]
        )

        # Maximum seconds to wait between tokens during streaming.
        # If no token is produced within this window the stream is terminated
        # with an error SSE event.  Set to 0 to disable.
        self.stream_timeout_s: float = float(
            os.environ.get("INFERENCE_X_STREAM_TIMEOUT_S", "120")
        )

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

    def get_vram_tier(self):
        """Resolve the VRAM tier (config/vram_tiers.yaml) for the probed GPU.

        Uses profile_hardware() for vram_total_gb — the same nvidia-ml-py →
        nvidia-smi → CPU-only fallback chain the benchmark advisor relies on.
        Import is local to avoid pulling benchmarks.hardware into every settings
        consumer (e.g. lightweight unit tests that stub AppSettings directly).
        """
        from inference_x.benchmarks.hardware import profile_hardware
        from inference_x.utils.vram_tiers import load_tiers, resolve_tier

        hw = profile_hardware()
        tiers = load_tiers(self.config_dir)
        return resolve_tier(hw.vram_total_gb, tiers)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the cached application settings singleton."""
    return AppSettings()
