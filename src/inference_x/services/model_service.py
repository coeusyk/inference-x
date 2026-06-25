from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from inference_x.schemas.model import ModelEntry


class ModelRegistry:
    """Loads and indexes model definitions from models.yaml.

    This is the single source of truth for which models are configured.
    It does not own engines — it only knows what models are declared.
    The engine layer is responsible for instantiation.
    """

    def __init__(self, models: list[ModelEntry]) -> None:
        self._models: dict[str, ModelEntry] = {m.name: m for m in models}

    @classmethod
    def from_config(cls, config_dir: str = "config") -> "ModelRegistry":
        """Load model definitions from models.yaml.

        Raises:
            FileNotFoundError: if models.yaml is absent.
            ValueError: if any model entry fails validation.
        """
        path = Path(config_dir) / "models.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Model config not found: {path}")

        with path.open() as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}

        entries = [ModelEntry(**m) for m in data.get("models", [])]
        return cls(entries)

    def get(self, name: str) -> ModelEntry:
        """Return a named model entry.

        Raises:
            ValueError: if the model is not registered.
        """
        if name not in self._models:
            available = list(self._models)
            raise ValueError(
                f"Model '{name}' is not registered. Available: {available}"
            )
        return self._models[name]

    def all(self) -> list[ModelEntry]:
        """Return all registered model entries in insertion order."""
        return list(self._models.values())

    def names(self) -> list[str]:
        return list(self._models)

    def __contains__(self, name: str) -> bool:
        return name in self._models
