"""Multi-model engine pool for serving multiple models from a single process."""

from __future__ import annotations

import logging

from inference_x.engines.base import BaseEngine

logger = logging.getLogger(__name__)


class EnginePool:
    """Holds a named set of loaded engines and dispatches by model name.

    A pool with a single entry is equivalent to the old single-engine setup.
    A pool with multiple entries lets one server process serve several models
    simultaneously (subject to GPU memory).
    """

    def __init__(self, engines: dict[str, BaseEngine]) -> None:
        if not engines:
            raise ValueError("EnginePool requires at least one engine")
        self._engines: dict[str, BaseEngine] = engines

    def get(self, name: str) -> BaseEngine:
        """Return the engine for *name*.

        Raises:
            ValueError: if *name* is not in the pool.
        """
        if name not in self._engines:
            raise ValueError(
                f"Model '{name}' is not loaded. "
                f"Loaded models: {self.loaded_models()}. "
                f"Set INFERENCE_X_LOADED_MODELS to include it."
            )
        return self._engines[name]

    def loaded_models(self) -> list[str]:
        """Return names of all models in the pool (insertion order)."""
        return list(self._engines)

    def all_healthy(self) -> bool:
        """Return True only when every engine in the pool is healthy."""
        return all(e.is_healthy() for e in self._engines.values())

    def health_status(self) -> dict[str, str]:
        """Return a per-model health map ``{name: "ok" | "unavailable"}``."""
        return {
            name: "ok" if engine.is_healthy() else "unavailable"
            for name, engine in self._engines.items()
        }

    def shutdown(self) -> None:
        """Shut down every loaded engine and release subprocess resources."""
        for name, engine in self._engines.items():
            shutdown_fn = getattr(engine, "shutdown", None)
            if not callable(shutdown_fn):
                continue
            try:
                shutdown_fn()
            except Exception as exc:
                logger.warning("Failed to shut down engine %s: %s", name, exc)
