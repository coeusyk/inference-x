from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from inference_x.core.settings import AppSettings, get_settings
from inference_x.observability.exporters import build_exporter
from inference_x.observability.recorder import MetricsRecorder
from inference_x.observability.storage import InMemoryStorage

logger = logging.getLogger(__name__)
from inference_x.engines.base import BaseEngine
from inference_x.engines.vllm_engine import VLLMEngine
from inference_x.routing.task_router import TaskRouter
from inference_x.services.chat_service import ChatService
from inference_x.services.model_service import ModelRegistry


@lru_cache(maxsize=1)
def _build_registry(config_dir: str) -> ModelRegistry:
    return ModelRegistry.from_config(config_dir)


@lru_cache(maxsize=1)
def _build_engine(config_dir: str, default_model: str) -> BaseEngine:
    registry = _build_registry(config_dir)
    model_config = registry.get(default_model)
    return VLLMEngine(model_config.model_dump())


@lru_cache(maxsize=1)
def _build_router(config_dir: str, default_model: str) -> TaskRouter:
    registry = _build_registry(config_dir)
    return TaskRouter(registry, default_model)


def get_registry(
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> ModelRegistry:
    return _build_registry(settings.config_dir)


def get_engine(
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> BaseEngine:
    return _build_engine(settings.config_dir, settings.default_model)


def get_chat_service(
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> ChatService:
    registry = _build_registry(settings.config_dir)
    engine = _build_engine(settings.config_dir, settings.default_model)
    router = _build_router(settings.config_dir, settings.default_model)
    return ChatService(
        engine=engine,
        registry=registry,
        router=router,
        loaded_model=settings.default_model,
    )


@lru_cache(maxsize=1)
def _build_recorder() -> MetricsRecorder:
    storage = InMemoryStorage()
    exporter = build_exporter()
    return MetricsRecorder(storage=storage, exporter=exporter)


def get_recorder() -> MetricsRecorder:
    """Return the process-level recorder singleton (not a FastAPI Depends)."""
    return _build_recorder()


def initialize_app() -> None:
    """Eagerly build registry, router, and engine at application startup.

    Raises on config or model-load failure so the process fails fast instead
    of accepting requests that would fail on first dependency resolution.
    """
    settings = get_settings()
    config_dir = settings.config_dir
    default_model = settings.default_model

    logger.info(
        "Initializing InferenceX (config_dir=%s, default_model=%s)",
        config_dir,
        default_model,
    )

    registry = _build_registry(config_dir)
    logger.info("Model registry loaded: %s", registry.names())

    _build_router(config_dir, default_model)
    logger.info("Task router ready (default_model=%s)", default_model)

    engine = _build_engine(config_dir, default_model)
    if not engine.is_healthy():
        raise RuntimeError(
            f"Engine for model '{default_model}' is not healthy after startup init"
        )

    logger.info("Engine loaded and healthy for model=%s", default_model)
