from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from inference_x.core.settings import AppSettings, get_settings
from inference_x.engines.pool import EnginePool
from inference_x.engines.vllm_engine import VLLMEngine
from inference_x.observability.exporters import build_exporter
from inference_x.observability.recorder import MetricsRecorder
from inference_x.observability.storage import InMemoryStorage
from inference_x.routing.task_router import TaskRouter
from inference_x.services.chat_service import ChatService
from inference_x.services.model_service import ModelRegistry
from inference_x.utils.vllm_pool_config import validate_pool_fits

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _build_registry(config_dir: str) -> ModelRegistry:
    return ModelRegistry.from_config(config_dir)


@lru_cache(maxsize=1)
def _build_engine_pool(config_dir: str, loaded_models: tuple[str, ...]) -> EnginePool:
    """Build an EnginePool loading one VLLMEngine per model in *loaded_models*."""
    registry = _build_registry(config_dir)
    pool_size = len(loaded_models)
    validate_pool_fits(list(loaded_models))
    engines: dict = {}
    for model_name in loaded_models:
        model_config = registry.get(model_name)
        logger.info("Loading engine for model=%s (pool_size=%d)", model_name, pool_size)
        engines[model_name] = VLLMEngine(
            model_config.model_dump(),
            pool_size=pool_size,
            pool_models=list(loaded_models),
        )
    return EnginePool(engines)


@lru_cache(maxsize=1)
def _build_router(config_dir: str, default_model: str) -> TaskRouter:
    registry = _build_registry(config_dir)
    return TaskRouter(registry, default_model)


def get_registry(
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> ModelRegistry:
    return _build_registry(settings.config_dir)


def get_chat_service(
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> ChatService:
    pool = _build_engine_pool(settings.config_dir, tuple(settings.loaded_models))
    registry = _build_registry(settings.config_dir)
    router = _build_router(settings.config_dir, settings.default_model)
    return ChatService(engine_pool=pool, registry=registry, router=router)


@lru_cache(maxsize=1)
def _build_recorder() -> MetricsRecorder:
    storage = InMemoryStorage()
    exporter = build_exporter()
    return MetricsRecorder(storage=storage, exporter=exporter)


def get_recorder() -> MetricsRecorder:
    """Return the process-level recorder singleton (not a FastAPI Depends)."""
    return _build_recorder()


def initialize_app() -> None:
    """Eagerly build registry, router, and engine pool at application startup.

    Raises on config or model-load failure so the process fails fast instead
    of accepting requests that would fail on first dependency resolution.
    """
    settings = get_settings()
    config_dir = settings.config_dir
    default_model = settings.default_model
    loaded_models = settings.loaded_models

    logger.info(
        "Initializing InferenceX (config_dir=%s, default_model=%s, loaded_models=%s)",
        config_dir,
        default_model,
        loaded_models,
    )

    registry = _build_registry(config_dir)
    logger.info("Model registry loaded: %s", registry.names())

    _build_router(config_dir, default_model)
    logger.info("Task router ready (default_model=%s)", default_model)

    pool = _build_engine_pool(config_dir, tuple(loaded_models))
    if not pool.all_healthy():
        unhealthy = [n for n, s in pool.health_status().items() if s != "ok"]
        raise RuntimeError(
            f"Engines not healthy after startup init: {unhealthy}"
        )

    logger.info("Engine pool ready: %s", pool.loaded_models())
