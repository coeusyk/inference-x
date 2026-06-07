from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from inference_x.core.settings import AppSettings, get_settings
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
