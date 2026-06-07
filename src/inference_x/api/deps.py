from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from inference_x.core.settings import AppSettings, get_settings
from inference_x.engines.base import BaseEngine
from inference_x.engines.vllm_engine import VLLMEngine
from inference_x.services.chat_service import ChatService


@lru_cache(maxsize=1)
def _build_engine(settings: AppSettings) -> BaseEngine:
    """Construct and cache the engine singleton for the lifetime of the process."""
    model_config = settings.get_model_config()
    return VLLMEngine(model_config)


def get_engine(settings: Annotated[AppSettings, Depends(get_settings)]) -> BaseEngine:
    return _build_engine(settings)


def get_chat_service(
    engine: Annotated[BaseEngine, Depends(get_engine)],
) -> ChatService:
    return ChatService(engine)
