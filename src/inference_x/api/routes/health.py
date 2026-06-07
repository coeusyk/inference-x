from typing import Annotated

from fastapi import APIRouter, Depends

from inference_x.api.deps import get_chat_service
from inference_x.services.chat_service import ChatService

router = APIRouter()


@router.get("/health", summary="Health check")
def health(
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> dict:
    healthy = service.engine_healthy()
    return {
        "status": "healthy" if healthy else "degraded",
        "engine": "ok" if healthy else "unavailable",
    }
