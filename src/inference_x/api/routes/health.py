from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from inference_x.api.deps import get_chat_service
from inference_x.services.chat_service import ChatService

router = APIRouter()


@router.get("/health", summary="Health check")
def health(
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> JSONResponse:
    healthy = service.engine_healthy()
    body = {
        "status": "healthy" if healthy else "degraded",
        "engine": "ok" if healthy else "unavailable",
    }
    return JSONResponse(status_code=200 if healthy else 503, content=body)
