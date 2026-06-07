from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from inference_x.api.deps import get_chat_service
from inference_x.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from inference_x.services.chat_service import ChatService

router = APIRouter()


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    summary="Create a chat completion",
)
async def chat_completions(
    request: ChatCompletionRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatCompletionResponse:
    return await service.complete(request)
