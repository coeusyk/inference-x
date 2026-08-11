from typing import Annotated

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from inference_x.api.deps import get_chat_service
from inference_x.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from inference_x.services.chat_service import ChatService

router = APIRouter()


@router.post(
    "/chat/completions",
    response_model=None,
    summary="Create a chat completion",
)
async def chat_completions(
    request: ChatCompletionRequest,
    response: Response,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatCompletionResponse | StreamingResponse:
    if request.stream is True:
        return StreamingResponse(
            service.stream_response(request),
            media_type="text/event-stream",
        )

    result = await service.complete(request)
    # Phase C, C1 (add-run-manifest): non-streaming only — the run_id is not
    # known until generation finishes, and the streaming path's single
    # pre-generation event cannot carry a not-yet-determined fact (DEC-053).
    if result.run_id:
        response.headers["X-Run-Id"] = result.run_id
    return result
