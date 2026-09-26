from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from inference_x.api.deps import get_chat_service
from inference_x.schemas.chat import ChatCompletionRequest
from inference_x.services.chat_service import ChatService

router = APIRouter()


@router.post(
    "/chat/completions",
    response_model=None,
    summary="Create a chat completion",
)
async def chat_completions(
    request: ChatCompletionRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> JSONResponse | StreamingResponse:
    if request.stream is True:
        return StreamingResponse(
            service.stream_response(request),
            media_type="text/event-stream",
        )

    result = await service.complete(request)
    # Phase C, C1 (add-run-manifest): non-streaming only — the run_id is not
    # known until generation finishes, and the streaming path's single
    # pre-generation event cannot carry a not-yet-determined fact (DEC-053).
    headers = {"X-Run-Id": result.run_id} if result.run_id else None
    # V1-0: the default body has no `manifest` key at all (not `null`); other
    # None-valued fields keep their existing `null` serialization.
    exclude = {"manifest"} if result.manifest is None else None
    return JSONResponse(content=result.model_dump(mode="json", exclude=exclude), headers=headers)
