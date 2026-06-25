import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from inference_x.schemas.common import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)

_CLIENT_ERROR_MESSAGE = "Request could not be processed."


async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    logger.error("RuntimeError on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorDetail(message=_CLIENT_ERROR_MESSAGE, type="internal_error")
        ).model_dump(),
    )


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.error("ValueError on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=ErrorDetail(
                message=_CLIENT_ERROR_MESSAGE, type="invalid_request_error"
            )
        ).model_dump(),
    )
