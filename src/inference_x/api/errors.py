import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from inference_x.schemas.common import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    logger.error("RuntimeError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorDetail(message=str(exc), type="internal_error")
        ).model_dump(),
    )


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning("ValueError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=ErrorDetail(message=str(exc), type="invalid_request_error")
        ).model_dump(),
    )
