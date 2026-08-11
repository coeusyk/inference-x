import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from inference_x.routing.admission import EngineSaturatedError
from inference_x.schemas.common import ErrorDetail, ErrorResponse
from inference_x.utils.determinism import DeterminismUnsupportedError

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


async def determinism_unsupported_error_handler(
    request: Request, exc: DeterminismUnsupportedError
) -> JSONResponse:
    """`deterministic: true` refused (unsupported hardware or not started with it).

    design.md D4 (add-deterministic-execution) requires "HTTP 400 with a
    stable error code" for this refusal — unlike the generic ValueError
    path, the message is not sanitized (it names no internals) and a stable
    `code` is set so clients can branch on it.
    """
    logger.warning("DeterminismUnsupportedError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=ErrorDetail(
                message=str(exc),
                type="invalid_request_error",
                code="deterministic_unsupported",
            )
        ).model_dump(),
    )


async def engine_saturated_error_handler(
    request: Request, exc: EngineSaturatedError
) -> JSONResponse:
    """AdmissionController rejected a batch-tier request under KV pressure (429).

    Unlike the other handlers, the message isn't sanitized to a generic string:
    it names no internals (no paths, no stack details) and telling the client
    to retry is the whole point of a 429.
    """
    logger.warning("EngineSaturatedError on %s: %s", request.url.path, exc)
    retry_after = max(1, round(exc.retry_after_s))
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(retry_after)},
        content=ErrorResponse(
            error=ErrorDetail(message=str(exc), type="rate_limit_error")
        ).model_dump(),
    )
