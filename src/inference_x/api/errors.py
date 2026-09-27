import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from inference_x.routing.admission import (
    ContextTooLongError,
    EngineSaturatedError,
    StrictModeViolationError,
)
from inference_x.schemas.common import ErrorDetail, ErrorResponse
from inference_x.services.chat_service import ToolCallingUnsupportedError
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


def _invalid_request(message: str, *, param: str | None = None, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=ErrorDetail(
                message=message, type="invalid_request_error", param=param, code=code
            )
        ).model_dump(),
    )


async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Schema validation failure → OpenAI-shaped 400 instead of FastAPI's 422 (DEC-063).

    Pydantic's messages name only the field and the constraint — no internals —
    so they are returned as-is; `param` is the first error's location with the
    leading `body` segment dropped.
    """
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = [str(p) for p in first.get("loc", ()) if p != "body"]
    param = ".".join(loc) or None
    msg = str(first.get("msg", "Invalid request."))
    if first.get("type") == "extra_forbidden":
        msg = "Unsupported parameter"
    logger.warning("Request validation failed on %s: %s", request.url.path, errors)
    return _invalid_request(f"{param}: {msg}" if param else msg, param=param)


async def context_too_long_error_handler(
    request: Request, exc: ContextTooLongError
) -> JSONResponse:
    """Context overflow: real message + OpenAI's `context_length_exceeded` code.

    The message follows OpenAI's wording so clients (litellm/Aider) recognize
    it as a context-window error. It names only token counts and the model.
    """
    logger.warning("ContextTooLongError on %s: %s", request.url.path, exc)
    return _invalid_request(str(exc), code="context_length_exceeded")


async def strict_violation_error_handler(
    request: Request, exc: StrictModeViolationError
) -> JSONResponse:
    logger.warning("StrictModeViolationError on %s: %s", request.url.path, exc)
    return _invalid_request(str(exc), code="strict_violation")


async def tool_calling_unsupported_error_handler(
    request: Request, exc: ToolCallingUnsupportedError
) -> JSONResponse:
    """Tool request for a model without a declared parser (DEC-065). The
    message names only the model, never the backend or parser."""
    logger.warning("ToolCallingUnsupportedError on %s: %s", request.url.path, exc)
    return _invalid_request(str(exc), param="tools", code="tool_calling_unsupported")


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
