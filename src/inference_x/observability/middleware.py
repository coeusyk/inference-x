"""ObservabilityMiddleware: cross-cutting timing and metadata capture.

This middleware wraps every request to:
  1. Record wall-clock latency (start → response complete).
  2. Extract model name from the request body for /v1/chat/completions.
  3. Extract token counts from the response body for /v1/chat/completions 200s.
  4. Call MetricsRecorder.record() after the response is ready to send.

For chat completion 200 responses, the body is buffered once to read token
counts, then re-wrapped in a new Response with the same status/headers so the
client receives an identical payload. For all other paths, the body_iterator
is never touched.

It does NOT alter request or response shape in any way.
It does NOT call services or route handlers directly.
All extraction errors are caught and swallowed — they never break a request.
"""
from __future__ import annotations

import json
import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from inference_x.observability.recorder import MetricsRecorder

logger = logging.getLogger(__name__)

_CHAT_PATH = "/v1/chat/completions"


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Timing and metadata middleware.

    Pass *recorder* as a keyword argument when attaching to the app:

        app.add_middleware(ObservabilityMiddleware, recorder=recorder)
    """

    def __init__(self, app, recorder: MetricsRecorder) -> None:
        super().__init__(app)
        self._recorder = recorder

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        is_chat = request.method == "POST" and request.url.path == _CHAT_PATH

        model: str | None = None
        if is_chat:
            model = await _extract_model(request)

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._recorder.record(
                path=request.url.path,
                method=request.method,
                status_code=500,
                latency_ms=elapsed_ms,
                model=model,
                error=True,
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        status_code = response.status_code

        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None

        if is_chat and status_code == 200:
            # Buffer the response body to read token counts, then re-wrap so
            # the client receives an identical payload.
            body, prompt_tokens, completion_tokens, total_tokens = (
                await _buffer_and_extract_tokens(response)
            )
            response = Response(
                content=body,
                status_code=status_code,
                headers={
                    k: v
                    for k, v in response.headers.items()
                    if k.lower() != "content-length"
                },
                media_type=response.media_type,
            )

        self._recorder.record(
            path=request.url.path,
            method=request.method,
            status_code=status_code,
            latency_ms=elapsed_ms,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            error=status_code >= 500,
        )

        return response


async def _extract_model(request: Request) -> str | None:
    """Parse the JSON request body and return the 'model' field if present.

    Starlette caches the body after the first `await request.body()` call so
    downstream handlers can still read it.
    """
    try:
        body_bytes = await request.body()
        payload = json.loads(body_bytes)
        return payload.get("model") or None
    except Exception:
        return None


async def _buffer_and_extract_tokens(
    response: Response,
) -> tuple[bytes, int | None, int | None, int | None]:
    """Consume body_iterator, parse JSON usage, return (body, p, c, total).

    Returns the raw body bytes so the caller can rebuild the Response.
    Returns (None, None, None) token values on any parse error.
    """
    try:
        body_bytes = b""
        async for chunk in response.body_iterator:  # type: ignore[attr-defined]
            body_bytes += chunk if isinstance(chunk, bytes) else chunk.encode()

        payload = json.loads(body_bytes)
        usage = payload.get("usage") or {}
        p = usage.get("prompt_tokens")
        c = usage.get("completion_tokens")
        t = usage.get("total_tokens")

        return (
            body_bytes,
            int(p) if p is not None else None,
            int(c) if c is not None else None,
            int(t) if t is not None else None,
        )
    except Exception as exc:
        logger.debug("ObservabilityMiddleware: token extraction failed: %s", exc)
        body_bytes = body_bytes if "body_bytes" in dir() else b""  # type: ignore[possibly-undefined]
        return body_bytes, None, None, None
