"""ObservabilityMiddleware: cross-cutting timing and metadata capture.

This middleware wraps every request to:
  1. Record wall-clock latency (start → response complete).
  2. Extract model name from the request body for /v1/chat/completions.
  3. Extract token counts from the response body for /v1/chat/completions 200s.
  4. Call MetricsRecorder.record() after the response is ready to send.

For non-streaming chat completion 200 responses, the body is buffered once to
read token counts, then re-wrapped in a new Response with the same
status/headers so the client receives an identical payload.

For streaming (SSE) chat completion 200 responses, the body is NOT buffered —
that would defeat the point of streaming. Instead ``body_iterator`` is wrapped
so chunks still pass through immediately, while the wrapper measures
time-to-first-chunk (TTFT) and accumulates an approximate completion-token
count (whitespace word count over each chunk's delta content — the same
approximation the playground UI and benchmarks/runner.py already use, because
streaming responses carry no final `usage` block; see DEC-023). The wrapper
records exactly one RequestRecord when the stream ends, so the unconditional
record() call at the bottom of dispatch() is skipped for this branch to avoid
double-counting.

Known limitation: an engine failure *after* headers are sent (mid-stream) is
not observable as `error=True` here. Starlette's BaseHTTPMiddleware runs the
inner app in a separate task and only surfaces its exception to the outer
ASGI call *after* our dispatch() has already finished sending the response —
from this wrapper's point of view the body_iterator just ends early, same as
a normal end-of-stream. The partial TTFT/token count for the truncated stream
is still recorded; only the error flag is unreliable for this specific case.
True client disconnects (GeneratorExit at the `yield` below) are handled the
same way — record whatever was captured so far.

For all other paths, the body_iterator is never touched.

It does NOT alter request or response shape in any way.
It does NOT call services or route handlers directly.
All extraction errors are caught and swallowed — they never break a request.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator

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
        stream_requested = False
        if is_chat:
            model, stream_requested = await _extract_chat_request_metadata(request)

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

        # Streaming chat (SSE): wrap body_iterator so it still records TTFT and an
        # approximate token count, then return immediately — the wrapper records
        # its own RequestRecord once the stream ends, so falling through to the
        # unconditional record() call below would double-count this request.
        if is_chat and status_code == 200 and _is_event_stream_response(response):
            response.body_iterator = _wrap_and_record_sse(  # type: ignore[attr-defined]
                response.body_iterator,  # type: ignore[attr-defined]
                start=start,
                model=model,
                path=request.url.path,
                method=request.method,
                recorder=self._recorder,
            )
            return response

        if (
            is_chat
            and status_code == 200
            and not stream_requested
            and not _is_event_stream_response(response)
        ):
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


async def _extract_chat_request_metadata(request: Request) -> tuple[str | None, bool]:
    """Parse cached chat request metadata used by metrics.

    Starlette caches the body after the first `await request.body()` call so
    downstream handlers can still read it.
    """
    try:
        body_bytes = await request.body()
        payload = json.loads(body_bytes)
        return payload.get("model") or None, bool(payload.get("stream"))
    except Exception:
        return None, False


def _is_event_stream_response(response: Response) -> bool:
    content_type = response.headers.get("content-type", "")
    media_type = getattr(response, "media_type", "") or ""
    return "text/event-stream" in content_type or "text/event-stream" in media_type


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


def _count_sse_delta_tokens(line: bytes) -> int:
    """Approximate completion-token count for one ``data: {...}`` SSE line.

    Uses a whitespace word count over the chunk's ``delta.content`` — the same
    approximation used client-side by the playground and by
    benchmarks/runner.py's ``_run_prompt_stream`` (streaming responses don't
    carry a final ``usage`` block from vLLM; see DEC-023). Returns 0 for
    ``[DONE]``, malformed, or non-``data:`` lines.
    """
    text = line.decode("utf-8", errors="ignore").strip()
    if not text.startswith("data:"):
        return 0
    data = text[len("data:") :].strip()
    if not data or data == "[DONE]":
        return 0
    try:
        payload = json.loads(data)
        delta = (payload.get("choices") or [{}])[0].get("delta", {})
        content = delta.get("content", "")
        return len(content.split()) if content else 0
    except Exception:
        return 0


async def _wrap_and_record_sse(
    body_iterator: AsyncIterator,
    *,
    start: float,
    model: str | None,
    path: str,
    method: str,
    recorder: MetricsRecorder,
) -> AsyncIterator[bytes]:
    """Pass SSE chunks through unmodified while measuring TTFT and tokens/sec.

    Records exactly one RequestRecord when the stream ends — on normal
    completion or on early client disconnect (which raises GeneratorExit at
    the ``yield`` when Starlette closes this generator) — so the caller must
    not also call recorder.record() for this response. See the module
    docstring for why a mid-stream *engine* failure can't reliably set
    ``error=True`` here.
    """
    ttft_ms: float | None = None
    tokens_generated = 0
    buffer = b""
    status_code = 200
    had_error = False
    try:
        async for chunk in body_iterator:
            raw = chunk if isinstance(chunk, bytes) else chunk.encode()
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000
            buffer += raw
            while b"\n\n" in buffer:
                line, buffer = buffer.split(b"\n\n", 1)
                tokens_generated += _count_sse_delta_tokens(line)
            yield raw
    except Exception:
        had_error = True
        status_code = 500
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        tokens_per_sec = (
            tokens_generated / elapsed_ms * 1000
            if elapsed_ms > 0 and tokens_generated
            else None
        )
        try:
            recorder.record(
                path=path,
                method=method,
                status_code=status_code,
                latency_ms=elapsed_ms,
                model=model,
                completion_tokens=tokens_generated or None,
                total_tokens=tokens_generated or None,
                error=had_error,
                ttft_ms=ttft_ms,
                tokens_per_sec=tokens_per_sec,
            )
        except Exception as exc:
            logger.warning("ObservabilityMiddleware: SSE stream record failed: %s", exc)
