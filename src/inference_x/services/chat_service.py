from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator

from inference_x.core.settings import get_settings
from inference_x.engines.base import BaseEngine
from inference_x.engines.pool import EnginePool
from inference_x.routing.admission import AdmissionController
from inference_x.routing.task_router import TaskRouter
from inference_x.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatStreamChunk,
    ResolvedRequest,
)
from inference_x.services.model_service import ModelRegistry

_RESOLVED_FIELDS = tuple(ResolvedRequest.model_fields)


def _resolved(effective_request: ChatCompletionRequest) -> ResolvedRequest:
    """Serialize the Effective Request — what the server actually ran (OS-4).

    Built from *effective_request*, never from the client's original: building it
    from the original would report what was asked for rather than what ran, which
    inverts the point of the block.

    Field membership is derived from ``ResolvedRequest`` rather than listed here,
    so the derivability rule has exactly one home and this function cannot drift
    from it (see ResolvedRequest, and the schema test that enforces the rule).
    """
    return ResolvedRequest(
        **{name: getattr(effective_request, name) for name in _RESOLVED_FIELDS}
    )


class ChatService:
    """Orchestrates chat completion requests through a router and engine pool.

    Phase 4 multi-model: the service dispatches to whichever engine in the pool
    matches the routed model name.  A pool with a single entry behaves identically
    to the previous single-engine setup.

    Phase 7/2 (VRAM-aware admission, see DEC-037): every request passes through
    an AdmissionController after routing and before dispatch, which enforces
    context-length limits and clamps/rejects under KV-pool pressure. Callers
    that construct ChatService directly (tests, stub engines) get a default
    controller with no VRAM tier — behaves like the pre-admission 4096-token cap.
    """

    def __init__(
        self,
        engine_pool: EnginePool,
        registry: ModelRegistry,
        router: TaskRouter,
        admission: AdmissionController | None = None,
    ) -> None:
        self._pool = engine_pool
        self._registry = registry
        self._router = router
        self._admission = admission or AdmissionController(registry)

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Route and execute a chat completion request.

        Raises:
            ValueError: routed model not in engine pool, or request rejected
                by admission control (context too long).
            EngineSaturatedError: KV pool saturated for a batch-tier request.
            RuntimeError: engine inference failure.
        """
        routed_model, engine = self._resolve_engine(request)
        admitted = await self._admission.admit(routed_model, request, engine)
        effective_request = request.model_copy(
            update={"max_tokens": admitted.effective_max_tokens}
        )
        try:
            response = await engine.generate(effective_request)
        finally:
            self._admission.release(routed_model, admitted.reserved_tokens)
        # Attached here, never by the engine: the Engine Boundary does not learn
        # about admission (DEC-047).
        return response.model_copy(
            update={
                "resolved": _resolved(effective_request),
                "warnings": list(admitted.warnings),
            }
        )

    async def stream_response(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Route a chat request and yield OpenAI-compatible SSE events.

        Event order is fixed (DEC-049 and OS-2 R3, extended at the head by
        DEC-053) and is the protocol:

        0. exactly one pre-generation event with ``choices: []`` carrying
           ``resolved`` and ``warnings``, always, before any content;
        1. zero or more content events, each with ``finish_reason: null``;
        2. exactly one terminal event with an empty delta and a real
           ``finish_reason``;
        3. one usage event with ``choices: []`` — only when the client asked via
           ``stream_options.include_usage`` and the engine accounted usage.
           Carries ``timing`` alongside ``usage`` (Phase B3) when the engine
           supplied it — one opt-in event for all terminal metadata, not a
           second flag;
        4. ``data: [DONE]``, always last.

        Event 0 sits at the head rather than before ``[DONE]`` because everything
        it carries is fixed the moment ``admit()`` returns, and a trailer would
        be lost on the timeout path — the path where knowing what the server
        resolved matters most (DEC-053). Nothing is ever emitted after the usage
        event: OpenAI documents that chunk as the one streamed before ``[DONE]``
        and clients use it as an end sentinel.

        The terminal event is separate rather than folded into the last content
        event, because the service cannot know a content event is the last one
        until the engine says so.

        Applies a per-token timeout (INFERENCE_X_STREAM_TIMEOUT_S) so that a
        stalled engine does not hold the connection open indefinitely. On
        timeout the error event is emitted and neither a terminal nor a usage
        event follows — only ``[DONE]``.

        Reservation lifetime: from the instant ``admit()`` returns until this
        generator terminates for any reason (normal completion, timeout, engine
        exception, cancellation, or the caller closing the generator — which is
        how a client disconnect surfaces here, including during the prologue
        event above), exactly one matching ``release()`` occurs. The ``try``
        below starts immediately after ``admit()`` and has a single ``finally``,
        so every suspension point in between — the prologue ``yield`` included —
        is covered by the same, single release call site.
        """
        routed_model, engine = self._resolve_engine(request)
        admitted = await self._admission.admit(routed_model, request, engine)
        gen: AsyncGenerator[ChatStreamChunk, None] | None = None
        try:
            effective_request = request.model_copy(
                update={"max_tokens": admitted.effective_max_tokens}
            )
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            timeout_s = get_settings().stream_timeout_s
            include_usage = bool(
                request.stream_options and request.stream_options.include_usage
            )

            def _event(payload: dict) -> str:
                return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

            yield _event(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "choices": [],
                    "resolved": _resolved(effective_request).model_dump(),
                    "warnings": [w.model_dump() for w in admitted.warnings],
                }
            )

            gen = engine.generate_stream(effective_request)
            while True:
                try:
                    if timeout_s > 0:
                        chunk = await asyncio.wait_for(
                            gen.__anext__(), timeout=timeout_s
                        )
                    else:
                        chunk = await gen.__anext__()
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    yield (
                        f"data: {{\"error\":\"stream timed out after {timeout_s}s\"}}\n\n"
                    )
                    break

                if chunk.content:
                    yield _event(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "delta": {"content": chunk.content},
                                    "index": 0,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )

                if chunk.finish_reason is not None:
                    yield _event(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "delta": {},
                                    "index": 0,
                                    "finish_reason": chunk.finish_reason,
                                }
                            ],
                        }
                    )
                    if include_usage and chunk.usage is not None:
                        payload = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "choices": [],
                            "usage": chunk.usage.model_dump(),
                        }
                        if chunk.timing is not None:
                            payload["timing"] = chunk.timing.model_dump()
                        yield _event(payload)
        finally:
            if gen is not None:
                await gen.aclose()
            self._admission.release(routed_model, admitted.reserved_tokens)

        yield "data: [DONE]\n\n"

    def _resolve_engine(self, request: ChatCompletionRequest) -> tuple[str, BaseEngine]:
        routed_model = self._router.select(request)

        loaded = self._pool.loaded_models()
        if routed_model not in loaded:
            loaded_str = loaded[0] if len(loaded) == 1 else str(loaded)
            raise ValueError(
                f"Routed to model '{routed_model}' but loaded model is "
                f"'{loaded_str}'. "
                f"Restart with INFERENCE_X_LOADED_MODELS={routed_model} "
                f"(or add it to the existing list)."
            )

        return routed_model, self._pool.get(routed_model)

    def engine_healthy(self) -> bool:
        return self._pool.all_healthy()

    def loaded_models(self) -> list[str]:
        return self._pool.loaded_models()

    def registry(self) -> ModelRegistry:
        return self._registry
