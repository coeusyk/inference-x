from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator

from inference_x.core.settings import get_settings
from inference_x.engines.base import BaseEngine
from inference_x.engines.pool import EnginePool
from inference_x.routing.admission import AdmissionController
from inference_x.routing.base import BaseRouter
from inference_x.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from inference_x.services.model_service import ModelRegistry


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
        router: BaseRouter,
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
        admitted = self._admission.admit(routed_model, request, engine)
        effective_request = request.model_copy(
            update={"max_tokens": admitted.effective_max_tokens}
        )
        try:
            return await engine.generate(effective_request)
        finally:
            self._admission.release(routed_model, admitted.reserved_tokens)

    async def stream_response(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Route a chat request and yield OpenAI-compatible SSE events.

        Applies a per-token timeout (INFERENCE_X_STREAM_TIMEOUT_S) so that a
        stalled engine does not hold the connection open indefinitely.
        """
        routed_model, engine = self._resolve_engine(request)
        admitted = self._admission.admit(routed_model, request, engine)
        effective_request = request.model_copy(
            update={"max_tokens": admitted.effective_max_tokens}
        )
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        timeout_s = get_settings().stream_timeout_s

        gen = engine.generate_stream(effective_request)
        try:
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

                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "delta": {"content": chunk},
                            "index": 0,
                        }
                    ],
                }
                yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
        finally:
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
