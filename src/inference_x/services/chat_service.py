from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator

from inference_x.engines.base import BaseEngine
from inference_x.engines.pool import EnginePool
from inference_x.routing.base import BaseRouter
from inference_x.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from inference_x.services.model_service import ModelRegistry


class ChatService:
    """Orchestrates chat completion requests through a router and engine pool.

    Phase 4 multi-model: the service dispatches to whichever engine in the pool
    matches the routed model name.  A pool with a single entry behaves identically
    to the previous single-engine setup.
    """

    def __init__(
        self,
        engine_pool: EnginePool,
        registry: ModelRegistry,
        router: BaseRouter,
    ) -> None:
        self._pool = engine_pool
        self._registry = registry
        self._router = router

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Route and execute a chat completion request.

        Raises:
            ValueError: routed model not in engine pool.
            RuntimeError: engine inference failure.
        """
        engine = self._resolve_engine(request)
        return await engine.generate(request)

    async def stream_response(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Route a chat request and yield OpenAI-compatible SSE events."""
        engine = self._resolve_engine(request)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        async for chunk in engine.generate_stream(request):
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

        yield "data: [DONE]\n\n"

    def _resolve_engine(self, request: ChatCompletionRequest) -> BaseEngine:
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

        return self._pool.get(routed_model)

    def engine_healthy(self) -> bool:
        return self._pool.all_healthy()

    def loaded_models(self) -> list[str]:
        return self._pool.loaded_models()

    def registry(self) -> ModelRegistry:
        return self._registry
