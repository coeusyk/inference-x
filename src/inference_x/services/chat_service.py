from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator

from inference_x.core.settings import get_settings
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
        """Route a chat request and yield OpenAI-compatible SSE events.

        Applies a per-token timeout (INFERENCE_X_STREAM_TIMEOUT_S) so that a
        stalled engine does not hold the connection open indefinitely.
        """
        engine = self._resolve_engine(request)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        timeout_s = get_settings().stream_timeout_s

        gen = engine.generate_stream(request)
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

        yield "data: [DONE]\n\n"

    _GLOBAL_MAX_TOKENS = 4096

    def _enforce_max_tokens_cap(self, routed_model: str, request: ChatCompletionRequest) -> None:
        """Reject max_tokens above min(4096, model max_model_len) when configured."""
        cap = self._GLOBAL_MAX_TOKENS
        if routed_model in self._registry:
            entry = self._registry.get(routed_model)
            if entry.max_model_len is not None:
                cap = min(cap, entry.max_model_len)
        requested = request.max_tokens if request.max_tokens is not None else 512
        if requested > cap:
            raise ValueError(
                f"max_tokens {requested} exceeds the limit of {cap} for model '{routed_model}'"
            )

    def _resolve_engine(self, request: ChatCompletionRequest) -> BaseEngine:
        routed_model = self._router.select(request)
        self._enforce_max_tokens_cap(routed_model, request)

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
