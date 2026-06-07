from inference_x.engines.base import BaseEngine
from inference_x.routing.base import BaseRouter
from inference_x.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from inference_x.services.model_service import ModelRegistry


class ChatService:
    """Orchestrates chat completion requests through a router and engine.

    Phase 2 adds routing: the service resolves which model should handle a
    request before dispatching to the engine. The route handler is unchanged.

    The engine is still a single loaded instance (one GPU). If the routed model
    differs from the loaded engine's model, a clear error is raised rather than
    silently returning wrong results.
    """

    def __init__(
        self,
        engine: BaseEngine,
        registry: ModelRegistry,
        router: BaseRouter,
        loaded_model: str,
    ) -> None:
        self._engine = engine
        self._registry = registry
        self._router = router
        self._loaded_model = loaded_model

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Route and execute a chat completion request.

        Raises:
            ValueError: streaming requested, or routed model not loaded.
            RuntimeError: engine inference failure.
        """
        if request.stream:
            raise ValueError("Streaming is not supported in Phase 1")

        routed_model = self._router.select(request)

        if routed_model != self._loaded_model:
            raise ValueError(
                f"Routed to model '{routed_model}' but loaded model is "
                f"'{self._loaded_model}'. This server loads one model at a time "
                f"(see DEC-011). Restart with INFERENCE_X_DEFAULT_MODEL={routed_model}."
            )

        return await self._engine.generate(request)

    def engine_healthy(self) -> bool:
        return self._engine.is_healthy()

    def loaded_model(self) -> str:
        return self._loaded_model

    def registry(self) -> ModelRegistry:
        return self._registry
