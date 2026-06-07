from inference_x.engines.base import BaseEngine
from inference_x.schemas.chat import ChatCompletionRequest, ChatCompletionResponse


class ChatService:
    """Orchestrates chat completion requests through an engine.

    This layer exists to keep route handlers free of inference logic and to
    provide a clear seam for later phases (routing policies, observability).
    """

    def __init__(self, engine: BaseEngine) -> None:
        self._engine = engine

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Run a chat completion request and return a typed response.

        Raises:
            RuntimeError: Propagated from the engine on inference failure.
        """
        return await self._engine.generate(request)

    def engine_healthy(self) -> bool:
        return self._engine.is_healthy()
