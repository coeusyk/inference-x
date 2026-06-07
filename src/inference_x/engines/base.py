from abc import ABC, abstractmethod

from inference_x.schemas.chat import ChatCompletionRequest, ChatCompletionResponse


class BaseEngine(ABC):
    """Stable engine contract. All inference implementations must satisfy this interface.

    The rest of the system depends on this interface, not on concrete implementations.
    Adding a second engine in a future phase must not require changes here.
    """

    @abstractmethod
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Run inference for a chat completion request and return a typed response."""

    @abstractmethod
    def is_healthy(self) -> bool:
        """Return True if the engine is loaded and ready to serve requests."""
