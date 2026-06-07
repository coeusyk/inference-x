from abc import ABC, abstractmethod

from inference_x.schemas.chat import ChatCompletionRequest


class BaseRouter(ABC):
    """Stable routing contract.

    Given a chat completion request, returns the name of the model that should
    serve it. Routers must not perform inference or build engines — they only
    select a model name from the configured registry.

    The service layer calls the router before dispatching to an engine.
    """

    @abstractmethod
    def select(self, request: ChatCompletionRequest) -> str:
        """Return the model name that should handle *request*.

        Raises:
            ValueError: if no suitable model can be selected.
        """
