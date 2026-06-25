from __future__ import annotations

from inference_x.schemas.chat import ChatCompletionRequest
from inference_x.services.model_service import ModelRegistry


class ExplicitModelPolicy:
    """Pass-through policy: honour the model name the client requested.

    If the requested model is registered, return it.
    Raises ValueError if the model is not in the registry.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def apply(self, request: ChatCompletionRequest) -> str | None:
        """Return the requested model name if registered, else None."""
        if request.model in self._registry:
            return request.model
        return None


class DefaultModelPolicy:
    """Fallback policy: return a fixed default model name.

    Used when no other policy matches or the client did not request
    a specific registered model.
    """

    def __init__(self, default_model: str, registry: ModelRegistry) -> None:
        if default_model not in registry:
            available = registry.names()
            raise ValueError(
                f"Default model '{default_model}' is not in the registry. "
                f"Available: {available}"
            )
        self._default = default_model

    def apply(self, request: ChatCompletionRequest) -> str:  # noqa: ARG002
        return self._default
