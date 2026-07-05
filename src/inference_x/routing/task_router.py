from __future__ import annotations

from inference_x.routing.policies import DefaultModelPolicy, ExplicitModelPolicy
from inference_x.schemas.chat import ChatCompletionRequest
from inference_x.services.model_service import ModelRegistry


class TaskRouter:
    """Default router for Phase 2.

    Selection order:
    1. ExplicitModelPolicy — honour the model name in the request if it is registered.
    2. DefaultModelPolicy  — fall back to the configured default model.

    This keeps the client in control when they specify a known model, while
    providing a safe fallback when they use an unknown or missing model name.
    """

    def __init__(self, registry: ModelRegistry, default_model: str) -> None:
        self._explicit = ExplicitModelPolicy(registry)
        self._default = DefaultModelPolicy(default_model, registry)

    def select(self, request: ChatCompletionRequest) -> str:
        result = self._explicit.apply(request)
        if result is not None:
            return result
        return self._default.apply(request)
