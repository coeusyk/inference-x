"""Unit tests for ChatService (Phase 2: routing-aware)."""
import pytest

from inference_x.engines.base import BaseEngine
from inference_x.routing.task_router import TaskRouter
from inference_x.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from inference_x.schemas.model import ModelEntry
from inference_x.services.chat_service import ChatService
from inference_x.services.model_service import ModelRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(name: str = "test") -> ModelRegistry:
    return ModelRegistry([ModelEntry(name=name, model_path="test/stub")])


def _make_service(
    model_name: str = "test",
    healthy: bool = True,
    raise_on_generate: bool = False,
) -> ChatService:
    registry = _make_registry(model_name)
    router = TaskRouter(registry, model_name)
    return ChatService(
        engine=_StubEngine(healthy=healthy, raise_on_generate=raise_on_generate),
        registry=registry,
        router=router,
        loaded_model=model_name,
    )


class _StubEngine(BaseEngine):
    def __init__(self, healthy: bool = True, raise_on_generate: bool = False) -> None:
        self._healthy = healthy
        self._raise = raise_on_generate

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if self._raise:
            raise RuntimeError("stub generation error")
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def is_healthy(self) -> bool:
        return self._healthy


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChatService:
    def _req(self, model: str = "test") -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model=model,
            messages=[ChatMessage(role="user", content="hello")],
        )

    @pytest.mark.asyncio
    async def test_complete_returns_response(self):
        svc = _make_service()
        resp = await svc.complete(self._req())
        assert isinstance(resp, ChatCompletionResponse)
        assert resp.choices[0].message.content == "ok"

    @pytest.mark.asyncio
    async def test_complete_rejects_streaming(self):
        svc = _make_service()
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hello")],
            stream=True,
        )
        with pytest.raises(ValueError, match="Streaming is not supported"):
            await svc.complete(req)

    @pytest.mark.asyncio
    async def test_complete_propagates_runtime_error(self):
        svc = _make_service(raise_on_generate=True)
        with pytest.raises(RuntimeError, match="stub generation error"):
            await svc.complete(self._req())

    def test_engine_healthy_delegates_to_engine(self):
        assert _make_service(healthy=True).engine_healthy() is True
        assert _make_service(healthy=False).engine_healthy() is False

    @pytest.mark.asyncio
    async def test_unregistered_model_falls_back_to_default(self):
        """When a client requests an unknown model, routing falls back to default."""
        svc = _make_service(model_name="test")
        req = self._req(model="unknown-model")
        resp = await svc.complete(req)
        assert resp is not None

    @pytest.mark.asyncio
    async def test_mismatched_loaded_model_raises(self):
        """If routing picks a model that differs from the loaded engine, raise ValueError."""
        registry = ModelRegistry([
            ModelEntry(name="model-a", model_path="a/a"),
            ModelEntry(name="model-b", model_path="b/b"),
        ])
        router = TaskRouter(registry, "model-a")
        svc = ChatService(
            engine=_StubEngine(),
            registry=registry,
            router=router,
            loaded_model="model-b",
        )
        req = self._req(model="model-a")
        with pytest.raises(ValueError, match="Routed to model"):
            await svc.complete(req)
