"""Unit tests for ChatService (Phase 4: multi-model engine pool)."""
import json

import pytest

from inference_x.engines.base import BaseEngine
from inference_x.engines.pool import EnginePool
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

def _make_registry(*names: str) -> ModelRegistry:
    return ModelRegistry([ModelEntry(name=n, model_path=f"test/{n}") for n in names])


def _make_pool(*names: str, healthy: bool = True, raise_on_generate: bool = False) -> EnginePool:
    return EnginePool({n: _StubEngine(healthy=healthy, raise_on_generate=raise_on_generate) for n in names})


def _make_service(
    model_name: str = "test",
    healthy: bool = True,
    raise_on_generate: bool = False,
) -> ChatService:
    registry = _make_registry(model_name)
    router = TaskRouter(registry, model_name)
    pool = _make_pool(model_name, healthy=healthy, raise_on_generate=raise_on_generate)
    return ChatService(engine_pool=pool, registry=registry, router=router)


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

    async def generate_stream(self, request: ChatCompletionRequest):
        if self._raise:
            raise RuntimeError("stub streaming error")
        yield "ok"

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
    async def test_stream_response_formats_sse(self):
        svc = _make_service()
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hello")],
            stream=True,
        )
        events = [event async for event in svc.stream_response(req)]
        assert events[-1] == "data: [DONE]\n\n"
        payload = json.loads(events[0].removeprefix("data: ").strip())
        assert payload["object"] == "chat.completion.chunk"
        assert payload["choices"][0]["delta"]["content"] == "ok"

    @pytest.mark.asyncio
    async def test_complete_propagates_runtime_error(self):
        svc = _make_service(raise_on_generate=True)
        with pytest.raises(RuntimeError, match="stub generation error"):
            await svc.complete(self._req())

    def test_engine_healthy_delegates_to_pool(self):
        assert _make_service(healthy=True).engine_healthy() is True
        assert _make_service(healthy=False).engine_healthy() is False

    def test_loaded_models_returns_pool_contents(self):
        registry = _make_registry("alpha", "beta")
        router = TaskRouter(registry, "alpha")
        pool = _make_pool("alpha", "beta")
        svc = ChatService(engine_pool=pool, registry=registry, router=router)
        assert svc.loaded_models() == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_unregistered_model_falls_back_to_default(self):
        """When a client requests an unknown model, routing falls back to default."""
        svc = _make_service(model_name="test")
        req = self._req(model="unknown-model")
        resp = await svc.complete(req)
        assert resp is not None

    @pytest.mark.asyncio
    async def test_model_not_in_pool_raises_value_error(self):
        """Requesting a registered but unloaded model raises ValueError."""
        registry = _make_registry("model-a", "model-b")
        router = TaskRouter(registry, "model-a")
        # Pool only has model-a loaded
        pool = EnginePool({"model-a": _StubEngine()})
        svc = ChatService(engine_pool=pool, registry=registry, router=router)
        req = self._req(model="model-b")
        with pytest.raises(ValueError, match="Routed to model"):
            await svc.complete(req)

    @pytest.mark.asyncio
    async def test_multi_model_pool_dispatches_correctly(self):
        """Each model in the pool gets its own engine calls."""
        registry = _make_registry("alpha", "beta")
        router = TaskRouter(registry, "alpha")

        class _NamedEngine(BaseEngine):
            def __init__(self, name: str) -> None:
                self._name = name

            async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
                return ChatCompletionResponse(
                    model=self._name,
                    choices=[ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionMessage(content=f"from {self._name}"),
                        finish_reason="stop",
                    )],
                    usage=ChatCompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

            async def generate_stream(self, request: ChatCompletionRequest):
                yield f"from {self._name}"

            def is_healthy(self) -> bool:
                return True

        pool = EnginePool({"alpha": _NamedEngine("alpha"), "beta": _NamedEngine("beta")})
        svc = ChatService(engine_pool=pool, registry=registry, router=router)

        resp_a = await svc.complete(self._req(model="alpha"))
        resp_b = await svc.complete(self._req(model="beta"))

        assert resp_a.choices[0].message.content == "from alpha"
        assert resp_b.choices[0].message.content == "from beta"
