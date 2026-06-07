"""Unit tests for ChatService."""
import pytest

from inference_x.engines.base import BaseEngine
from inference_x.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from inference_x.services.chat_service import ChatService


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


class TestChatService:
    def _req(self) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hello")],
        )

    @pytest.mark.asyncio
    async def test_complete_returns_response(self):
        svc = ChatService(_StubEngine())
        resp = await svc.complete(self._req())
        assert isinstance(resp, ChatCompletionResponse)
        assert resp.choices[0].message.content == "ok"

    @pytest.mark.asyncio
    async def test_complete_propagates_runtime_error(self):
        svc = ChatService(_StubEngine(raise_on_generate=True))
        with pytest.raises(RuntimeError, match="stub generation error"):
            await svc.complete(self._req())

    def test_engine_healthy_delegates_to_engine(self):
        assert ChatService(_StubEngine(healthy=True)).engine_healthy() is True
        assert ChatService(_StubEngine(healthy=False)).engine_healthy() is False
