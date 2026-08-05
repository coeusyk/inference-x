"""Unit tests for the engine interface and a stub implementation."""
import pytest

from inference_x.engines.base import BaseEngine
from inference_x.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    ChatStreamChunk,
)


class _StubEngine(BaseEngine):
    """Minimal concrete engine used only for interface contract tests."""

    def __init__(self, healthy: bool = True) -> None:
        self._healthy = healthy

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content="stub response"),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=1, completion_tokens=2, total_tokens=3
            ),
        )

    async def generate_stream(self, request: ChatCompletionRequest):
        yield ChatStreamChunk(content="stub response")
        yield ChatStreamChunk(content="", finish_reason="stop")

    def is_healthy(self) -> bool:
        return self._healthy


class TestBaseEngineContract:
    def test_stub_satisfies_interface(self):
        engine = _StubEngine()
        assert isinstance(engine, BaseEngine)

    def test_is_healthy_returns_bool(self):
        assert _StubEngine(healthy=True).is_healthy() is True
        assert _StubEngine(healthy=False).is_healthy() is False

    @pytest.mark.asyncio
    async def test_generate_returns_response(self):
        engine = _StubEngine()
        req = ChatCompletionRequest(
            model="stub",
            messages=[ChatMessage(role="user", content="hi")],
        )
        resp = await engine.generate(req)
        assert isinstance(resp, ChatCompletionResponse)
        assert resp.model == "stub"
        assert resp.choices[0].message.content == "stub response"

    @pytest.mark.asyncio
    async def test_generate_stream_yields_chunks(self):
        engine = _StubEngine()
        req = ChatCompletionRequest(
            model="stub",
            messages=[ChatMessage(role="user", content="hi")],
        )
        chunks = [chunk async for chunk in engine.generate_stream(req)]
        assert [c.content for c in chunks if c.content] == ["stub response"]
        # The contract requires a terminal chunk carrying a finish reason.
        assert chunks[-1].finish_reason == "stop"
