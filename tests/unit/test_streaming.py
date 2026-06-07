"""Tests for SSE streaming chat completions."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from inference_x.api.deps import get_chat_service, get_registry
from inference_x.api.main import app
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

_TEST_MODEL = "test-model"


class _StreamingEngine(BaseEngine):
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content="Hello world"),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=2,
                completion_tokens=2,
                total_tokens=4,
            ),
        )

    async def generate_stream(self, request: ChatCompletionRequest):
        yield "Hello"
        yield " world"

    def is_healthy(self) -> bool:
        return True


def _registry() -> ModelRegistry:
    return ModelRegistry([ModelEntry(name=_TEST_MODEL, model_path="test/stub")])


def _service() -> ChatService:
    registry = _registry()
    router = TaskRouter(registry, _TEST_MODEL)
    pool = EnginePool({_TEST_MODEL: _StreamingEngine()})
    return ChatService(engine_pool=pool, registry=registry, router=router)


def _payload(stream: bool | None = None) -> dict:
    payload = {
        "model": _TEST_MODEL,
        "messages": [{"role": "user", "content": "hello"}],
    }
    if stream is not None:
        payload["stream"] = stream
    return payload


def _data_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("data: ")]


@pytest.mark.asyncio
async def test_stream_response_formats_each_chunk_as_valid_sse_data():
    req = ChatCompletionRequest(
        model=_TEST_MODEL,
        messages=[ChatMessage(role="user", content="hello")],
        stream=True,
    )
    events = [event async for event in _service().stream_response(req)]

    data_events = events[:-1]
    assert len(data_events) == 2
    for event in data_events:
        assert event.startswith("data: ")
        assert event.endswith("\n\n")
        payload = json.loads(event.removeprefix("data: ").strip())
        assert payload["object"] == "chat.completion.chunk"
        assert payload["choices"][0]["index"] == 0
        assert "content" in payload["choices"][0]["delta"]

    assert json.loads(data_events[0].removeprefix("data: ").strip())["choices"][0][
        "delta"
    ]["content"] == "Hello"
    assert json.loads(data_events[1].removeprefix("data: ").strip())["choices"][0][
        "delta"
    ]["content"] == " world"


@pytest.mark.asyncio
async def test_stream_response_finishes_with_done_terminator():
    req = ChatCompletionRequest(
        model=_TEST_MODEL,
        messages=[ChatMessage(role="user", content="hello")],
        stream=True,
    )
    events = [event async for event in _service().stream_response(req)]

    assert events[-1] == "data: [DONE]\n\n"


def test_streaming_route_returns_event_stream_with_chunks_and_done():
    app.dependency_overrides[get_chat_service] = _service
    app.dependency_overrides[get_registry] = _registry
    try:
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions", json=_payload(stream=True))
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    lines = _data_lines(resp.text)
    assert any(line.startswith("data: {") for line in lines)
    assert lines[-1] == "data: [DONE]"


def test_non_streaming_route_still_returns_json():
    app.dependency_overrides[get_chat_service] = _service
    app.dependency_overrides[get_registry] = _registry
    try:
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions", json=_payload())
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello world"
