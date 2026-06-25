"""Unit tests for chat schemas."""
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

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


class _StubEngine(BaseEngine):
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
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
        yield "ok"

    def is_healthy(self) -> bool:
        return True


def _registry() -> ModelRegistry:
    return ModelRegistry([ModelEntry(name=_TEST_MODEL, model_path="test/stub")])


def _service() -> ChatService:
    registry = _registry()
    router = TaskRouter(registry, _TEST_MODEL)
    pool = EnginePool({_TEST_MODEL: _StubEngine()})
    return ChatService(engine_pool=pool, registry=registry, router=router)


class TestChatMessage:
    def test_valid_roles(self):
        for role in ("system", "user", "assistant"):
            msg = ChatMessage(role=role, content="hello")
            assert msg.role == role

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="unknown", content="hi")


class TestChatCompletionRequest:
    def test_minimal_request(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="hello")],
        )
        assert req.model == "test-model"
        assert req.temperature == 0.7
        assert req.max_tokens == 512
        assert req.stream is False

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x")],
                temperature=3.0,
            )

    def test_temperature_below_zero_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x")],
                temperature=-0.1,
            )

    def test_max_tokens_must_be_positive(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x")],
                max_tokens=0,
            )

    def test_max_tokens_exceeds_4096_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x")],
                max_tokens=4097,
            )

    def test_content_exceeds_max_length_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="x" * 32001)

    def test_content_at_max_length_is_valid(self):
        msg = ChatMessage(role="user", content="x" * 32000)
        assert len(msg.content) == 32000

    def test_temperature_at_boundaries_is_valid(self):
        req_min = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            temperature=0.0,
        )
        req_max = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            temperature=2.0,
        )
        assert req_min.temperature == 0.0
        assert req_max.temperature == 2.0

    def test_max_tokens_at_boundary_is_valid(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            max_tokens=4096,
        )
        assert req.max_tokens == 4096


class TestSchemaConstraintsViaAPI:
    """Verify that out-of-range values produce HTTP 422 when sent through the API."""

    def _client(self) -> TestClient:
        app.dependency_overrides[get_chat_service] = _service
        app.dependency_overrides[get_registry] = _registry
        return TestClient(app)

    def teardown_method(self):
        app.dependency_overrides.clear()

    def _base_payload(self) -> dict:
        return {
            "model": _TEST_MODEL,
            "messages": [{"role": "user", "content": "hello"}],
        }

    def test_temperature_above_2_returns_422(self):
        client = self._client()
        payload = self._base_payload()
        payload["temperature"] = 3.0
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422

    def test_temperature_below_0_returns_422(self):
        client = self._client()
        payload = self._base_payload()
        payload["temperature"] = -0.5
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422

    def test_max_tokens_above_4096_returns_422(self):
        client = self._client()
        payload = self._base_payload()
        payload["max_tokens"] = 5000
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422

    def test_max_tokens_zero_returns_422(self):
        client = self._client()
        payload = self._base_payload()
        payload["max_tokens"] = 0
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422

    def test_content_exceeds_32000_chars_returns_422(self):
        client = self._client()
        payload = self._base_payload()
        payload["messages"] = [{"role": "user", "content": "x" * 32001}]
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422

    def test_valid_request_returns_200(self):
        client = self._client()
        resp = client.post("/v1/chat/completions", json=self._base_payload())
        assert resp.status_code == 200


class TestChatCompletionResponse:
    def _make_response(self) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            model="test-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content="hello"),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=5,
                completion_tokens=3,
                total_tokens=8,
            ),
        )

    def test_response_structure(self):
        resp = self._make_response()
        assert resp.object == "chat.completion"
        assert resp.model == "test-model"
        assert len(resp.choices) == 1
        assert resp.choices[0].finish_reason == "stop"
        assert resp.usage.total_tokens == 8

    def test_auto_id_generated(self):
        r1 = self._make_response()
        r2 = self._make_response()
        assert r1.id.startswith("chatcmpl-")
        assert r1.id != r2.id

    def test_serializes_to_dict(self):
        resp = self._make_response()
        data = resp.model_dump()
        assert "choices" in data
        assert "usage" in data
