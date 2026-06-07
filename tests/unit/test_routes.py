"""Contract tests for /v1/chat/completions and /health endpoints.

Uses a stub engine injected via FastAPI dependency override so that vLLM
is never imported in this test — it runs on any machine without a GPU.
"""
import pytest
from fastapi.testclient import TestClient

from inference_x.api.main import app
from inference_x.api.deps import get_chat_service
from inference_x.engines.base import BaseEngine
from inference_x.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
)
from inference_x.services.chat_service import ChatService


class _StubEngine(BaseEngine):
    def __init__(self, healthy: bool = True) -> None:
        self._healthy = healthy

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content="Hello from stub"),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=4, completion_tokens=4, total_tokens=8
            ),
        )

    def is_healthy(self) -> bool:
        return self._healthy


def _stub_service_factory(healthy: bool = True):
    def _override() -> ChatService:
        return ChatService(_StubEngine(healthy=healthy))

    return _override


@pytest.fixture()
def client():
    app.dependency_overrides[get_chat_service] = _stub_service_factory(healthy=True)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def unhealthy_client():
    app.dependency_overrides[get_chat_service] = _stub_service_factory(healthy=False)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    def test_health_returns_200_when_healthy(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["engine"] == "ok"

    def test_health_returns_degraded_when_engine_unhealthy(self, unhealthy_client):
        resp = unhealthy_client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["engine"] == "unavailable"

    def test_health_returns_500_when_engine_fails_to_load(self):
        def _fail_service() -> ChatService:
            raise RuntimeError("vLLM initialization failed")

        app.dependency_overrides[get_chat_service] = _fail_service
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 500
            body = resp.json()
            assert body["error"]["type"] == "internal_error"
            assert "vLLM initialization failed" in body["error"]["message"]
        app.dependency_overrides.clear()


class TestChatCompletionsEndpoint:
    _payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
    }

    def test_returns_200_with_valid_request(self, client):
        resp = client.post("/v1/chat/completions", json=self._payload)
        assert resp.status_code == 200

    def test_response_schema(self, client):
        resp = client.post("/v1/chat/completions", json=self._payload)
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "test-model"
        assert len(body["choices"]) == 1
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "Hello from stub"
        assert body["usage"]["total_tokens"] == 8

    def test_missing_messages_returns_422(self, client):
        resp = client.post("/v1/chat/completions", json={"model": "m"})
        assert resp.status_code == 422

    def test_invalid_role_returns_422(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "bad", "content": "hi"}]},
        )
        assert resp.status_code == 422

    def test_temperature_out_of_range_returns_422(self, client):
        payload = dict(self._payload)
        payload["temperature"] = 5.0
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422

    def test_stream_true_returns_400(self, client):
        payload = dict(self._payload)
        payload["stream"] = True
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert "Streaming" in body["error"]["message"]

    def test_engine_failure_returns_structured_500(self, client):
        class _FailingEngine(BaseEngine):
            async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
                raise RuntimeError("inference exploded")

            def is_healthy(self) -> bool:
                return True

        app.dependency_overrides[get_chat_service] = lambda: ChatService(_FailingEngine())
        with TestClient(app) as c:
            resp = c.post("/v1/chat/completions", json=self._payload)
            assert resp.status_code == 500
            body = resp.json()
            assert body["error"]["type"] == "internal_error"
        app.dependency_overrides.clear()
