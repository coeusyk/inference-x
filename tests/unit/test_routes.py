"""Contract tests for /v1/chat/completions, /health, and /v1/models endpoints.

Uses a stub engine injected via FastAPI dependency override so that vLLM
is never imported in this test — it runs on any machine without a GPU.
"""
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
)
from inference_x.schemas.model import ModelEntry
from inference_x.services.chat_service import ChatService
from inference_x.services.model_service import ModelRegistry

_TEST_MODEL = "test-model"


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

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

    async def generate_stream(self, request: ChatCompletionRequest):
        yield "Hello "
        yield "from stub"

    def is_healthy(self) -> bool:
        return self._healthy


def _make_stub_registry() -> ModelRegistry:
    return ModelRegistry([ModelEntry(name=_TEST_MODEL, model_path="test/stub")])


def _make_stub_service(healthy: bool = True) -> ChatService:
    registry = _make_stub_registry()
    router = TaskRouter(registry, _TEST_MODEL)
    pool = EnginePool({_TEST_MODEL: _StubEngine(healthy=healthy)})
    return ChatService(engine_pool=pool, registry=registry, router=router)


def _stub_service_factory(healthy: bool = True):
    def _override() -> ChatService:
        return _make_stub_service(healthy=healthy)
    return _override


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    app.dependency_overrides[get_chat_service] = _stub_service_factory(healthy=True)
    app.dependency_overrides[get_registry] = _make_stub_registry
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def unhealthy_client():
    app.dependency_overrides[get_chat_service] = _stub_service_factory(healthy=False)
    app.dependency_overrides[get_registry] = _make_stub_registry
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200_when_healthy(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["engine"] == "ok"
        assert _TEST_MODEL in body["loaded_models"]

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


# ---------------------------------------------------------------------------
# Chat completions endpoint
# ---------------------------------------------------------------------------

class TestChatCompletionsEndpoint:
    _payload = {
        "model": _TEST_MODEL,
        "messages": [{"role": "user", "content": "hello"}],
    }

    def test_returns_200_with_valid_request(self, client):
        resp = client.post("/v1/chat/completions", json=self._payload)
        assert resp.status_code == 200

    def test_response_schema(self, client):
        resp = client.post("/v1/chat/completions", json=self._payload)
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == _TEST_MODEL
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

    def test_stream_true_returns_event_stream(self, client):
        payload = dict(self._payload)
        payload["stream"] = True
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in resp.text.splitlines() if line.startswith("data: ")]
        assert lines[0].startswith("data: {")
        assert lines[-1] == "data: [DONE]"

    def test_engine_failure_returns_structured_500(self, client):
        class _FailingEngine(BaseEngine):
            async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
                raise RuntimeError("inference exploded")

            async def generate_stream(self, request: ChatCompletionRequest):
                raise RuntimeError("inference exploded")
                yield ""

            def is_healthy(self) -> bool:
                return True

        registry = _make_stub_registry()
        router = TaskRouter(registry, _TEST_MODEL)
        pool = EnginePool({_TEST_MODEL: _FailingEngine()})
        failing_svc = ChatService(engine_pool=pool, registry=registry, router=router)
        app.dependency_overrides[get_chat_service] = lambda: failing_svc
        with TestClient(app) as c:
            resp = c.post("/v1/chat/completions", json=self._payload)
            assert resp.status_code == 500
            body = resp.json()
            assert body["error"]["type"] == "internal_error"
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------

class TestModelsEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_returns_list_object(self, client):
        resp = client.get("/v1/models")
        body = resp.json()
        assert body["object"] == "list"
        assert isinstance(body["data"], list)

    def test_contains_stub_model(self, client):
        resp = client.get("/v1/models")
        body = resp.json()
        ids = [m["id"] for m in body["data"]]
        assert _TEST_MODEL in ids

    def test_model_object_shape(self, client):
        resp = client.get("/v1/models")
        body = resp.json()
        model = body["data"][0]
        assert "id" in model
        assert model["object"] == "model"
        assert model["owned_by"] == "inferencex"

    def test_reflects_models_yaml(self, client):
        """Models endpoint should list every entry from config/models.yaml."""
        from inference_x.api.deps import get_registry
        from inference_x.services.model_service import ModelRegistry

        real_registry = ModelRegistry.from_config("config")
        app.dependency_overrides[get_registry] = lambda: real_registry
        with TestClient(app) as c:
            resp = c.get("/v1/models")
            assert resp.status_code == 200
            ids = {m["id"] for m in resp.json()["data"]}
            assert ids == set(real_registry.names())
        app.dependency_overrides.clear()
