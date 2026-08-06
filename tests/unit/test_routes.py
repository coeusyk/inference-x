"""Contract tests for /v1/chat/completions, /health, and /v1/models endpoints.

Uses a stub engine injected via FastAPI dependency override so that vLLM
is never imported in this test — it runs on any machine without a GPU.
"""
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY, Gauge

from inference_x.api.deps import get_chat_service, get_engine_pool, get_metrics_service, get_registry
from inference_x.api.main import app
from inference_x.engines.base import BaseEngine
from inference_x.engines.pool import EnginePool
from inference_x.observability.recorder import MetricsRecorder
from inference_x.observability.storage import InMemoryStorage
from inference_x.routing.task_router import TaskRouter
from inference_x.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    ChatStreamChunk,
)
from inference_x.schemas.model import ModelEntry
from inference_x.services.chat_service import ChatService
from inference_x.services.metrics_service import MetricsService
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
        yield ChatStreamChunk(content="Hello ")
        yield ChatStreamChunk(content="from stub")
        yield ChatStreamChunk(
            content="",
            finish_reason="stop",
            usage=ChatCompletionUsage(
                prompt_tokens=2, completion_tokens=3, total_tokens=5
            ),
        )

    def is_healthy(self) -> bool:
        return self._healthy


class _AdmissionAwareEngine(BaseEngine):
    """Stub engine exposing count_prompt_tokens/kv_capacity_tokens so
    AdmissionController's context and KV-saturation gates are exercisable
    from route-level tests (real VLLMEngine isn't importable without a GPU)."""

    def __init__(self, prompt_tokens: int = 5, kv_capacity_tokens: int | None = None) -> None:
        self._healthy = True
        self._prompt_tokens = prompt_tokens
        self.kv_capacity_tokens = kv_capacity_tokens

    def count_prompt_tokens(self, request: ChatCompletionRequest) -> int:
        return self._prompt_tokens

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
        yield ChatStreamChunk(content="ok")
        yield ChatStreamChunk(content="", finish_reason="stop")

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
            assert body["error"]["message"] == "Request could not be processed."
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
                yield ChatStreamChunk(content="")

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

    def test_prompt_exceeding_max_context_tokens_returns_400(self):
        """AdmissionController rejects a prompt over the client's own
        max_context_tokens ceiling with a sanitized 400 (DEC-037 Phase 2)."""
        engine = _AdmissionAwareEngine(prompt_tokens=50)
        registry = _make_stub_registry()
        router = TaskRouter(registry, _TEST_MODEL)
        pool = EnginePool({_TEST_MODEL: engine})
        svc = ChatService(engine_pool=pool, registry=registry, router=router)
        app.dependency_overrides[get_chat_service] = lambda: svc
        with TestClient(app) as c:
            payload = dict(self._payload)
            payload["max_context_tokens"] = 10
            resp = c.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 400
            body = resp.json()
            assert body["error"]["type"] == "invalid_request_error"
        app.dependency_overrides.clear()

    def test_strict_rejects_with_400_where_the_default_clamps(self):
        """§9 C.7: strict: true rejects where the default clamps; both covered.

        DEC-052 — strict may only convert a substitution into a rejection, so the
        default path here must still succeed and report the substitution rather
        than hiding it.
        """
        engine = _AdmissionAwareEngine(prompt_tokens=50)
        registry = _make_stub_registry()
        router = TaskRouter(registry, _TEST_MODEL)
        pool = EnginePool({_TEST_MODEL: engine})
        svc = ChatService(engine_pool=pool, registry=registry, router=router)
        app.dependency_overrides[get_chat_service] = lambda: svc
        with TestClient(app) as c:
            payload = dict(self._payload)
            payload["max_context_tokens"] = 200
            payload["max_tokens"] = 4000

            lenient = c.post("/v1/chat/completions", json=payload)
            assert lenient.status_code == 200
            codes = {w["code"] for w in lenient.json()["warnings"]}
            assert "max_tokens_clamped_to_context" in codes

            strict = c.post("/v1/chat/completions", json={**payload, "strict": True})
            assert strict.status_code == 400
            assert strict.json()["error"]["type"] == "invalid_request_error"
        app.dependency_overrides.clear()

    def test_kv_saturation_returns_429_with_retry_after(self):
        """A batch-tier request is rejected (not silently truncated) when an
        in-flight reservation has already consumed the KV safety budget."""
        engine = _AdmissionAwareEngine(prompt_tokens=5, kv_capacity_tokens=20)
        registry = _make_stub_registry()
        router = TaskRouter(registry, _TEST_MODEL)
        pool = EnginePool({_TEST_MODEL: engine})
        svc = ChatService(engine_pool=pool, registry=registry, router=router)
        # Simulate an in-flight request holding most of the KV budget
        # (safety margin 0.9 * capacity 20 = 18 tokens) without releasing it.
        svc._admission.admit(
            _TEST_MODEL,
            ChatCompletionRequest(
                model=_TEST_MODEL,
                messages=[ChatMessage(role="user", content="x")],
                max_tokens=12,
                priority="batch",
            ),
            engine,
        )
        app.dependency_overrides[get_chat_service] = lambda: svc
        with TestClient(app) as c:
            payload = dict(self._payload)
            payload["max_tokens"] = 10
            payload["priority"] = "batch"
            resp = c.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 429
            assert "Retry-After" in resp.headers
            body = resp.json()
            assert body["error"]["type"] == "rate_limit_error"
        app.dependency_overrides.clear()

    def test_sequence_concurrency_saturation_returns_429(self):
        """An interactive-priority request is also rejected (no clamp path exists
        for a sequence slot) when the resolved max_num_seqs ceiling is already
        full (add-engine-knob-surfacing)."""
        from dataclasses import dataclass

        from inference_x.routing.admission import AdmissionController

        @dataclass
        class _FakeTier:
            max_model_len_cap: int
            max_num_seqs: int

        engine = _AdmissionAwareEngine(prompt_tokens=5)
        registry = _make_stub_registry()
        router = TaskRouter(registry, _TEST_MODEL)
        pool = EnginePool({_TEST_MODEL: engine})
        admission = AdmissionController(
            registry, tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1)
        )
        svc = ChatService(engine_pool=pool, registry=registry, router=router, admission=admission)
        # Simulate one in-flight request already holding the model's only sequence slot.
        admission.admit(
            _TEST_MODEL,
            ChatCompletionRequest(
                model=_TEST_MODEL, messages=[ChatMessage(role="user", content="x")]
            ),
            engine,
        )
        app.dependency_overrides[get_chat_service] = lambda: svc
        with TestClient(app) as c:
            resp = c.post("/v1/chat/completions", json=self._payload)
            assert resp.status_code == 429
            assert "Retry-After" in resp.headers
            body = resp.json()
            assert body["error"]["type"] == "rate_limit_error"
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
        assert model["quantization"] is None
        assert model["estimated_weights_gib"] > 0

    def test_model_object_reflects_quantization_and_max_model_len(self, client):
        """A quantized entry must surface its variant + context cap for client selection."""
        registry = ModelRegistry(
            [
                ModelEntry(
                    name="quant-model",
                    model_path="Qwen/Qwen2.5-7B-Instruct-AWQ",
                    quantization="awq",
                    max_model_len=4096,
                )
            ]
        )
        app.dependency_overrides[get_registry] = lambda: registry
        with TestClient(app) as c:
            resp = c.get("/v1/models")
            model = resp.json()["data"][0]
            assert model["quantization"] == "awq"
            assert model["max_model_len"] == 4096
            assert model["estimated_weights_gib"] > 0
        app.dependency_overrides.clear()

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


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    @pytest.fixture()
    def metrics_client(self):
        registry = _make_stub_registry()
        pool = EnginePool({_TEST_MODEL: _StubEngine()})
        recorder = MetricsRecorder(storage=InMemoryStorage())

        app.dependency_overrides[get_registry] = lambda: registry
        app.dependency_overrides[get_engine_pool] = lambda: pool
        app.dependency_overrides[get_metrics_service] = lambda: MetricsService(recorder)
        app.dependency_overrides[get_chat_service] = _stub_service_factory(healthy=True)
        with TestClient(app) as c:
            yield c, recorder
        app.dependency_overrides.clear()

    def test_returns_200(self, metrics_client):
        client, _ = metrics_client
        resp = client.get("/v1/metrics")
        assert resp.status_code == 200

    def test_empty_metrics_shape(self, metrics_client):
        client, _ = metrics_client
        resp = client.get("/v1/metrics")
        body = resp.json()
        assert body["total_requests"] == 0
        assert body["avg_latency_ms"] is None
        assert body["avg_ttft_ms"] is None

    def test_vram_breakdown_lists_loaded_stub_model(self, metrics_client):
        client, _ = metrics_client
        resp = client.get("/v1/metrics")
        body = resp.json()
        model_names = [m["name"] for m in body["vram"]["models"]]
        assert _TEST_MODEL in model_names
        entry = next(m for m in body["vram"]["models"] if m["name"] == _TEST_MODEL)
        assert entry["estimated_weights_gib"] > 0
        # _StubEngine has no kv_capacity_tokens attribute — must degrade to None, not error.
        assert entry["kv_capacity_tokens"] is None

    def test_reflects_recorded_requests(self, metrics_client):
        client, recorder = metrics_client
        recorder.record(path="/v1/chat/completions", method="POST", status_code=200, latency_ms=12.0)
        resp = client.get("/v1/metrics")
        body = resp.json()
        assert body["total_requests"] == 1
        assert body["avg_latency_ms"] == 12.0


# ---------------------------------------------------------------------------
# Native Prometheus metrics endpoint (GET /metrics) — expose-native-engine-metrics
# ---------------------------------------------------------------------------

class TestNativeMetricsEndpoint:
    """GET /metrics: vLLM's native Prometheus stat logger, mounted as a
    passthrough (see openspec/changes/expose-native-engine-metrics). Distinct
    from GET /v1/metrics (InferenceX's own JSON summary, tested above).
    """

    @pytest.fixture()
    def native_metrics_client(self):
        registry = _make_stub_registry()
        pool = EnginePool({_TEST_MODEL: _StubEngine()})
        recorder = MetricsRecorder(storage=InMemoryStorage())

        app.dependency_overrides[get_registry] = lambda: registry
        app.dependency_overrides[get_engine_pool] = lambda: pool
        app.dependency_overrides[get_metrics_service] = lambda: MetricsService(recorder)
        app.dependency_overrides[get_chat_service] = _stub_service_factory(healthy=True)
        with TestClient(app) as c:
            yield c, recorder
        app.dependency_overrides.clear()

    def test_returns_200(self, native_metrics_client):
        client, _ = native_metrics_client
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_content_type_is_prometheus_exposition(self, native_metrics_client):
        client, _ = native_metrics_client
        resp = client.get("/metrics")
        content_type = resp.headers["content-type"]
        assert content_type.startswith("text/plain")
        assert "version=0.0.4" in content_type

    def test_contains_a_vllm_prefixed_metric(self, native_metrics_client):
        """No real AsyncLLM is constructed in this unit test (per this module's
        own "vLLM is never imported" contract), so there is no real
        PrometheusStatLogger to produce vllm: series here. Instead this
        registers a vllm:-prefixed Gauge directly onto prometheus_client's
        global REGISTRY — exactly where design.md Decision 1 verified vLLM's
        own default stat logger registers its series — and asserts the mount
        surfaces it. This proves the mount is a correct, unfiltered passthrough
        of whatever the registry holds, independent of vLLM being importable.
        """
        client, _ = native_metrics_client
        gauge = Gauge("vllm:test_probe_metric", "test probe", registry=REGISTRY)
        gauge.set(1)
        try:
            resp = client.get("/metrics")
            assert "vllm:test_probe_metric" in resp.text
        finally:
            REGISTRY.unregister(gauge)

    def test_scrape_does_not_affect_metrics_service_summary(self, native_metrics_client):
        client, recorder = native_metrics_client
        recorder.record(path="/v1/chat/completions", method="POST", status_code=200, latency_ms=12.0)

        service = MetricsService(recorder)
        before = service.summary()

        for _ in range(3):
            resp = client.get("/metrics")
            assert resp.status_code == 200

        after = service.summary()
        assert after.total_requests == before.total_requests
        assert after.avg_latency_ms == before.avg_latency_ms
        assert after.error_count == before.error_count
        assert len(recorder.storage.all()) == 1
