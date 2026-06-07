"""Unit tests for observability layer: storage, recorder, exporters, middleware."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from inference_x.api.deps import get_chat_service, get_recorder, get_registry
from inference_x.api.main import app
from inference_x.engines.base import BaseEngine
from inference_x.observability.exporters import JsonLineExporter, NullExporter, build_exporter
from inference_x.observability.recorder import MetricsRecorder
from inference_x.observability.storage import InMemoryStorage, RequestRecord
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
from inference_x.services.metrics_service import MetricsService
from inference_x.services.model_service import ModelRegistry

_TEST_MODEL = "test-model"


# ---------------------------------------------------------------------------
# Shared stubs
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
                    message=ChatCompletionMessage(content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )

    def is_healthy(self) -> bool:
        return self._healthy


def _make_registry() -> ModelRegistry:
    return ModelRegistry([ModelEntry(name=_TEST_MODEL, model_path="test/stub")])


def _make_service() -> ChatService:
    registry = _make_registry()
    return ChatService(
        engine=_StubEngine(),
        registry=registry,
        router=TaskRouter(registry, _TEST_MODEL),
        loaded_model=_TEST_MODEL,
    )


# ---------------------------------------------------------------------------
# InMemoryStorage
# ---------------------------------------------------------------------------

def _record(**kwargs) -> RequestRecord:
    defaults = dict(
        request_id="rid",
        path="/test",
        method="GET",
        status_code=200,
        latency_ms=10.0,
    )
    defaults.update(kwargs)
    return RequestRecord(**defaults)


class TestInMemoryStorage:
    def test_append_and_all(self):
        s = InMemoryStorage()
        r = _record()
        s.append(r)
        assert s.all() == [r]

    def test_len(self):
        s = InMemoryStorage()
        s.append(_record())
        s.append(_record(request_id="r2"))
        assert len(s) == 2

    def test_recent(self):
        s = InMemoryStorage()
        for i in range(5):
            s.append(_record(request_id=str(i)))
        recent = s.recent(3)
        assert len(recent) == 3
        assert [r.request_id for r in recent] == ["2", "3", "4"]

    def test_cap_drops_oldest(self):
        s = InMemoryStorage(max_records=3)
        for i in range(5):
            s.append(_record(request_id=str(i)))
        ids = [r.request_id for r in s.all()]
        assert ids == ["2", "3", "4"]

    def test_clear(self):
        s = InMemoryStorage()
        s.append(_record())
        s.clear()
        assert len(s) == 0

    def test_thread_safety(self):
        import threading

        s = InMemoryStorage()
        errors = []

        def _write():
            try:
                for i in range(50):
                    s.append(_record(request_id=str(i)))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_write) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ---------------------------------------------------------------------------
# RequestRecord
# ---------------------------------------------------------------------------

class TestRequestRecord:
    def test_to_dict_flat(self):
        r = _record(model="m", prompt_tokens=1, completion_tokens=2, total_tokens=3)
        d = r.to_dict()
        assert d["model"] == "m"
        assert d["prompt_tokens"] == 1
        assert d["total_tokens"] == 3
        assert d["error"] is False


# ---------------------------------------------------------------------------
# MetricsRecorder
# ---------------------------------------------------------------------------

class TestMetricsRecorder:
    def _recorder(self) -> MetricsRecorder:
        return MetricsRecorder(storage=InMemoryStorage())

    def test_record_stores_entry(self):
        rec = self._recorder()
        rec.record(path="/test", method="GET", status_code=200, latency_ms=15.0)
        assert len(rec.storage) == 1

    def test_record_returns_record(self):
        rec = self._recorder()
        r = rec.record(path="/p", method="POST", status_code=200, latency_ms=5.5)
        assert isinstance(r, RequestRecord)
        assert r.latency_ms == 5.5

    def test_record_with_tokens(self):
        rec = self._recorder()
        r = rec.record(
            path="/v1/chat/completions",
            method="POST",
            status_code=200,
            latency_ms=100.0,
            model="qwen",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        assert r.prompt_tokens == 10
        assert r.total_tokens == 30

    def test_storage_error_is_swallowed(self):
        bad_storage = MagicMock()
        bad_storage.append.side_effect = RuntimeError("disk full")
        rec = MetricsRecorder(storage=bad_storage)
        rec.record(path="/", method="GET", status_code=200, latency_ms=1.0)

    def test_exporter_is_called(self):
        mock_exporter = MagicMock()
        rec = MetricsRecorder(storage=InMemoryStorage(), exporter=mock_exporter)
        rec.record(path="/", method="GET", status_code=200, latency_ms=1.0)
        mock_exporter.export.assert_called_once()

    def test_exporter_error_is_swallowed(self):
        mock_exporter = MagicMock()
        mock_exporter.export.side_effect = IOError("no space")
        rec = MetricsRecorder(storage=InMemoryStorage(), exporter=mock_exporter)
        rec.record(path="/", method="GET", status_code=200, latency_ms=1.0)


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------

class TestNullExporter:
    def test_no_op(self):
        NullExporter().export(_record())


class TestJsonLineExporter:
    def test_writes_ndjson(self, tmp_path):
        f = tmp_path / "metrics.jsonl"
        exp = JsonLineExporter(f)
        r = _record(model="m", request_id="abc123")
        exp.export(r)
        lines = f.read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["request_id"] == "abc123"
        assert parsed["model"] == "m"

    def test_appends_multiple(self, tmp_path):
        f = tmp_path / "m.jsonl"
        exp = JsonLineExporter(f)
        exp.export(_record(request_id="1"))
        exp.export(_record(request_id="2"))
        lines = f.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_bad_path_does_not_raise(self):
        exp = JsonLineExporter("/nonexistent/path/metrics.jsonl")
        exp.export(_record())

    def test_build_exporter_null_by_default(self, monkeypatch):
        monkeypatch.delenv("INFERENCE_X_METRICS_FILE", raising=False)
        assert isinstance(build_exporter(), NullExporter)

    def test_build_exporter_json_when_env_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("INFERENCE_X_METRICS_FILE", str(tmp_path / "out.jsonl"))
        assert isinstance(build_exporter(), JsonLineExporter)


# ---------------------------------------------------------------------------
# MetricsService
# ---------------------------------------------------------------------------

class TestMetricsService:
    def _svc(self, records: list[RequestRecord]) -> MetricsService:
        s = InMemoryStorage()
        for r in records:
            s.append(r)
        rec = MetricsRecorder(storage=s)
        return MetricsService(rec)

    def test_recent(self):
        records = [_record(request_id=str(i)) for i in range(10)]
        svc = self._svc(records)
        assert len(svc.recent(5)) == 5

    def test_all(self):
        records = [_record(request_id=str(i)) for i in range(3)]
        svc = self._svc(records)
        assert len(svc.all()) == 3

    def test_summary_empty(self):
        svc = self._svc([])
        s = svc.summary()
        assert s.total_requests == 0
        assert s.avg_latency_ms is None

    def test_summary_with_records(self):
        records = [
            _record(request_id="1", latency_ms=10.0, error=False),
            _record(request_id="2", latency_ms=20.0, error=True),
        ]
        svc = self._svc(records)
        s = svc.summary()
        assert s.total_requests == 2
        assert s.error_count == 1
        assert s.avg_latency_ms == 15.0


# ---------------------------------------------------------------------------
# Middleware integration via TestClient
# ---------------------------------------------------------------------------

@pytest.fixture()
def obs_client():
    """Test client with stub service.

    The middleware was wired at app creation with deps.get_recorder() — the
    singleton recorder.  We clear its storage before each test so records from
    prior tests don't leak, then yield the same recorder so assertions can
    inspect what the middleware actually wrote.
    """
    from inference_x.api.deps import get_recorder as _get_recorder

    recorder = _get_recorder()
    recorder.storage.clear()

    app.dependency_overrides[get_chat_service] = lambda: _make_service()
    app.dependency_overrides[get_registry] = _make_registry

    with TestClient(app) as c:
        yield c, recorder

    app.dependency_overrides.clear()


class TestObservabilityMiddleware:
    _chat_payload = {
        "model": _TEST_MODEL,
        "messages": [{"role": "user", "content": "hello"}],
    }

    def test_health_request_is_recorded(self, obs_client):
        client, recorder = obs_client
        client.get("/health")
        records = recorder.storage.all()
        paths = [r.path for r in records]
        assert "/health" in paths

    def test_chat_request_is_recorded(self, obs_client):
        client, recorder = obs_client
        resp = client.post("/v1/chat/completions", json=self._chat_payload)
        assert resp.status_code == 200
        records = recorder.storage.all()
        chat_records = [r for r in records if r.path == "/v1/chat/completions"]
        assert chat_records

    def test_latency_is_positive(self, obs_client):
        client, recorder = obs_client
        client.get("/health")
        records = recorder.storage.all()
        assert all(r.latency_ms > 0 for r in records)

    def test_response_body_unchanged_after_middleware(self, obs_client):
        """Response received by client must be identical to what the handler returns."""
        client, _ = obs_client
        resp = client.post("/v1/chat/completions", json=self._chat_payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "ok"
        assert body["usage"]["total_tokens"] == 8

    def test_content_length_recalculated_after_body_buffer(self, obs_client):
        """Starlette must set content-length to match the re-wrapped response body."""
        client, _ = obs_client
        resp = client.post("/v1/chat/completions", json=self._chat_payload)
        assert resp.status_code == 200
        content_length = resp.headers.get("content-length")
        assert content_length is not None
        assert int(content_length) == len(resp.content)

    def test_models_endpoint_is_recorded(self, obs_client):
        client, recorder = obs_client
        client.get("/v1/models")
        paths = [r.path for r in recorder.storage.all()]
        assert "/v1/models" in paths

    def test_status_code_recorded_correctly(self, obs_client):
        client, recorder = obs_client
        client.get("/health")
        health_records = [r for r in recorder.storage.all() if r.path == "/health"]
        assert health_records[0].status_code == 200

    def test_error_flag_set_on_500(self, obs_client):
        client, recorder = obs_client

        class _BrokenEngine(BaseEngine):
            async def generate(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
                raise RuntimeError("boom")

            def is_healthy(self) -> bool:
                return True

        def _broken_service() -> ChatService:
            registry = _make_registry()
            return ChatService(
                engine=_BrokenEngine(),
                registry=registry,
                router=TaskRouter(registry, _TEST_MODEL),
                loaded_model=_TEST_MODEL,
            )

        app.dependency_overrides[get_chat_service] = _broken_service
        resp = client.post("/v1/chat/completions", json=self._chat_payload)
        assert resp.status_code == 500
        error_records = [r for r in recorder.storage.all() if r.error]
        assert error_records
