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
from inference_x.engines.pool import EnginePool
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
    ChatStreamChunk,
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

    async def generate_stream(self, request: ChatCompletionRequest):
        yield ChatStreamChunk(content="hello ")
        yield ChatStreamChunk(content="world")
        yield ChatStreamChunk(
            content="",
            finish_reason="stop",
            usage=ChatCompletionUsage(
                prompt_tokens=4, completion_tokens=7, total_tokens=11
            ),
        )

    def is_healthy(self) -> bool:
        return self._healthy


def _make_registry() -> ModelRegistry:
    return ModelRegistry([ModelEntry(name=_TEST_MODEL, model_path="test/stub")])


def _make_service() -> ChatService:
    registry = _make_registry()
    return ChatService(
        engine_pool=EnginePool({_TEST_MODEL: _StubEngine()}),
        registry=registry,
        router=TaskRouter(registry, _TEST_MODEL),
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

    def test_summary_ttft_and_tokens_per_sec_absent_when_no_streaming_records(self):
        records = [_record(request_id="1", latency_ms=10.0)]
        svc = self._svc(records)
        s = svc.summary()
        assert s.avg_ttft_ms is None
        assert s.avg_tokens_per_sec is None

    def test_summary_averages_ttft_and_tokens_per_sec_over_streaming_records_only(self):
        records = [
            _record(request_id="1", latency_ms=10.0, ttft_ms=50.0, tokens_per_sec=20.0),
            _record(request_id="2", latency_ms=20.0, ttft_ms=150.0, tokens_per_sec=40.0),
            # Non-streaming request: no ttft/tokens_per_sec — must not skew the average.
            _record(request_id="3", latency_ms=5.0),
        ]
        svc = self._svc(records)
        s = svc.summary()
        assert s.avg_ttft_ms == 100.0
        assert s.avg_tokens_per_sec == 30.0


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

            async def generate_stream(self, req: ChatCompletionRequest):
                raise RuntimeError("boom")
                yield ChatStreamChunk(content="")

            def is_healthy(self) -> bool:
                return True

        def _broken_service() -> ChatService:
            registry = _make_registry()
            return ChatService(
                engine_pool=EnginePool({_TEST_MODEL: _BrokenEngine()}),
                registry=registry,
                router=TaskRouter(registry, _TEST_MODEL),
            )

        app.dependency_overrides[get_chat_service] = _broken_service
        resp = client.post("/v1/chat/completions", json=self._chat_payload)
        assert resp.status_code == 500
        error_records = [r for r in recorder.storage.all() if r.error]
        assert error_records

    def test_streaming_chat_response_unchanged(self, obs_client):
        """SSE body_iterator wrapping must not alter the bytes the client receives."""
        client, _ = obs_client
        payload = {**self._chat_payload, "stream": True}
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert '"content":"hello "' in resp.text
        assert '"content":"world"' in resp.text
        assert resp.text.rstrip().endswith("data: [DONE]")

    def test_streaming_without_include_usage_records_no_token_counts(self, obs_client):
        """No usage event -> no token figures. Absent, never estimated (DEC-049).

        This is the absence half of OS-2's metric truth: before DEC-049 this
        assertion read ``rec.completion_tokens == 2``, which was the whitespace
        word count of "hello " + "world" — a number that had nothing to do with
        tokens. Recording nothing is correct. Do not "fix" this by reinstating a
        fallback estimate in the middleware.
        """
        client, recorder = obs_client
        payload = {**self._chat_payload, "stream": True}
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200

        chat_records = [r for r in recorder.storage.all() if r.path == "/v1/chat/completions"]
        assert len(chat_records) == 1

        rec = chat_records[0]
        assert rec.error is False
        # Timing is still observable without a usage event.
        assert rec.ttft_ms is not None and rec.ttft_ms >= 0
        # Token figures are absent — specifically None, not zero.
        assert rec.completion_tokens is None
        assert rec.total_tokens is None
        assert rec.prompt_tokens is None
        assert rec.tokens_per_sec is None

    def test_streaming_with_include_usage_records_engine_counts(self, obs_client):
        """include_usage -> the recorded counts are the engine's, not an estimate."""
        client, recorder = obs_client
        payload = {
            **self._chat_payload,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200

        chat_records = [r for r in recorder.storage.all() if r.path == "/v1/chat/completions"]
        assert len(chat_records) == 1

        rec = chat_records[0]
        assert rec.error is False
        assert rec.ttft_ms is not None and rec.ttft_ms >= 0
        # The stub engine accounts 7 completion tokens for "hello " + "world";
        # the whitespace word count would have been 2.
        assert rec.completion_tokens == 7
        assert rec.prompt_tokens == 4
        assert rec.total_tokens == 11
        assert rec.tokens_per_sec is not None and rec.tokens_per_sec > 0

    def test_ttft_is_anchored_to_the_first_content_event(self):
        """DEC-053: the pre-generation event must not enter the measurement.

        Left on the first raw chunk, TTFT would silently absorb admission latency
        and every /v1/metrics figure would stop being comparable with the ones
        recorded before the event existed — a discontinuity introduced by
        accident rather than by decision.
        """
        from inference_x.observability.middleware import _has_content_delta

        pre_generation = (
            b'data: {"id":"c","object":"chat.completion.chunk","choices":[],'
            b'"resolved":{"model":"m"},"warnings":[]}'
        )
        content = (
            b'data: {"id":"c","object":"chat.completion.chunk",'
            b'"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}'
        )
        terminal = (
            b'data: {"id":"c","object":"chat.completion.chunk",'
            b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
        )
        usage = (
            b'data: {"id":"c","object":"chat.completion.chunk","choices":[],'
            b'"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}'
        )

        assert _has_content_delta(pre_generation) is False
        assert _has_content_delta(content) is True
        assert _has_content_delta(terminal) is False
        assert _has_content_delta(usage) is False
        assert _has_content_delta(b"data: [DONE]") is False

    def test_pre_generation_event_yields_no_token_figure(self):
        """C4: the usage extractor must ignore it — no `usage` key to find."""
        from inference_x.observability.middleware import _extract_sse_usage

        pre_generation = (
            b'data: {"id":"c","object":"chat.completion.chunk","choices":[],'
            b'"resolved":{"model":"m"},"warnings":[{"type":"degraded",'
            b'"code":"kv_gate_skipped","message":"x","field":null}]}'
        )
        assert _extract_sse_usage(pre_generation) is None

    def test_streaming_mid_stream_engine_failure_still_records_partial_progress(
        self, obs_client
    ):
        """A mid-stream engine failure can't set error=True (see middleware docstring:
        Starlette's BaseHTTPMiddleware only surfaces the inner app's exception *after*
        our dispatch() has already finished sending the response) — but the wrapper
        must still record whatever partial TTFT/token progress it saw, and must not
        crash the request.
        """
        client, recorder = obs_client

        class _BrokenStreamEngine(BaseEngine):
            async def generate(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
                raise RuntimeError("boom")

            async def generate_stream(self, req: ChatCompletionRequest):
                yield ChatStreamChunk(content="partial ")
                raise RuntimeError("boom mid-stream")

            def is_healthy(self) -> bool:
                return True

        def _broken_service() -> ChatService:
            registry = _make_registry()
            return ChatService(
                engine_pool=EnginePool({_TEST_MODEL: _BrokenStreamEngine()}),
                registry=registry,
                router=TaskRouter(registry, _TEST_MODEL),
            )

        app.dependency_overrides[get_chat_service] = _broken_service
        payload = {**self._chat_payload, "stream": True}
        try:
            client.post("/v1/chat/completions", json=payload)
        except Exception:
            pass  # A mid-stream exception after headers are sent may surface client-side.

        chat_records = [r for r in recorder.storage.all() if r.path == "/v1/chat/completions"]
        assert chat_records
        # Partial *timing* progress is still recorded. Token counts are not: the
        # stream died before any usage event, and a truncated stream has no
        # engine-accounted count to report (DEC-049). Previously this asserted
        # completion_tokens == 1, the word count of the one delta that arrived.
        assert chat_records[-1].ttft_ms is not None
        assert chat_records[-1].completion_tokens is None
