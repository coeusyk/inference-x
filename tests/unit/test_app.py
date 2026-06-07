"""Unit tests for Textual playground helper functions."""
from __future__ import annotations

import httpx

from playground.app import InferenceXApp, fetch_health, fetch_models, parse_sse_line, parse_sse_stream


class _FakeClient:
    def __init__(self, responses: dict[str, httpx.Response]) -> None:
        self._responses = responses

    def get(self, url: str, timeout: float) -> httpx.Response:
        response = self._responses[url]
        if response.request is None:
            response.request = httpx.Request("GET", url)
        return response


def _response(url: str, status: int, payload: dict) -> httpx.Response:
    return httpx.Response(status, json=payload, request=httpx.Request("GET", url))


def test_parse_sse_line_extracts_token():
    line = (
        'data: {"id":"x","object":"chat.completion.chunk",'
        '"choices":[{"delta":{"content":"hello"},"index":0}]}'
    )

    assert parse_sse_line(line) == "hello"


def test_parse_sse_line_ignores_done_and_blank_lines():
    assert parse_sse_line("data: [DONE]") is None
    assert parse_sse_line("") is None


def test_parse_sse_line_ignores_malformed_json():
    assert parse_sse_line("data: {not-json") is None


def test_parse_sse_stream_returns_tokens_only():
    lines = [
        'data: {"choices":[{"delta":{"content":"a"},"index":0}]}',
        "event: ignored",
        'data: {"choices":[{"delta":{"content":"b"},"index":0}]}',
        "data: [DONE]",
    ]

    assert parse_sse_stream(lines) == ["a", "b"]


def test_fetch_health_returns_true_for_healthy_server():
    base_url = "http://testserver"
    client = _FakeClient(
        {f"{base_url}/health": _response(f"{base_url}/health", 200, {"status": "healthy"})}
    )

    assert fetch_health(base_url, client) is True


def test_fetch_health_returns_false_for_degraded_server():
    base_url = "http://testserver"
    client = _FakeClient(
        {f"{base_url}/health": _response(f"{base_url}/health", 503, {"status": "degraded"})}
    )

    assert fetch_health(base_url, client) is False


def test_fetch_models_returns_model_ids():
    base_url = "http://testserver"
    client = _FakeClient(
        {
            f"{base_url}/v1/models": _response(
                f"{base_url}/v1/models",
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "qwen2.5-0.5b", "object": "model"},
                        {"id": "tinyllama-chat", "object": "model"},
                    ],
                },
            )
        }
    )

    assert fetch_models(base_url, client) == ["qwen2.5-0.5b", "tinyllama-chat"]


def test_fetch_models_returns_empty_list_on_error_status():
    base_url = "http://testserver"
    client = _FakeClient(
        {f"{base_url}/v1/models": _response(f"{base_url}/v1/models", 500, {"error": {}})}
    )

    assert fetch_models(base_url, client) == []


def test_active_model_in_compare_mode_ignores_stale_index():
    app = InferenceXApp(
        base_url="http://testserver",
        model="qwen2.5-0.5b",
        compare=("qwen2.5-0.5b", "tinyllama-chat"),
    )
    app.current_model_index = 4
    app.models = ["opt-125m", "qwen2.5-0.5b", "tinyllama-chat", "qwen2.5-1.5b", "llama3-8b"]

    assert app._active_model() == "qwen2.5-0.5b"


def test_active_model_clamps_out_of_range_index():
    app = InferenceXApp(base_url="http://testserver", model="qwen2.5-0.5b")
    app.models = ["qwen2.5-0.5b", "tinyllama-chat"]
    app.current_model_index = 9

    assert app._active_model() == "qwen2.5-0.5b"
    assert app.current_model_index == 0
