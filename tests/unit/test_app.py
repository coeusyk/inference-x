"""Unit tests for Textual playground helper functions."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "playground"))

import app as app_module
from app import InferenceXApp, fetch_health, fetch_models, parse_sse_line, parse_sse_stream
from url_validation import validate_base_url


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
        compare=("qwen2.5-0.5b", "tinyllama-chat"),
    )

    assert app.compare == ("qwen2.5-0.5b", "tinyllama-chat")


def test_validate_base_url_accepts_public_host():
    assert validate_base_url("https://api.example.com") is None


def test_validate_base_url_rejects_loopback_without_flag():
    err = validate_base_url("http://127.0.0.1:8000")
    assert err is not None
    assert "internal" in err.lower()


def test_validate_base_url_rejects_rfc1918_without_flag():
    err = validate_base_url("http://192.168.1.10:8000")
    assert err is not None
    assert "internal" in err.lower()


def test_validate_base_url_rejects_link_local_without_flag():
    err = validate_base_url("http://169.254.169.254")
    assert err is not None


def test_validate_base_url_allows_loopback_with_flag():
    assert validate_base_url("http://127.0.0.1:8000", allow_internal=True) is None
    assert validate_base_url("http://localhost:8000", allow_internal=True) is None


def test_validate_base_url_rejects_non_http_scheme():
    err = validate_base_url("file:///etc/passwd")
    assert err is not None
    assert "scheme" in err.lower()


async def test_ctrl_c_quits_even_with_failed_loading_screen_on_top(monkeypatch):
    """Regression (2026-07-03, DEC-046): Textual's Screen/ModalScreen classes
    claim plain ctrl+c for copy_text, which silently shadowed InferenceXApp's
    own non-priority ctrl+c->quit binding whenever any screen (e.g. a failed
    LoadingScreen) was pushed on top — the app became unkillable via ctrl+c.
    Fixed by marking the app's binding priority=True, which Textual checks
    app-wide before the focus-chain walk that let the screen's binding win.
    """

    async def fake_ensure_models_loaded(base_url, models, *, on_status=None, on_log=None, load_timeout_s=600):
        if on_log:
            on_log("X Application startup failed")
        return None

    import log_feed

    monkeypatch.setattr(app_module, "ensure_models_loaded", fake_ensure_models_loaded)
    monkeypatch.setattr(
        log_feed,
        "extract_error_summary",
        lambda *a, **k: "cannot load sequentially on a 8 GiB GPU",
    )

    app = InferenceXApp(base_url="http://127.0.0.1:8000", compare=("qwen2.5-0.5b", "qwen2.5-1.5b"))
    async with app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
        assert any(screen.__class__.__name__ == "LoadingScreen" for screen in app.screen_stack)

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app.is_running is False
