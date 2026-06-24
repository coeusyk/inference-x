"""Unit tests for playground/streaming.py."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "playground"))

import streaming as pg_streaming  # noqa: E402


def _sse_lines(*chunks: str) -> list[str]:
    return [f'data: {{"choices":[{{"delta":{{"content":"{c}"}},"index":0}}]}}' for c in chunks]


class _FakeStreamResponse:
    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code
        self.request = httpx.Request("POST", "http://test/v1/chat/completions")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://test/v1/chat/completions")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeAsyncClient:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def stream(self, method: str, url: str, json: dict):
        return self._response


@pytest.mark.asyncio
async def test_stream_chat_tokens_yields_tokens(monkeypatch):
    lines = _sse_lines("a", "b") + ["data: [DONE]"]
    fake_client = _FakeAsyncClient(_FakeStreamResponse(lines))

    class ClientFactory:
        async def __aenter__(self):
            return fake_client

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(pg_streaming.httpx, "AsyncClient", lambda **kwargs: ClientFactory())

    tokens = [
        token
        async for token in pg_streaming.stream_chat_tokens(
            "http://test", {"model": "m", "messages": [], "stream": True}
        )
    ]
    assert tokens == ["a", "b"]


@pytest.mark.asyncio
async def test_stream_chat_tokens_stops_on_done(monkeypatch):
    lines = _sse_lines("only") + ["data: [DONE]", _sse_lines("ignored")[0]]
    fake_client = _FakeAsyncClient(_FakeStreamResponse(lines))

    class ClientFactory:
        async def __aenter__(self):
            return fake_client

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(pg_streaming.httpx, "AsyncClient", lambda **kwargs: ClientFactory())

    tokens = [
        token
        async for token in pg_streaming.stream_chat_tokens(
            "http://test", {"model": "m", "messages": [], "stream": True}
        )
    ]
    assert tokens == ["only"]


@pytest.mark.asyncio
async def test_stream_chat_tokens_skips_blank_and_non_data_lines(monkeypatch):
    lines = [
        "",
        "event: ping",
        _sse_lines("tok")[0],
        "data: [DONE]",
    ]
    fake_client = _FakeAsyncClient(_FakeStreamResponse(lines))

    class ClientFactory:
        async def __aenter__(self):
            return fake_client

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(pg_streaming.httpx, "AsyncClient", lambda **kwargs: ClientFactory())

    tokens = [
        token
        async for token in pg_streaming.stream_chat_tokens(
            "http://test", {"model": "m", "messages": [], "stream": True}
        )
    ]
    assert tokens == ["tok"]


@pytest.mark.asyncio
async def test_stream_chat_tokens_raises_on_http_error(monkeypatch):
    fake_client = _FakeAsyncClient(_FakeStreamResponse([], status_code=500))

    class ClientFactory:
        async def __aenter__(self):
            return fake_client

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(pg_streaming.httpx, "AsyncClient", lambda **kwargs: ClientFactory())

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in pg_streaming.stream_chat_tokens(
            "http://test", {"model": "m", "messages": [], "stream": True}
        ):
            pass
