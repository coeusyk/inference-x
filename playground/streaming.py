"""Shared SSE streaming helpers for playground TUIs."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from app import fetch_health, fetch_models, parse_sse_line, parse_sse_stream

__all__ = [
    "fetch_health",
    "fetch_models",
    "parse_sse_line",
    "parse_sse_stream",
    "stream_chat_tokens",
]


async def stream_chat_tokens(
    base_url: str,
    payload: dict,
    *,
    timeout: httpx.Timeout | None = None,
) -> AsyncIterator[str]:
    """POST *payload* to /v1/chat/completions and yield decoded SSE tokens."""
    timeout = timeout or httpx.Timeout(10.0, read=120.0)
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip() == "data: [DONE]":
                    return
                token = parse_sse_line(line)
                if token:
                    yield token
