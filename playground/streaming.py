"""Shared SSE streaming helpers for playground TUIs."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import httpx

from app import fetch_health, fetch_models, parse_sse_line, parse_sse_stream

__all__ = [
    "fetch_health",
    "fetch_models",
    "parse_sse_line",
    "parse_sse_stream",
    "stream_chat_tokens",
    "wait_for_model",
]


async def wait_for_model(
    base_url: str,
    model: str,
    *,
    timeout_s: int = 60,
    poll_interval_s: float = 2.0,
) -> bool:
    """Poll until *model* appears in ``GET /health`` ``loaded_models``."""
    deadline = time.monotonic() + timeout_s
    health_url = f"{base_url.rstrip('/')}/health"
    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(health_url)
                if resp.status_code == 200:
                    loaded = resp.json().get("loaded_models") or []
                    if model in loaded:
                        return True
        except httpx.HTTPError:
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(poll_interval_s, remaining))
    return False


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
