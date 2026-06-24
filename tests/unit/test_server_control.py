"""Unit tests for playground/server_control.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "playground"))

import server_control as sc  # noqa: E402


@pytest.mark.asyncio
async def test_fetch_loaded_models_returns_list(monkeypatch):
    class _Resp:
        status_code = 200

        def json(self):
            return {"loaded_models": ["qwen2.5-0.5b"]}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url: str):
            return _Resp()

    monkeypatch.setattr(sc.httpx, "AsyncClient", lambda **kwargs: _Client())
    assert await sc.fetch_loaded_models("http://test") == ["qwen2.5-0.5b"]


@pytest.mark.asyncio
async def test_ensure_models_loaded_skips_restart_when_already_loaded(monkeypatch):
    calls = {"restart": 0, "wait": 0}

    async def fake_fetch(_base_url: str):
        return ["tinyllama-chat"]

    async def fake_wait(*_args, **_kwargs):
        calls["wait"] += 1
        return True

    async def fake_stop():
        calls["restart"] += 1

    async def fake_start(_models):
        calls["restart"] += 1

    monkeypatch.setattr(sc, "fetch_loaded_models", fake_fetch)
    import streaming

    monkeypatch.setattr(streaming, "wait_for_model", fake_wait)
    monkeypatch.setattr(sc, "stop_playground_server", fake_stop)
    monkeypatch.setattr(sc, "start_playground_server", fake_start)

    ok = await sc.ensure_models_loaded("http://test", ["tinyllama-chat"])
    assert ok is True
    assert calls["restart"] == 0
    assert calls["wait"] == 1


@pytest.mark.asyncio
async def test_ensure_models_loaded_restarts_when_model_missing(monkeypatch):
    calls = {"start": []}

    async def fake_fetch(_base_url: str):
        return ["qwen2.5-0.5b"]

    async def fake_wait(*_args, **_kwargs):
        return True

    async def fake_stop():
        return None

    async def fake_start(models):
        calls["start"].append(list(models))

    async def fake_health(*_a, **_k):
        return True

    monkeypatch.setattr(sc, "fetch_loaded_models", fake_fetch)
    import streaming

    monkeypatch.setattr(streaming, "wait_for_model", fake_wait)
    monkeypatch.setattr(sc, "stop_playground_server", fake_stop)
    monkeypatch.setattr(sc, "start_playground_server", fake_start)
    monkeypatch.setattr(sc, "wait_for_health", fake_health)

    ok = await sc.ensure_models_loaded("http://test", ["tinyllama-chat"])
    assert ok is True
    assert calls["start"] == [["tinyllama-chat"]]
