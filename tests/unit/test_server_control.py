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

    async def fake_start(_model, **_kwargs):
        calls["restart"] += 1

    monkeypatch.setattr(sc, "fetch_loaded_models", fake_fetch)
    import streaming

    monkeypatch.setattr(streaming, "wait_for_model", fake_wait)
    monkeypatch.setattr(sc, "stop_playground_server", fake_stop)
    monkeypatch.setattr(sc, "start_playground_server", fake_start)

    result = await sc.ensure_models_loaded("http://test", ["tinyllama-chat"])
    assert result == {"tinyllama-chat": "http://test"}
    assert calls["restart"] == 0
    assert calls["wait"] == 1
    assert sc.playground_started_server() is False


@pytest.mark.asyncio
async def test_ensure_models_loaded_restarts_when_model_missing(monkeypatch):
    calls = {"start": [], "stop": 0}

    async def fake_fetch(_base_url: str):
        return ["qwen2.5-0.5b"]

    async def fake_wait(*_args, **_kwargs):
        return True

    async def fake_stop():
        calls["stop"] += 1

    async def fake_start(model, **kwargs):
        calls["start"].append((model, kwargs.get("port")))

    async def fake_health(*_a, **_k):
        return True

    monkeypatch.setattr(sc, "fetch_loaded_models", fake_fetch)
    import streaming

    monkeypatch.setattr(streaming, "wait_for_model", fake_wait)
    monkeypatch.setattr(sc, "stop_playground_server", fake_stop)
    monkeypatch.setattr(sc, "start_playground_server", fake_start)
    monkeypatch.setattr(sc, "wait_for_health", fake_health)

    result = await sc.ensure_models_loaded("http://test", ["tinyllama-chat"])
    assert result == {"tinyllama-chat": "http://test"}
    assert calls["start"] == [("tinyllama-chat", 8000)]
    assert calls["stop"] == 1
    assert sc.playground_started_server() is True


@pytest.mark.asyncio
async def test_ensure_models_loaded_starts_one_process_per_model(monkeypatch):
    """B6: compare mode launches N independent single-model processes on N ports."""
    calls = {"start": []}

    async def fake_fetch(_base_url: str):
        return []

    async def fake_wait(*_args, **_kwargs):
        return True

    async def fake_stop():
        return None

    async def fake_start(model, **kwargs):
        calls["start"].append((model, kwargs.get("port")))

    async def fake_health(*_a, **_k):
        return True

    monkeypatch.setattr(sc, "fetch_loaded_models", fake_fetch)
    import streaming

    monkeypatch.setattr(streaming, "wait_for_model", fake_wait)
    monkeypatch.setattr(sc, "stop_playground_server", fake_stop)
    monkeypatch.setattr(sc, "start_playground_server", fake_start)
    monkeypatch.setattr(sc, "wait_for_health", fake_health)

    result = await sc.ensure_models_loaded(
        "http://localhost:8000", ["qwen2.5-0.5b", "tinyllama-chat"]
    )
    assert result == {
        "qwen2.5-0.5b": "http://localhost:8000",
        "tinyllama-chat": "http://localhost:8001",
    }
    assert calls["start"] == [
        ("qwen2.5-0.5b", 8000),
        ("tinyllama-chat", 8001),
    ]


@pytest.fixture(autouse=True)
def _reset_playground_server_flag():
    sc._reset_playground_server_state()
    yield
    sc._reset_playground_server_state()


@pytest.mark.asyncio
async def test_cleanup_stops_only_when_playground_started(monkeypatch):
    stop_calls: list[str] = []

    async def fake_stop():
        stop_calls.append("stop")

    monkeypatch.setattr(sc, "stop_playground_server", fake_stop)

    await sc.cleanup_playground_server_if_started()
    assert stop_calls == []

    sc._server_started_by_playground = True
    await sc.cleanup_playground_server_if_started()
    assert stop_calls == ["stop"]
    assert sc.playground_started_server() is False
