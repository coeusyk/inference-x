"""Supervisor lifecycle against fake worker processes (add-model-lifecycle-supervisor).

Workers are real child processes (tests/unit/fake_worker.py), so loading,
eviction, draining and crashes are exercised for real; only the model is fake.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

import httpx
import pytest

from inference_x.schemas.model import ModelEntry
from inference_x.services.model_service import ModelRegistry
from inference_x.supervisor.app import app
from inference_x.supervisor.workers import Supervisor

_FAKE = str(Path(__file__).with_name("fake_worker.py"))


def _command(port: int) -> list[str]:
    return [sys.executable, _FAKE, "--port", str(port)]


@pytest.fixture
async def make_client(tmp_path, monkeypatch):
    """Factory: an HTTP client for the supervisor app with fake workers."""
    monkeypatch.setenv("FAKE_WORKER_FAIL", "bad")
    supervisors: list[Supervisor] = []

    async def _make(**kwargs) -> httpx.AsyncClient:
        supervisor = Supervisor(worker_command=_command, logs_dir=tmp_path, startup_timeout_s=20, **kwargs)
        supervisors.append(supervisor)
        app.state.supervisor = supervisor
        app.state.registry = ModelRegistry([
            ModelEntry(name="a", model_path="org/a", aliases=["alias-a"]),
            ModelEntry(name="b", model_path="org/b"),
            ModelEntry(name="bad", model_path="org/bad"),
        ])
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://sup", timeout=60)

    yield _make
    for supervisor in supervisors:
        await supervisor.shutdown()


def _chat(model: str, **extra) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}


async def _state(client: httpx.AsyncClient, model: str) -> dict:
    states = (await client.get("/v1/lifecycle")).json()["models"]
    return next(s for s in states if s["model"] == model)


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


async def test_unknown_model_is_404(make_client):
    client = await make_client()
    resp = await client.post("/v1/chat/completions", json=_chat("nope"))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


async def test_request_loads_worker_and_passes_response_through(make_client):
    client = await make_client()
    assert (await _state(client, "a"))["state"] == "unloaded"
    resp = await client.post("/v1/chat/completions", json=_chat("alias-a"))
    assert resp.status_code == 200
    body = resp.json()
    # Routed to the canonical model's worker; the client's body reaches it unchanged.
    assert body["model"] == "a" and body["asked"] == "alias-a"
    assert resp.headers["x-run-id"] == "sha256:fake"
    state = await _state(client, "a")
    assert state["state"] == "loaded" and state["pid"] == body["pid"] and state["in_flight"] == 0
    again = await client.post("/v1/chat/completions", json=_chat("a"))
    assert again.json()["pid"] == body["pid"]


async def test_concurrent_requests_share_one_load(make_client):
    client = await make_client()
    responses = await asyncio.gather(*(client.post("/v1/chat/completions", json=_chat("a")) for _ in range(4)))
    assert {r.json()["pid"] for r in responses} == {responses[0].json()["pid"]}


async def test_loading_another_model_evicts_the_least_recently_used(make_client):
    client = await make_client(max_loaded=1)
    pid_a = (await client.post("/v1/chat/completions", json=_chat("a"))).json()["pid"]
    resp = await client.post("/v1/chat/completions", json=_chat("b"))
    assert resp.json()["model"] == "b"
    assert _gone(pid_a)
    assert (await _state(client, "a"))["state"] == "unloaded"


async def test_eviction_waits_for_in_flight_requests(make_client):
    client = await make_client(max_loaded=1, switch_wait_s=10)
    await client.post("/v1/chat/completions", json=_chat("a"))
    slow = asyncio.create_task(client.post("/v1/chat/completions", json=_chat("a", sleep=1.0)))
    await asyncio.sleep(0.3)
    switched = await client.post("/v1/chat/completions", json=_chat("b"))
    finished = await slow
    assert finished.status_code == 200 and finished.json()["model"] == "a"
    assert switched.status_code == 200 and switched.json()["model"] == "b"


async def test_busy_workers_make_the_switch_fail_with_retry_after(make_client):
    client = await make_client(max_loaded=1, switch_wait_s=0.3)
    await client.post("/v1/chat/completions", json=_chat("a"))
    slow = asyncio.create_task(client.post("/v1/chat/completions", json=_chat("a", sleep=1.5)))
    await asyncio.sleep(0.2)
    resp = await client.post("/v1/chat/completions", json=_chat("b"))
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "model_busy"
    assert resp.headers["retry-after"]
    assert (await slow).status_code == 200


async def test_load_failure_is_reported_and_not_retried_on_its_own(make_client):
    client = await make_client()
    resp = await client.post("/v1/chat/completions", json=_chat("bad"))
    assert resp.status_code == 503
    error = resp.json()["error"]
    assert error["code"] == "model_load_failed"
    assert "exited with code 3" in error["message"] and "cannot load bad" in error["message"]
    state = await _state(client, "bad")
    assert state["state"] == "failed" and state["exit_code"] == 3


async def test_crashed_worker_is_replaced_on_next_request(make_client):
    client = await make_client()
    pid = (await client.post("/v1/chat/completions", json=_chat("a"))).json()["pid"]
    os.kill(pid, signal.SIGKILL)
    await asyncio.sleep(0.3)
    state = await _state(client, "a")
    assert state["state"] == "failed" and state["exit_code"] == -9
    resp = await client.post("/v1/chat/completions", json=_chat("a"))
    assert resp.status_code == 200 and resp.json()["pid"] != pid


async def test_worker_dying_mid_request_is_503_engine_unavailable(make_client):
    client = await make_client()
    await client.post("/v1/chat/completions", json=_chat("a"))
    resp = await client.post("/v1/chat/completions", json=_chat("a", die=True))
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "engine_unavailable"
    assert (await _state(client, "a"))["state"] == "failed"


async def test_crash_does_not_affect_other_loaded_models(make_client):
    client = await make_client(max_loaded=2)
    pid_a = (await client.post("/v1/chat/completions", json=_chat("a"))).json()["pid"]
    pid_b = (await client.post("/v1/chat/completions", json=_chat("b"))).json()["pid"]
    os.kill(pid_a, signal.SIGKILL)
    resp = await client.post("/v1/chat/completions", json=_chat("b"))
    assert resp.status_code == 200 and resp.json()["pid"] == pid_b


async def test_streaming_is_passed_through_unchanged(make_client):
    client = await make_client()
    async with client.stream("POST", "/v1/chat/completions", json=_chat("a", stream=True)) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        data = (await resp.aread()).decode()
    assert data.count("data: ") == 4 and data.endswith("data: [DONE]\n\n")
    assert (await _state(client, "a"))["in_flight"] == 0


async def test_explicit_load_and_unload(make_client):
    client = await make_client()
    loaded = (await client.post("/v1/lifecycle/load", json={"model": "b"})).json()
    assert loaded["state"] == "loaded"
    unloaded = await client.post("/v1/lifecycle/unload", json={"model": "b"})
    assert unloaded.json()["state"] == "unloaded"
    assert _gone(loaded["pid"])
    assert (await client.post("/v1/lifecycle/unload", json={"model": "nope"})).status_code == 404


async def test_unload_of_a_busy_model_is_409(make_client):
    client = await make_client(switch_wait_s=0.3)
    await client.post("/v1/chat/completions", json=_chat("a"))
    slow = asyncio.create_task(client.post("/v1/chat/completions", json=_chat("a", sleep=1.5)))
    await asyncio.sleep(0.2)
    resp = await client.post("/v1/lifecycle/unload", json={"model": "a"})
    assert resp.status_code == 409 and resp.json()["error"]["code"] == "model_busy"
    assert (await slow).status_code == 200


async def test_shutdown_stops_every_worker(make_client):
    client = await make_client(max_loaded=2)
    pids = [(await client.post("/v1/chat/completions", json=_chat(m))).json()["pid"] for m in ("a", "b")]
    await app.state.supervisor.shutdown()
    assert all(_gone(pid) for pid in pids)
    assert (await client.get("/health")).json()["loaded_models"] == []
