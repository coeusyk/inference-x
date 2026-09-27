"""LlamaCppEngine against a fake llama-server process (add-llama-cpp-backend).

The fake (tests/unit/fake_llama_server.py) is a real subprocess speaking
llama-server's HTTP API, so process startup, failure and shutdown are exercised
for real; only the model is missing.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from inference_x.engines.base import EngineUnavailableError
from inference_x.engines.llama_cpp_engine import LlamaCppEngine
from inference_x.routing.admission import ContextTooLongError
from inference_x.schemas.chat import ChatCompletionRequest, ChatMessage

_FAKE = Path(__file__).with_name("fake_llama_server.py")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Point the engine at the fake server; returns (gguf, record file)."""
    binary = tmp_path / "llama-server"
    binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{_FAKE}" "$@"\n')
    binary.chmod(0o755)
    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"GGUF fake weights")
    record = tmp_path / "requests.jsonl"
    monkeypatch.setenv("INFERENCE_X_LLAMA_SERVER", str(binary))
    monkeypatch.setenv("FAKE_LLAMA_RECORD", str(record))
    monkeypatch.setenv("INFERENCE_X_LLAMA_STARTUP_TIMEOUT_S", "20")
    monkeypatch.chdir(tmp_path)  # the server log goes to ./logs
    return gguf, record


def _config(gguf: Path, **overrides) -> dict:
    return {"name": "tiny-gguf", "model_path": str(gguf), "max_model_len": 4096, **overrides}


def _request(**overrides) -> ChatCompletionRequest:
    fields = {"model": "tiny-gguf", "messages": [ChatMessage(role="user", content="hi")], "max_tokens": 16}
    return ChatCompletionRequest(**{**fields, **overrides})


def _bodies(record: Path) -> list[dict]:
    return [json.loads(line) for line in record.read_text().splitlines()]


@pytest.fixture
def engine(fake):
    gguf, _ = fake
    eng = LlamaCppEngine(_config(gguf))
    yield eng
    eng.shutdown()


def test_startup_reads_server_identity(engine, fake):
    gguf, record = fake
    assert engine.is_healthy()
    assert engine.kv_capacity_tokens == 4096
    assert engine.chat_template_sha256 == hashlib.sha256(b"{{ messages }}").hexdigest()
    identity = engine.manifest_identity
    assert identity["backend"] == "llama.cpp"
    assert identity["backend_version"] == "b1-fake"
    assert identity["quantization"] == "Q4_K - Medium"
    assert identity["weights_sha256"] == hashlib.sha256(gguf.read_bytes()).hexdigest()
    assert identity["gguf_file"] == "tiny.gguf"
    assert identity["hf_repo"] is None and identity["hf_revision"] is None
    assert identity["hardware_cuda"] is None
    assert engine.runtime_snapshot == {
        "max_model_len": 4096, "kv_cache_dtype": "f16", "prefix_caching": False,
        "batch_invariant": False,
    }
    args = json.loads(Path(str(record) + ".args").read_text())
    assert args[args.index("-c") + 1] == "4096"
    assert args[args.index("-np") + 1] == "1"
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert "--no-cache-prompt" in args
    assert "-ngl" not in args


def test_n_gpu_layers_is_passed_only_when_configured(fake):
    gguf, record = fake
    eng = LlamaCppEngine(_config(gguf, n_gpu_layers=20))
    try:
        args = json.loads(Path(str(record) + ".args").read_text())
        assert args[args.index("-ngl") + 1] == "20"
    finally:
        eng.shutdown()


def test_count_prompt_tokens_asks_the_server(engine):
    assert engine.count_prompt_tokens(_request()) == 7


async def test_generate_translates_request_and_response(engine, fake):
    _, record = fake
    response = await engine.generate(_request(temperature=0.0, seed=5, stop="END"))
    assert response.choices[0].message.content == "Hello there"
    assert response.choices[0].finish_reason == "length"
    assert response.usage.total_tokens == 9
    assert response.timing is None  # no queue time from llama-server: never invented
    body = _bodies(record)[-1]
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["max_tokens"] == 16
    assert body["temperature"] == 0.0
    assert body["seed"] == 5
    assert body["stop"] == ["END"]
    # Samplers InferenceX does not expose are sent disabled.
    assert body["top_k"] == 0 and body["min_p"] == 0.0
    assert "repeat_penalty" not in body
    assert body["stream"] is False


async def test_base_model_gets_repetition_penalty(fake):
    gguf, record = fake
    eng = LlamaCppEngine(_config(gguf, instruction_tuned=False))
    try:
        await eng.generate(_request())
        assert _bodies(record)[-1]["repeat_penalty"] == 1.15
    finally:
        eng.shutdown()


async def test_generate_stream_yields_content_then_terminal_usage(engine, fake):
    _, record = fake
    chunks = [c async for c in engine.generate_stream(_request())]
    assert [c.content for c in chunks[:-1]] == ["Hello", " there"]
    terminal = chunks[-1]
    assert terminal.finish_reason == "stop"
    assert terminal.usage is not None and terminal.usage.prompt_tokens == 7
    assert terminal.timing is None
    assert _bodies(record)[-1]["stream_options"] == {"include_usage": True}


async def test_context_overflow_maps_to_context_too_long(fake, monkeypatch):
    gguf, _ = fake
    monkeypatch.setenv("FAKE_LLAMA_MODE", "ctx")
    eng = LlamaCppEngine(_config(gguf))
    try:
        with pytest.raises(ContextTooLongError):
            await eng.generate(_request())
        with pytest.raises(ContextTooLongError):
            _ = [c async for c in eng.generate_stream(_request())]
    finally:
        eng.shutdown()


async def test_dead_server_is_unavailable(engine):
    engine._process.kill()  # type: ignore[union-attr]
    engine._process.wait()  # type: ignore[union-attr]
    assert not engine.is_healthy()
    with pytest.raises(EngineUnavailableError):
        engine.count_prompt_tokens(_request())
    with pytest.raises(EngineUnavailableError):
        await engine.generate(_request())


def test_exit_during_startup_reports_log_tail(fake, monkeypatch):
    gguf, _ = fake
    monkeypatch.setenv("FAKE_LLAMA_MODE", "exit")
    with pytest.raises(RuntimeError, match=r"exited with code 3(.|\n)*failed to load model"):
        LlamaCppEngine(_config(gguf))


def test_never_healthy_times_out_and_stops_the_process(fake, monkeypatch):
    gguf, record = fake
    monkeypatch.setenv("FAKE_LLAMA_MODE", "hang")
    monkeypatch.setenv("INFERENCE_X_LLAMA_STARTUP_TIMEOUT_S", "1")
    with pytest.raises(RuntimeError, match="did not become healthy"):
        LlamaCppEngine(_config(gguf))


def test_missing_binary(fake, monkeypatch, tmp_path):
    gguf, _ = fake
    monkeypatch.setenv("INFERENCE_X_LLAMA_SERVER", str(tmp_path / "nope"))
    with pytest.raises(RuntimeError, match="llama-server not found"):
        LlamaCppEngine(_config(gguf))


def test_missing_gguf(fake, tmp_path):
    with pytest.raises(RuntimeError, match="GGUF file not found"):
        LlamaCppEngine(_config(tmp_path / "missing.gguf"))


def test_shutdown_stops_the_process(fake):
    gguf, _ = fake
    eng = LlamaCppEngine(_config(gguf))
    process = eng._process
    eng.shutdown()
    assert process is not None and process.poll() is not None
    assert not eng.is_healthy()
