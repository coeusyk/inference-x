"""Tests for VLLMEngine's AsyncLLM-backed generate/generate_stream.

Replaces test_engine_driver.py (migrate-async-llm-engine, Task 4.2): same
properties, reproduced against the AsyncLLM shape (`self._llm.generate()` as
an async generator consumed by `_stream_chunks`) rather than EngineDriver's
thread+queue demultiplexing.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

import inference_x.engines.vllm_engine as vllm_engine_module
from inference_x.engines.vllm_engine import VLLMEngine
from inference_x.schemas.chat import ChatCompletionRequest, ChatMessage


class _EngineDeadError(RuntimeError):
    """Stand-in for AsyncLLM's real EngineDeadError (verified Task 1.3/1.4)."""


class _FakeCompletion:
    def __init__(self, text="", finish_reason=None, token_ids=None):
        self.text = text
        self.finish_reason = finish_reason
        self.token_ids = token_ids or []


class _FakeRequestOutput:
    def __init__(self, outputs, finished, prompt_token_ids=(1, 2, 3)):
        self.outputs = outputs
        self.finished = finished
        self.prompt_token_ids = list(prompt_token_ids)


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeAsyncLLM:
    """Fake `AsyncLLM`: one scripted output sequence per prompt string.

    `generate()` is an async generator, matching `AsyncLLM.generate()`'s real
    shape. `errored` is checked before yielding anything, matching the
    verified Task 1.4 fact that a dead engine rejects immediately rather than
    orphaning the request. A `GeneratorExit` thrown into an in-flight
    `generate()` call is recorded before re-raising, standing in for
    `AsyncLLM`'s own engine-side abort (verified Task 1.2) without requiring
    the real engine.
    """

    def __init__(self, scripts: dict[str, list[_FakeRequestOutput]] | None = None):
        self.errored = False
        self.aborted_request_ids: list[str] = []
        self._scripts = scripts or {}
        self.raise_after: Exception | None = None

    def get_tokenizer(self):
        tok = MagicMock()
        tok.chat_template = None
        return tok

    async def generate(self, prompt, sampling_params, request_id):
        if self.errored:
            raise _EngineDeadError("engine is dead")
        try:
            for output in self._scripts[prompt]:
                await asyncio.sleep(0)
                yield output
            if self.raise_after is not None:
                raise self.raise_after
        except GeneratorExit:
            self.aborted_request_ids.append(request_id)
            raise

    def shutdown(self):
        self.errored = True


@pytest.fixture(autouse=True)
def _patch_vllm_available(monkeypatch):
    monkeypatch.setattr(vllm_engine_module, "_VLLM_AVAILABLE", True)
    monkeypatch.setitem(vllm_engine_module.__dict__, "SamplingParams", _FakeSamplingParams)


def _make_engine(llm: _FakeAsyncLLM) -> VLLMEngine:
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._model_name = "test-model"
    engine._max_completion_tokens = None
    engine._instruction_tuned = True
    engine._repetition_penalty = None
    engine._supports_chat = False
    engine._healthy = True
    engine._llm = llm
    return engine


def _req(marker: str = "hi") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model", messages=[ChatMessage(role="user", content=marker)]
    )


def _prompt(marker: str) -> str:
    return f"User: {marker}\nAssistant:"


async def test_two_concurrent_completions_return_distinct_correct_output():
    llm = _FakeAsyncLLM(
        {
            _prompt("A"): [_FakeRequestOutput([_FakeCompletion("A-final", "stop", [1, 2])], True)],
            _prompt("B"): [
                _FakeRequestOutput([_FakeCompletion("B-final", "stop", [1, 2, 3])], True)
            ],
        }
    )
    engine = _make_engine(llm)

    resp_a, resp_b = await asyncio.gather(engine.generate(_req("A")), engine.generate(_req("B")))

    assert resp_a.choices[0].message.content == "A-final"
    assert resp_b.choices[0].message.content == "B-final"


async def test_stream_yields_incremental_deltas():
    llm = _FakeAsyncLLM(
        {
            _prompt("hi"): [
                _FakeRequestOutput([_FakeCompletion("Hello", None, [])], False),
                _FakeRequestOutput([_FakeCompletion("Hello world", "stop", [1, 2, 3])], True),
            ],
        }
    )
    engine = _make_engine(llm)

    chunks = [c async for c in engine.generate_stream(_req())]

    assert [c.content for c in chunks if c.content] == ["Hello", " world"]
    terminal = chunks[-1]
    assert terminal.finish_reason == "stop"
    assert terminal.usage is not None


async def test_two_concurrent_streams_do_not_cross_contaminate():
    llm = _FakeAsyncLLM(
        {
            _prompt("A"): [
                _FakeRequestOutput([_FakeCompletion("Hi", None, [])], False),
                _FakeRequestOutput([_FakeCompletion("Hi there", "stop", [1, 2])], True),
            ],
            _prompt("B"): [
                _FakeRequestOutput([_FakeCompletion("Yo", None, [])], False),
                _FakeRequestOutput([_FakeCompletion("Yo dude", "stop", [1, 2])], True),
            ],
        }
    )
    engine = _make_engine(llm)

    async def collect(marker: str) -> str:
        text = ""
        async for chunk in engine.generate_stream(_req(marker)):
            if chunk.content:
                text += chunk.content
        return text

    text_a, text_b = await asyncio.gather(collect("A"), collect("B"))

    assert text_a == "Hi there"
    assert text_b == "Yo dude"


async def test_engine_exception_mid_stream_is_observable():
    llm = _FakeAsyncLLM({_prompt("hi"): [_FakeRequestOutput([_FakeCompletion("partial")], False)]})
    llm.raise_after = RuntimeError("simulated failure")
    engine = _make_engine(llm)

    with pytest.raises(RuntimeError, match="simulated failure"):
        [c async for c in engine.generate_stream(_req())]


async def test_engine_exception_mid_generation_is_observable_to_pending_completion():
    llm = _FakeAsyncLLM({_prompt("hi"): [_FakeRequestOutput([_FakeCompletion("partial")], False)]})
    llm.raise_after = RuntimeError("simulated failure")
    engine = _make_engine(llm)

    with pytest.raises(RuntimeError, match="simulated failure"):
        await engine.generate(_req())


def test_shutdown_stops_cleanly():
    llm = _FakeAsyncLLM()
    engine = _make_engine(llm)

    engine.shutdown()

    assert engine.is_healthy() is False


async def test_mid_stream_disconnect_triggers_engine_side_abort():
    """Decision 4's compatibility invariant: a disconnected consumer results
    in an engine-side abort, not merely ceasing to read from a queue.

    `generate_stream()` wraps `_stream_chunks()` via a plain `async for`, so
    closing the outer generator does not synchronously cascade `GeneratorExit`
    into the inner one (`async for` has no implicit `aclose()` on early exit).
    The abandoned `_stream_chunks()` generator becomes unreferenced once
    `generate_stream()`'s frame is torn down, and CPython's async-generator
    GC finalizer hook schedules its `aclose()` as a task — reaching
    `AsyncLLM.generate()` within a few event-loop iterations, not
    immediately. Same eventually-consistent GC-driven pattern already
    documented for `ChatService.stream_response` (Phase A audit, SEV-A1) —
    verified here empirically rather than assumed.
    """
    llm = _FakeAsyncLLM(
        {
            _prompt("hi"): [
                _FakeRequestOutput([_FakeCompletion("Hello", None, [])], False),
                _FakeRequestOutput([_FakeCompletion("Hello world", "stop", [1, 2, 3])], True),
            ],
        }
    )
    engine = _make_engine(llm)

    stream = engine.generate_stream(_req())
    await stream.__anext__()
    await stream.aclose()
    for _ in range(10):
        if llm.aborted_request_ids:
            break
        await asyncio.sleep(0)

    assert llm.aborted_request_ids, "engine-side abort was not observed"


async def test_submission_after_engine_death_raises_immediately():
    """Task 1.4 verified fact, now asserted unconditionally."""
    llm = _FakeAsyncLLM()
    llm.errored = True
    engine = _make_engine(llm)

    with pytest.raises(RuntimeError, match="engine is dead"):
        await engine.generate(_req())

    with pytest.raises(RuntimeError, match="engine is dead"):
        [c async for c in engine.generate_stream(_req())]


async def test_streamed_and_non_streamed_usage_agree():
    """OS-2 acceptance criterion, re-verified against the AsyncLLM-backed
    engine: both paths derive from `derive_terminal_metadata`, so agreement
    is structural, not coincidental."""
    final = _FakeRequestOutput(
        [_FakeCompletion("Hello world", "stop", [10, 11, 12, 13, 14])], True
    )
    stream_engine = _make_engine(_FakeAsyncLLM({_prompt("hi"): [final]}))
    complete_engine = _make_engine(_FakeAsyncLLM({_prompt("hi"): [final]}))

    chunks = [c async for c in stream_engine.generate_stream(_req())]
    resp = await complete_engine.generate(_req())

    streamed_usage = chunks[-1].usage
    assert streamed_usage is not None
    assert streamed_usage.completion_tokens == resp.usage.completion_tokens == 5
    assert streamed_usage.prompt_tokens == resp.usage.prompt_tokens == 3
    assert streamed_usage.total_tokens == resp.usage.total_tokens == 8
    assert chunks[-1].finish_reason == resp.choices[0].finish_reason == "stop"
