"""Tests for VLLMEngine streaming via the startup-loaded sync llm_engine."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from inference_x.engines.vllm_engine import VLLMEngine
from inference_x.schemas.chat import ChatCompletionRequest, ChatMessage


@dataclass
class _FakeCompletion:
    text: str = ""
    finish_reason: str | None = None
    token_ids: list[int] = field(default_factory=list)


@dataclass
class _FakeRequestOutput:
    request_id: str
    outputs: list[_FakeCompletion]
    finished: bool


def _make_engine(step_batches: list[list[_FakeRequestOutput]]) -> VLLMEngine:
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._model_name = "test-model"
    engine._pool_size = 1
    engine._healthy = True
    engine._supports_chat = False
    engine._engine_lock = threading.Lock()
    engine._sampling_params = lambda request: MagicMock()
    engine._stream_prompt = lambda request: "User: hi\nAssistant:"
    engine._llm = MagicMock()
    llm_engine = MagicMock()
    llm_engine.has_unfinished_requests.side_effect = [True] * len(step_batches) + [False]
    llm_engine.step.side_effect = step_batches
    engine._llm.llm_engine = llm_engine

    def add_request(rid, prompt, sampling):
        for batch in step_batches:
            for item in batch:
                item.request_id = rid

    llm_engine.add_request.side_effect = add_request
    return engine


@pytest.mark.asyncio
async def test_generate_stream_yields_incremental_chunks_from_llm_engine():
    engine = _make_engine(
        [
            [
                _FakeRequestOutput(
                    request_id="",
                    outputs=[_FakeCompletion(text="Hello")],
                    finished=False,
                )
            ],
            [
                _FakeRequestOutput(
                    request_id="",
                    outputs=[_FakeCompletion(text="Hello world")],
                    finished=True,
                )
            ],
        ]
    )

    req = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
    )

    with patch("inference_x.engines.vllm_engine._VLLM_AVAILABLE", True):
        chunks = [chunk async for chunk in engine.generate_stream(req)]

    assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_generate_stream_does_not_use_async_llm_engine():
    engine = _make_engine(
        [
            [
                _FakeRequestOutput(
                    request_id="",
                    outputs=[_FakeCompletion(text="ok")],
                    finished=True,
                )
            ]
        ]
    )

    req = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
    )

    with patch("inference_x.engines.vllm_engine._VLLM_AVAILABLE", True):
        chunks = [chunk async for chunk in engine.generate_stream(req)]

    assert chunks == ["ok"]
    assert getattr(engine, "_async_llm", None) is None


def test_log_kv_cache_stats_reads_cache_config_from_vllm_config():
    """Regression guard: vLLM 0.22.1's V1 LLMEngine has cache_config=None at the top
    level — the real CacheConfig lives under llm_engine.vllm_config.cache_config.
    Reading the wrong attribute silently leaves kv_capacity_tokens as None forever
    (no exception), which is exactly what happened before this was caught via a live
    GPU smoke test. This test pins the correct attribute path with a mock engine.
    """
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._model_name = "test-model"
    engine._kv_capacity_tokens = None
    engine._llm = MagicMock()

    llm_engine = MagicMock()
    llm_engine.cache_config = None  # top-level attribute is always None on V1
    llm_engine.vllm_config.cache_config.num_gpu_blocks = 14822
    llm_engine.vllm_config.cache_config.block_size = 16
    engine._llm.llm_engine = llm_engine

    engine._log_kv_cache_stats()

    assert engine.kv_capacity_tokens == 14822 * 16


def test_log_kv_cache_stats_defaults_to_none_when_cache_config_missing():
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._model_name = "test-model"
    engine._kv_capacity_tokens = None
    engine._llm = MagicMock()

    llm_engine = MagicMock()
    llm_engine.cache_config = None
    llm_engine.vllm_config = None
    engine._llm.llm_engine = llm_engine

    engine._log_kv_cache_stats()

    assert engine.kv_capacity_tokens is None
