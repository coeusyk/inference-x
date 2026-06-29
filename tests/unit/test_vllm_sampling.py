"""Tests for VLLMEngine SamplingParams derived from model config."""

from __future__ import annotations

import sys

import pytest

from inference_x.engines.vllm_engine import VLLMEngine
from inference_x.schemas.chat import ChatCompletionRequest, ChatMessage


class _CaptureSamplingParams:
    """Record kwargs passed to SamplingParams without loading vLLM."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        _CaptureSamplingParams.last_kwargs = kwargs


@pytest.fixture
def capture_sampling_params(monkeypatch):
    _CaptureSamplingParams.last_kwargs = None
    mod = sys.modules["inference_x.engines.vllm_engine"]
    monkeypatch.setitem(mod.__dict__, "SamplingParams", _CaptureSamplingParams)
    yield _CaptureSamplingParams


def _engine_with_config(**config_overrides) -> VLLMEngine:
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._max_completion_tokens = config_overrides.get("max_completion_tokens")
    engine._instruction_tuned = config_overrides.get("instruction_tuned", True)
    engine._repetition_penalty = config_overrides.get("repetition_penalty")
    return engine


def test_max_completion_tokens_applied(capture_sampling_params):
    """max_completion_tokens in model config overrides request max_tokens."""
    engine = _engine_with_config(max_completion_tokens=256, instruction_tuned=False)
    request = ChatCompletionRequest(
        model="opt-125m",
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=512,
    )

    engine._sampling_params(request)

    assert capture_sampling_params.last_kwargs is not None
    assert capture_sampling_params.last_kwargs["max_tokens"] == 256


def test_instruction_tuned_model_has_no_repetition_penalty(capture_sampling_params):
    engine = _engine_with_config(instruction_tuned=True)
    request = ChatCompletionRequest(
        model="qwen2.5-0.5b",
        messages=[ChatMessage(role="user", content="hi")],
    )

    engine._sampling_params(request)

    assert "repetition_penalty" not in capture_sampling_params.last_kwargs


def test_base_model_gets_repetition_penalty(capture_sampling_params):
    engine = _engine_with_config(
        instruction_tuned=False,
        repetition_penalty=1.15,
    )
    request = ChatCompletionRequest(
        model="opt-125m",
        messages=[ChatMessage(role="user", content="hi")],
    )

    engine._sampling_params(request)

    assert capture_sampling_params.last_kwargs["repetition_penalty"] == 1.15


def test_base_model_default_repetition_penalty(capture_sampling_params):
    engine = _engine_with_config(instruction_tuned=False, repetition_penalty=None)
    request = ChatCompletionRequest(
        model="opt-125m",
        messages=[ChatMessage(role="user", content="hi")],
    )

    engine._sampling_params(request)

    assert capture_sampling_params.last_kwargs["repetition_penalty"] == 1.15
