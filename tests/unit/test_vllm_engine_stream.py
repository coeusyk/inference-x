"""Tests for VLLMEngine's tokenizer-facing helpers and KV-cache introspection.

Streaming/completion behavior against the engine's generation primitive moved
to test_vllm_engine_async.py (migrate-async-llm-engine, Task 4) — this file
now covers only what doesn't depend on that primitive.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from inference_x.engines.vllm_engine import VLLMEngine
from inference_x.schemas.chat import ChatCompletionRequest, ChatMessage


def test_log_kv_cache_stats_reads_cache_config_from_vllm_config():
    """Regression guard: AsyncLLM exposes `vllm_config` directly as an
    instance attribute (no `llm_engine` indirection, unlike the offline `LLM`
    class this replaced) — verified against vLLM 0.22.1
    (migrate-async-llm-engine Decision 6). Reading the wrong attribute
    silently leaves kv_capacity_tokens as None forever (no exception).
    """
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._model_name = "test-model"
    engine._kv_capacity_tokens = None
    engine._llm = MagicMock()
    engine._llm.vllm_config.cache_config.num_gpu_blocks = 14822
    engine._llm.vllm_config.cache_config.block_size = 16

    engine._log_kv_cache_stats()

    assert engine.kv_capacity_tokens == 14822 * 16


def test_log_kv_cache_stats_defaults_to_none_when_cache_config_missing():
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._model_name = "test-model"
    engine._kv_capacity_tokens = None
    engine._llm = MagicMock()
    engine._llm.vllm_config = None

    engine._log_kv_cache_stats()

    assert engine.kv_capacity_tokens is None


# ---------------------------------------------------------------------------
# count_prompt_tokens (admission control)
# ---------------------------------------------------------------------------


def test_count_prompt_tokens_uses_tokenizer_encode():
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._model_name = "test-model"
    engine._stream_prompt = lambda request: "User: hi\nAssistant:"
    engine._llm = MagicMock()
    engine._llm.get_tokenizer.return_value.encode.return_value = [1, 2, 3, 4, 5]

    req = ChatCompletionRequest(
        model="test-model", messages=[ChatMessage(role="user", content="hi")]
    )
    assert engine.count_prompt_tokens(req) == 5


def test_count_prompt_tokens_falls_back_to_chars_over_4_on_error():
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._model_name = "test-model"
    engine._stream_prompt = MagicMock(side_effect=RuntimeError("tokenizer unavailable"))
    engine._llm = MagicMock()

    req = ChatCompletionRequest(
        model="test-model", messages=[ChatMessage(role="user", content="12345678")]
    )
    assert engine.count_prompt_tokens(req) == 2  # 8 chars // 4
