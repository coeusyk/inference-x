"""Tests that tier-resolved engine knobs reach LLM(...) (see add-engine-knob-surfacing)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from inference_x.engines.vllm_engine import VLLMEngine


def _init_engine(scaled_config: dict) -> dict:
    captured: dict = {}

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeLLM:
        def get_tokenizer(self):
            tok = MagicMock()
            tok.chat_template = None
            return tok

    class FakeAsyncLLM:
        @classmethod
        def from_engine_args(cls, engine_args):
            return FakeLLM()

    with patch("inference_x.engines.vllm_engine._VLLM_AVAILABLE", True):
        with patch("inference_x.engines.vllm_engine._load_vllm"):
            with patch(
                "inference_x.engines.vllm_engine.scale_model_config_for_pool",
                return_value=scaled_config,
            ):
                with patch("inference_x.engines.vllm_engine.preflight_hf_access"):
                    with patch("inference_x.engines.vllm_engine.ensure_vllm_runtime_env"):
                        with patch("inference_x.engines.vllm_engine._check_vram_budget"):
                            with patch.dict(
                                "inference_x.engines.vllm_engine.__dict__",
                                {
                                    "AsyncEngineArgs": FakeAsyncEngineArgs,
                                    "AsyncLLM": FakeAsyncLLM,
                                },
                            ):
                                VLLMEngine(
                                    {
                                        "name": scaled_config["name"],
                                        "model_path": scaled_config["model_path"],
                                    }
                                )
    return captured


def test_tier_knobs_passed_to_vllm_when_present():
    captured = _init_engine(
        {
            "name": "qwen2.5-0.5b",
            "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "gpu_memory_utilization": 0.82,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 2048,
            "block_size": 16,
            "kv_cache_dtype": "auto",
            "enable_prefix_caching": False,
        }
    )
    assert captured["max_num_seqs"] == 4
    assert captured["max_num_batched_tokens"] == 2048
    assert captured["block_size"] == 16
    assert captured["kv_cache_dtype"] == "auto"
    assert captured["enable_prefix_caching"] is False


def test_tier_knobs_omitted_when_absent():
    """No tier resolved (apply_tier_knobs never ran) -> config has none of these keys,
    and VLLMEngine must not invent defaults that weren't there before this change."""
    captured = _init_engine(
        {
            "name": "qwen2.5-0.5b",
            "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "gpu_memory_utilization": 0.82,
        }
    )
    for key in (
        "max_num_seqs",
        "max_num_batched_tokens",
        "block_size",
        "kv_cache_dtype",
        "enable_prefix_caching",
    ):
        assert key not in captured
