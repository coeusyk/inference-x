"""Tests for max_model_len kwarg handling in VLLMEngine."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from inference_x.engines.vllm_engine import VLLMEngine


def _init_engine(scaled_config: dict) -> dict:
    captured: dict = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_engine = MagicMock()  # EngineDriver needs a real-ish llm_engine

        def get_tokenizer(self):
            tok = MagicMock()
            tok.chat_template = None
            return tok

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
                                {"LLM": FakeLLM},
                            ):
                                VLLMEngine(
                                    {
                                        "name": scaled_config["name"],
                                        "model_path": scaled_config["model_path"],
                                    }
                                )
    return captured


class TestMaxModelLen:
    def test_passed_to_vllm_when_set(self):
        captured = _init_engine(
            {
                "name": "qwen2.5-0.5b",
                "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "gpu_memory_utilization": 0.82,
                "max_model_len": 8192,
            }
        )
        assert captured["max_model_len"] == 8192

    def test_omitted_when_none(self):
        captured = _init_engine(
            {
                "name": "qwen2.5-0.5b",
                "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "gpu_memory_utilization": 0.82,
            }
        )
        assert "max_model_len" not in captured
