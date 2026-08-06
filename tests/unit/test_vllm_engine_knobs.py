"""Tests that tier-resolved engine knobs reach LLM(...) (see add-engine-knob-surfacing)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from inference_x.engines import vllm_engine as vllm_engine_module
from inference_x.engines.vllm_engine import VLLMEngine


def _init_engine(scaled_config: dict, *, pool_size: int = 1) -> dict:
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
                                    },
                                    pool_size=pool_size,
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


# ---------------------------------------------------------------------------
# pool_size > 1 metric-label collision warning (expose-native-engine-metrics)
# ---------------------------------------------------------------------------

_BASE_CONFIG = {
    "name": "qwen2.5-0.5b",
    "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
    "gpu_memory_utilization": 0.82,
}


def _capture_warnings(monkeypatch) -> list[str]:
    # Assert on the logger call directly rather than caplog: config/logging.yaml sets
    # propagate: false on the "inference_x" logger once any test imports
    # inference_x.api.main (which applies the dictConfig), so records never reach
    # caplog's root-attached handler even though the console handler does fire
    # (same workaround as tests/unit/test_vram_tiers.py).
    warnings: list[str] = []
    monkeypatch.setattr(
        vllm_engine_module.logger,
        "warning",
        lambda msg, *args, **kw: warnings.append(msg % args if args else msg),
    )
    return warnings


def test_pool_size_one_emits_no_collision_warning(monkeypatch):
    warnings = _capture_warnings(monkeypatch)
    _init_engine(dict(_BASE_CONFIG), pool_size=1)
    assert not any("process-wide" in w for w in warnings)


def test_pool_size_greater_than_one_emits_exactly_one_collision_warning(monkeypatch):
    warnings = _capture_warnings(monkeypatch)
    _init_engine(dict(_BASE_CONFIG), pool_size=2)
    collision_warnings = [w for w in warnings if "process-wide" in w]
    assert len(collision_warnings) == 1
    message = collision_warnings[0]
    assert "process-wide" in message
    assert "collision" in message
    assert "not yet verified" in message
