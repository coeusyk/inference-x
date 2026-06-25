"""Tests for multi-model GPU memory scaling."""

import pytest

from inference_x.utils import vllm_pool_config as pool
from inference_x.utils.vllm_pool_config import (
    scale_model_config_for_pool,
    validate_pool_fits,
)


@pytest.fixture
def fixed_footprints(monkeypatch):
    """Stable memory estimates for pool tests (no HuggingFace download)."""

    weights = {
        "Qwen/Qwen2.5-0.5B-Instruct": 1.0,
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2.05,
        "meta-llama/Meta-Llama-3-8B-Instruct": 15.0,
        "org/a": 1.0,
    }

    def weight(path: str) -> float:
        return weights.get(path, 1.5)

    def kv(path: str, max_model_len: int) -> float:
        if "Qwen" in path:
            return 0.15 * (max_model_len / 2048)
        if "TinyLlama" in path:
            return 0.4
        if "llama" in path.lower():
            return 1.0 * (max_model_len / 4096)
        return 0.3

    monkeypatch.setattr(pool, "estimate_weight_gib", weight)
    monkeypatch.setattr(pool, "estimate_kv_cache_gib", kv)
    pool._hf_config_dict.cache_clear()


def _qwen_cfg() -> dict:
    return {
        "name": "qwen2.5-0.5b",
        "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_model_len": 4096,
    }


def _tiny_cfg() -> dict:
    return {
        "name": "tinyllama-chat",
        "model_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_model_len": 2048,
    }


def test_single_model_computes_utilization(fixed_footprints):
    cfg = {"name": "a", "model_path": "org/a", "max_model_len": 2048}
    scaled = scale_model_config_for_pool(cfg, pool_size=1, total_vram_gib=8.0)
    assert scaled["gpu_memory_utilization"] > 0.1
    assert scaled["gpu_memory_utilization"] <= 0.92


def test_single_model_respects_free_vram_cap(fixed_footprints):
    cfg = _qwen_cfg()
    scaled = scale_model_config_for_pool(
        cfg,
        pool_size=1,
        total_vram_gib=6.0,
        free_vram_gib=2.4,
    )
    util = scaled["gpu_memory_utilization"]
    assert util * 6.0 <= 2.4 * 0.98 + 0.05
    assert util >= 0.33


def test_single_model_raises_when_free_vram_too_low(fixed_footprints):
    cfg = _tiny_cfg()
    with pytest.raises(ValueError, match="VRAM free"):
        scale_model_config_for_pool(
            cfg,
            pool_size=1,
            total_vram_gib=6.0,
            free_vram_gib=1.0,
        )


def test_estimate_weight_from_model_name():
    pool._hf_config_dict.cache_clear()
    weight = pool.estimate_weight_gib("Qwen/Qwen2.5-0.5B-Instruct")
    assert 0.7 < weight < 1.2


def test_single_model_respects_user_cap(fixed_footprints):
    cfg = {
        "name": "a",
        "model_path": "org/a",
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.25,
    }
    scaled = scale_model_config_for_pool(cfg, pool_size=1, total_vram_gib=8.0)
    assert scaled["gpu_memory_utilization"] <= 0.25


def test_two_model_pool_uses_weight_aware_share(fixed_footprints):
    qwen = _qwen_cfg()
    tiny = _tiny_cfg()
    pool_models = ["qwen2.5-0.5b", "tinyllama-chat"]
    pool_configs = [qwen, tiny]

    q_scaled = scale_model_config_for_pool(
        qwen,
        pool_size=2,
        pool_models=pool_models,
        pool_configs=pool_configs,
        engine_index=0,
        total_vram_gib=8.0,
    )
    t_scaled = scale_model_config_for_pool(
        tiny,
        pool_size=2,
        pool_models=pool_models,
        pool_configs=pool_configs,
        engine_index=1,
        free_vram_gib=3.5,
        total_vram_gib=8.0,
    )

    assert q_scaled["gpu_memory_utilization"] < t_scaled["gpu_memory_utilization"]
    assert q_scaled["gpu_memory_utilization"] == pytest.approx(0.175, abs=0.03)
    assert t_scaled["gpu_memory_utilization"] >= 0.35
    assert q_scaled["max_model_len"] == 2048


def test_sequential_cap_limits_second_engine_to_free_vram(fixed_footprints):
    tiny = _tiny_cfg()
    pool_models = ["qwen2.5-0.5b", "tinyllama-chat"]
    pool_configs = [_qwen_cfg(), tiny]
    scaled = scale_model_config_for_pool(
        tiny,
        pool_size=2,
        pool_models=pool_models,
        pool_configs=pool_configs,
        engine_index=1,
        free_vram_gib=2.3,
        total_vram_gib=8.0,
    )
    assert scaled["gpu_memory_utilization"] < 0.38
    assert scaled["gpu_memory_utilization"] >= 0.25


def test_validate_pool_fits_rejects_impossible_combo(fixed_footprints):
    with pytest.raises(ValueError, match="need ~"):
        validate_pool_fits(
            [_qwen_cfg(), {"name": "llama3-8b", "model_path": "meta-llama/Meta-Llama-3-8B-Instruct", "max_model_len": 4096}],
            total_vram_gib=8.0,
        )


def test_validate_pool_fits_allows_small_pair(fixed_footprints):
    validate_pool_fits([_qwen_cfg(), _tiny_cfg()], total_vram_gib=8.0)


def test_validate_pool_fits_rejects_dual_model_on_6gb(fixed_footprints):
    with pytest.raises(ValueError, match="need ~|cannot load sequentially"):
        validate_pool_fits([_qwen_cfg(), _tiny_cfg()], total_vram_gib=6.0)
