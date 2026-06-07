"""Tests for multi-model GPU memory scaling."""

import pytest

from inference_x.utils.vllm_pool_config import (
    scale_model_config_for_pool,
    validate_pool_fits,
)


def test_single_model_config_unchanged():
    cfg = {"name": "a", "model_path": "org/a", "gpu_memory_utilization": 0.5}
    assert scale_model_config_for_pool(cfg, pool_size=1) == cfg


def test_two_model_pool_uses_weight_aware_share():
    qwen = {
        "name": "qwen2.5-0.5b",
        "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "gpu_memory_utilization": 0.5,
        "max_model_len": 4096,
    }
    tiny = {
        "name": "tinyllama-chat",
        "model_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "gpu_memory_utilization": 0.5,
        "max_model_len": 2048,
    }
    pool = ["qwen2.5-0.5b", "tinyllama-chat"]

    q_scaled = scale_model_config_for_pool(
        qwen, pool_size=2, pool_models=pool, engine_index=0
    )
    t_scaled = scale_model_config_for_pool(
        tiny,
        pool_size=2,
        pool_models=pool,
        engine_index=1,
        free_vram_gib=3.5,
    )

    assert q_scaled["gpu_memory_utilization"] < t_scaled["gpu_memory_utilization"]
    assert q_scaled["gpu_memory_utilization"] == pytest.approx(0.2, abs=0.02)
    assert t_scaled["gpu_memory_utilization"] >= 0.38
    assert q_scaled["max_model_len"] == 2048


def test_sequential_cap_limits_second_engine_to_free_vram():
    tiny = {
        "name": "tinyllama-chat",
        "model_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "gpu_memory_utilization": 0.5,
    }
    pool = ["qwen2.5-0.5b", "tinyllama-chat"]
    scaled = scale_model_config_for_pool(
        tiny,
        pool_size=2,
        pool_models=pool,
        engine_index=1,
        free_vram_gib=2.3,
        total_vram_gib=8.0,
    )
    assert scaled["gpu_memory_utilization"] < 0.38
    assert scaled["gpu_memory_utilization"] >= 0.25


def test_validate_pool_fits_rejects_impossible_combo():
    with pytest.raises(ValueError, match="need ~"):
        validate_pool_fits(["qwen2.5-0.5b", "llama3-8b"], total_vram_gib=8.0)


def test_validate_pool_fits_allows_small_pair():
    validate_pool_fits(["qwen2.5-0.5b", "tinyllama-chat"], total_vram_gib=8.0)
