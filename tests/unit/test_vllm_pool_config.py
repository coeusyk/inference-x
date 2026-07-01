"""Tests for multi-model GPU memory scaling."""

import pytest

from inference_x.utils import vllm_pool_config as pool
from inference_x.utils.vllm_pool_config import (
    apply_tier_knobs,
    scale_model_config_for_pool,
    validate_pool_fits,
)
from inference_x.utils.vram_tiers import VramTier


def _tier(**overrides) -> VramTier:
    defaults = dict(
        name="6gb",
        min_vram_gb=0,
        description="test tier",
        gpu_memory_utilization_ceiling=0.90,
        max_model_len_cap=2048,
        max_num_seqs=4,
        block_size=16,
        kv_cache_dtype="auto",
        max_num_batched_tokens=2048,
        enable_prefix_caching=False,
    )
    defaults.update(overrides)
    return VramTier(**defaults)


@pytest.fixture
def fixed_footprints(monkeypatch):
    """Stable memory estimates for pool tests (no HuggingFace download)."""

    weights = {
        "Qwen/Qwen2.5-0.5B-Instruct": 1.0,
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2.05,
        "meta-llama/Meta-Llama-3-8B-Instruct": 15.0,
        "org/a": 1.0,
    }

    def weight(path: str, quantization: str | None = None) -> float:
        base = weights.get(path, 1.5)
        if quantization:
            return base * 0.28  # ~0.55/2 bytes-per-param ratio for 4-bit quant
        return base

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


@pytest.mark.parametrize(
    ("quantization", "expected_ratio"),
    [
        (None, 1.0),
        ("awq", 0.275),
        ("gptq", 0.275),
        ("awq_marlin", 0.275),
        ("int4", 0.275),
        ("int8", 0.5),
        ("fp8", 0.5),
        ("some-custom-gptq-variant", 0.275),  # substring fallback
        ("unknown-scheme", 1.0),  # unrecognized -> bf16 default
    ],
)
def test_estimate_weight_gib_is_quantization_aware(quantization, expected_ratio):
    pool._hf_config_dict.cache_clear()
    bf16 = pool.estimate_weight_gib("Qwen/Qwen2.5-0.5B-Instruct")
    quantized = pool.estimate_weight_gib("Qwen/Qwen2.5-0.5B-Instruct", quantization)
    assert quantized == pytest.approx(bf16 * expected_ratio, rel=1e-6)


def test_bytes_per_param_lookup():
    assert pool._bytes_per_param(None) == 2
    assert pool._bytes_per_param("awq") == 0.55
    assert pool._bytes_per_param("GPTQ") == 0.55  # case-insensitive
    assert pool._bytes_per_param("int8") == 1.0
    assert pool._bytes_per_param("totally-unknown") == 2


def test_single_model_respects_user_cap(fixed_footprints):
    cfg = {
        "name": "a",
        "model_path": "org/a",
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.25,
    }
    scaled = scale_model_config_for_pool(cfg, pool_size=1, total_vram_gib=8.0)
    assert scaled["gpu_memory_utilization"] <= 0.25


def test_auto_resolves_at_startup(fixed_footprints, monkeypatch):
    monkeypatch.setattr(
        "inference_x.benchmarks.hardware._vram_for_utilization",
        lambda: (6.93, 8.0, "torch"),
    )
    monkeypatch.setattr(
        "inference_x.benchmarks.hardware._probe_torch_vram_gib",
        lambda: (6.93, 8.0),
    )

    cfg = _qwen_cfg()
    cfg["gpu_memory_utilization"] = "auto"
    scaled = scale_model_config_for_pool(cfg, pool_size=1, total_vram_gib=8.0)
    assert scaled["gpu_memory_utilization"] == 0.82


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


def test_auto_multi_model_uses_fresh_suggest_not_div_n(fixed_footprints):
    """Auto in a 2-model pool samples (free − buffer) / total per engine, not ÷N."""
    qwen = _qwen_cfg()
    qwen["gpu_memory_utilization"] = "auto"
    opt = {
        "name": "opt-125m",
        "model_path": "facebook/opt-125m",
        "max_model_len": 2048,
        "gpu_memory_utilization": "auto",
    }
    pool_configs = [opt, qwen]
    opt_scaled = scale_model_config_for_pool(
        opt,
        pool_size=2,
        pool_configs=pool_configs,
        engine_index=0,
        free_vram_gib=6.93,
        total_vram_gib=8.0,
        session_free_vram_gib=6.93,
    )
    qwen_scaled = scale_model_config_for_pool(
        qwen,
        pool_size=2,
        pool_configs=pool_configs,
        engine_index=1,
        free_vram_gib=3.5,
        total_vram_gib=8.0,
        session_free_vram_gib=6.93,
    )
    assert opt_scaled["gpu_memory_utilization"] != 0.41
    assert 0.25 < opt_scaled["gpu_memory_utilization"] < 0.45
    assert qwen_scaled["gpu_memory_utilization"] != 0.41
    assert qwen_scaled["gpu_memory_utilization"] >= 0.42


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


# ---------------------------------------------------------------------------
# apply_tier_knobs
# ---------------------------------------------------------------------------


def test_apply_tier_knobs_no_tier_leaves_config_unchanged():
    config = {"name": "m", "model_path": "org/m"}
    resolved = apply_tier_knobs(config, None)
    assert resolved == config
    assert resolved is not config  # still a copy, not the same object


def test_apply_tier_knobs_uses_tier_defaults_when_no_model_override():
    config = {"name": "m", "model_path": "org/m"}
    resolved = apply_tier_knobs(config, _tier())
    assert resolved["max_num_seqs"] == 4
    assert resolved["max_num_batched_tokens"] == 2048
    assert resolved["block_size"] == 16
    assert resolved["kv_cache_dtype"] == "auto"
    assert resolved["enable_prefix_caching"] is False


def test_apply_tier_knobs_honors_tighter_model_override():
    config = {"name": "m", "model_path": "org/m", "max_num_seqs": 2, "max_num_batched_tokens": 512}
    resolved = apply_tier_knobs(config, _tier())
    assert resolved["max_num_seqs"] == 2
    assert resolved["max_num_batched_tokens"] == 512


def test_apply_tier_knobs_clamps_looser_model_override_down_to_tier():
    config = {"name": "m", "model_path": "org/m", "max_num_seqs": 64, "max_num_batched_tokens": 999999}
    resolved = apply_tier_knobs(config, _tier())
    assert resolved["max_num_seqs"] == 4
    assert resolved["max_num_batched_tokens"] == 2048


def test_apply_tier_knobs_enable_prefix_caching_is_tier_only():
    config = {"name": "m", "model_path": "org/m"}
    resolved = apply_tier_knobs(config, _tier(name="12gb", enable_prefix_caching=True))
    assert resolved["enable_prefix_caching"] is True


def test_apply_tier_knobs_skips_max_num_batched_tokens_when_tier_omits_it():
    config = {"name": "m", "model_path": "org/m", "max_num_batched_tokens": 1024}
    resolved = apply_tier_knobs(config, _tier(max_num_batched_tokens=None))
    assert resolved["max_num_batched_tokens"] == 1024  # left untouched, not cleared
