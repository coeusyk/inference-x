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
def test_estimate_weight_gib_is_quantization_aware(monkeypatch, quantization, expected_ratio):
    """Ratios assume no HF config (no embedding/lm_head split to apply) — the
    split-correction behavior itself is covered by
    test_awq_underestimate_fixed_for_untied_large_vocab below."""
    pool._hf_config_dict.cache_clear()
    monkeypatch.setattr(pool, "_hf_config_dict", lambda path: None)
    bf16 = pool.estimate_weight_gib("Qwen/Qwen2.5-0.5B-Instruct")
    quantized = pool.estimate_weight_gib("Qwen/Qwen2.5-0.5B-Instruct", quantization)
    assert quantized == pytest.approx(bf16 * expected_ratio, rel=1e-6)


def test_awq_underestimate_fixed_for_untied_large_vocab(monkeypatch):
    """Regression for qwen2.5-7b-awq (2026-07-02): AWQ leaves embedding/lm_head
    at bf16, and untied embeddings + a 152k vocab meant the old uniform-ratio
    estimate undercounted weight VRAM by ~2 GiB, causing a real load failure.
    No network: HF config is mocked.
    """
    pool._hf_config_dict.cache_clear()
    fake_config = {
        "hidden_size": 3584,
        "vocab_size": 152064,
        "tie_word_embeddings": False,
    }
    monkeypatch.setattr(pool, "_hf_config_dict", lambda path: fake_config)

    total_params = 7_000_000_000
    old_style_estimate_gib = (total_params * pool._bytes_per_param("awq")) / (1024**3)
    new_estimate_gib = pool.estimate_weight_gib("Qwen/Qwen2.5-7B-Instruct-AWQ", "awq")

    assert new_estimate_gib > old_style_estimate_gib
    embed_params = 2 * 152064 * 3584  # untied: input embedding + lm_head
    embed_gib = (embed_params * 2) / (1024**3)  # unquantized -> bf16 (2 bytes/param)
    quantized_gib = ((total_params - embed_params) * pool._bytes_per_param("awq")) / (1024**3)
    assert new_estimate_gib == pytest.approx(embed_gib + quantized_gib, rel=1e-6)


def test_minicpm_gets_conservative_overhead_margin(monkeypatch):
    """Regression for minicpm5-1b (2026-07-02): measured 6.55 GiB peak VRAM vs.
    a 3.79 GiB estimated budget, with no confirmed architecture-specific root
    cause. Rather than pretend the estimate is exact, apply a conservative
    margin so the budget isn't unrealistically low.

    Matched on model_path, not HF config `model_type`: openbmb/MiniCPM5-1B's
    real config self-reports model_type="llama" (architectures=
    ["LlamaForCausalLM"]) for tooling compatibility, so model_type can't be
    used to distinguish it — confirmed live on this hardware (2026-07-02).
    """
    overhead = pool._architecture_overhead_gib("openbmb/MiniCPM5-1B")
    assert overhead > 0

    with_overhead = pool.estimate_engine_footprint_gib("openbmb/MiniCPM5-1B", 8192)
    monkeypatch.setattr(pool, "_architecture_overhead_gib", lambda path: 0.0)
    without_overhead = pool.estimate_engine_footprint_gib("openbmb/MiniCPM5-1B", 8192)
    assert with_overhead > without_overhead


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


def test_auto_matches_explicit_footprint_sizing(fixed_footprints):
    """"auto" must size by the model's own footprint, like a non-binding explicit
    ceiling would — not a flat (free - buffer)/total ratio ignoring model size.

    Regression guard: a prior bug let "auto" grab ~(free - buffer)/total of
    VRAM regardless of the model's footprint, so a small model could claim
    >85% of an 8 GiB GPU (see DEC on the opt-125m benchmark investigation).
    """
    auto_cfg = _qwen_cfg()
    auto_cfg["gpu_memory_utilization"] = "auto"
    auto_scaled = scale_model_config_for_pool(auto_cfg, pool_size=1, total_vram_gib=8.0)

    non_binding_cfg = _qwen_cfg()
    non_binding_cfg["gpu_memory_utilization"] = 0.9  # above the footprint, so it never binds
    explicit_scaled = scale_model_config_for_pool(
        non_binding_cfg, pool_size=1, total_vram_gib=8.0
    )

    assert auto_scaled["gpu_memory_utilization"] == explicit_scaled["gpu_memory_utilization"]
    assert auto_scaled["gpu_memory_utilization"] < 0.3  # small model, not a flat 0.82


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
    # 0.2625, not the pre-DEC-046 0.175 — _multi_engine_overhead_gib's old flat
    # ~3.35 GiB reservation left less room here than the corrected 0.6 GiB does.
    assert q_scaled["gpu_memory_utilization"] == pytest.approx(0.2625, abs=0.03)
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


def test_auto_multi_model_uses_weight_scaled_sizing(fixed_footprints):
    """"auto" in a multi-model pool must weight-scale by footprint like an
    explicit ceiling would, not grab a flat (free - buffer)/total ratio with a
    0.50 floor regardless of model size (same bug class as
    test_auto_matches_explicit_footprint_sizing, in the pool_size>1 path).
    """
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
    # opt is the heavier mock footprint here, so it earns the larger weight share.
    assert opt_scaled["gpu_memory_utilization"] > qwen_scaled["gpu_memory_utilization"]
    assert 0.25 < opt_scaled["gpu_memory_utilization"] < 0.45
    assert 0.15 < qwen_scaled["gpu_memory_utilization"] < 0.35


def test_validate_pool_fits_rejects_impossible_combo(fixed_footprints):
    with pytest.raises(ValueError, match="need ~"):
        validate_pool_fits(
            [_qwen_cfg(), {"name": "llama3-8b", "model_path": "meta-llama/Meta-Llama-3-8B-Instruct", "max_model_len": 4096}],
            total_vram_gib=8.0,
        )


def test_validate_pool_fits_allows_small_pair(fixed_footprints):
    validate_pool_fits([_qwen_cfg(), _tiny_cfg()], total_vram_gib=8.0)


def test_validate_pool_fits_allows_real_compare_pair_on_8gib(monkeypatch):
    """Regression for the make playground compare-mode failure (2026-07-03):
    qwen2.5-0.5b + qwen2.5-1.5b at max_model_len=8192 each (~7.3 GiB combined,
    comfortably under 8 GiB) was rejected with a *negative* allowed
    utilization by _apply_sequential_vram_caps. Root cause was
    _multi_engine_overhead_gib's flat ~3.35 GiB reservation (42% of an 8 GiB
    card) on top of the next engine's full footprint, never subtracted from
    the pool-level budget validate_pool_fits() itself already confirmed fit.
    Live-verified after the fix: both engines load and serve real completions
    together on an 8 GiB card with ~0.74 GiB still free (DEC-046).

    Configs are real HF values (hidden_size/vocab_size/num_hidden_layers) for
    Qwen2.5-0.5B/1.5B-Instruct, mocked to avoid a network/cache dependency.
    """
    pool._hf_config_dict.cache_clear()
    configs = {
        "Qwen/Qwen2.5-0.5B-Instruct": {
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "vocab_size": 151936,
            "intermediate_size": 4864,
            "tie_word_embeddings": True,
        },
        "Qwen/Qwen2.5-1.5B-Instruct": {
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "vocab_size": 151936,
            "intermediate_size": 8960,
            "tie_word_embeddings": True,
        },
    }
    monkeypatch.setattr(pool, "_hf_config_dict", lambda path: configs.get(path))

    validate_pool_fits(
        [
            {"name": "qwen2.5-0.5b", "model_path": "Qwen/Qwen2.5-0.5B-Instruct", "max_model_len": 8192},
            {"name": "qwen2.5-1.5b", "model_path": "Qwen/Qwen2.5-1.5B-Instruct", "max_model_len": 8192},
        ],
        total_vram_gib=8.0,
    )


def test_validate_pool_fits_rejects_dual_model_on_6gb(fixed_footprints):
    """qwen2.5-0.5b + llama3-8b: llama3-8b alone (15 GiB mocked) can't fit any
    pool on a 6 GiB card regardless of overhead calibration — unlike
    qwen2.5-0.5b + tinyllama-chat, which _multi_engine_overhead_gib's old
    flat ~3.35 GiB reservation wrongly rejected on 8 GiB (see DEC-046) and,
    it turns out, would have wrongly rejected here too."""
    with pytest.raises(ValueError, match="need ~|cannot load sequentially"):
        validate_pool_fits(
            [
                _qwen_cfg(),
                {
                    "name": "llama3-8b",
                    "model_path": "meta-llama/Meta-Llama-3-8B-Instruct",
                    "max_model_len": 4096,
                },
            ],
            total_vram_gib=6.0,
        )


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
