"""Tests for single-model GPU memory scaling (B6: one model per process)."""

import sys

import pytest

from inference_x.utils import vllm_pool_config as pool
from inference_x.utils.vllm_pool_config import (
    apply_tier_knobs,
    scale_model_config,
    validate_model_fits,
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
    scaled = scale_model_config(cfg, total_vram_gib=8.0)
    assert scaled["gpu_memory_utilization"] > 0.1
    assert scaled["gpu_memory_utilization"] <= 0.92


def test_single_model_respects_free_vram_cap(fixed_footprints):
    cfg = _qwen_cfg()
    scaled = scale_model_config(
        cfg,
        total_vram_gib=6.0,
        free_vram_gib=2.4,
    )
    util = scaled["gpu_memory_utilization"]
    assert util * 6.0 <= 2.4 * 0.98 + 0.05
    assert util >= 0.33


def test_single_model_raises_when_free_vram_too_low(fixed_footprints):
    cfg = _tiny_cfg()
    with pytest.raises(ValueError, match="VRAM free"):
        scale_model_config(
            cfg,
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
    scaled = scale_model_config(cfg, total_vram_gib=8.0)
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
    auto_scaled = scale_model_config(auto_cfg, total_vram_gib=8.0)

    non_binding_cfg = _qwen_cfg()
    non_binding_cfg["gpu_memory_utilization"] = 0.9  # above the footprint, so it never binds
    explicit_scaled = scale_model_config(non_binding_cfg, total_vram_gib=8.0)

    assert auto_scaled["gpu_memory_utilization"] == explicit_scaled["gpu_memory_utilization"]
    assert auto_scaled["gpu_memory_utilization"] < 0.3  # small model, not a flat 0.82


def test_validate_model_fits_rejects_when_free_vram_too_low(fixed_footprints):
    with pytest.raises(ValueError, match="VRAM free"):
        validate_model_fits(_tiny_cfg(), total_vram_gib=6.0, free_vram_gib=1.0)


def test_validate_model_fits_allows_model_with_ample_free_vram(fixed_footprints):
    validate_model_fits(_qwen_cfg(), total_vram_gib=8.0, free_vram_gib=6.0)


def test_validate_model_fits_skips_check_when_free_vram_unknown(fixed_footprints):
    """No probe reading available (free_vram_gib=None) -> no raise, matching
    scale_model_config's own fail-open posture when the probe is unavailable."""
    validate_model_fits(_tiny_cfg(), total_vram_gib=6.0, free_vram_gib=None)


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


# ---------------------------------------------------------------------------
# probe_gpu_memory_gib (B6: nvidia-smi first, torch.cuda.mem_get_info() as a
# loud fallback — see docs/DECISIONS.md DEC-059)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    pool._probe_cache = None
    yield
    pool._probe_cache = None


def test_probe_gpu_memory_gib_uses_nvidia_smi(monkeypatch):
    class _Result:
        stdout = "1234, 8188\n"

    monkeypatch.setattr(pool.subprocess, "run", lambda *a, **kw: _Result())
    free, total = pool.probe_gpu_memory_gib()
    assert free == pytest.approx(1234 / 1024)
    assert total == pytest.approx(8188 / 1024)


def test_probe_gpu_memory_gib_falls_back_to_torch_with_warning(monkeypatch):
    def _raise(*a, **kw):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(pool.subprocess, "run", _raise)

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def mem_get_info(_index):
            return 2 * 1024**3, 8 * 1024**3

    fake_torch = type("FakeTorch", (), {"cuda": _FakeCuda})()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    warnings: list[str] = []
    monkeypatch.setattr(
        pool.logger, "warning", lambda msg, *a, **kw: warnings.append(msg % a if a else msg)
    )

    free, total = pool.probe_gpu_memory_gib()
    assert free == pytest.approx(2.0)
    assert total == pytest.approx(8.0)
    assert any("stale" in w for w in warnings)


def test_probe_gpu_memory_gib_returns_none_when_nothing_available(monkeypatch):
    def _raise(*a, **kw):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(pool.subprocess, "run", _raise)

    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

    fake_torch = type("FakeTorch", (), {"cuda": _FakeCuda})()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert pool.probe_gpu_memory_gib() == (None, None)


def test_probe_gpu_memory_gib_caches_briefly(monkeypatch):
    calls = {"n": 0}

    class _Result:
        stdout = "1000, 8000\n"

    def _run(*a, **kw):
        calls["n"] += 1
        return _Result()

    monkeypatch.setattr(pool.subprocess, "run", _run)
    pool.probe_gpu_memory_gib()
    pool.probe_gpu_memory_gib()
    assert calls["n"] == 1
