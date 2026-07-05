"""Tests that _build_engine_pool resolves the VRAM tier and applies engine knobs
before constructing each VLLMEngine (see add-engine-knob-surfacing)."""
from __future__ import annotations

import pytest

import inference_x.api.deps as deps_module
from inference_x.api.deps import _build_registry, _build_router
from inference_x.core.settings import get_settings
from inference_x.utils.vram_tiers import VramTier


@pytest.fixture(autouse=True)
def _clear_caches():
    from inference_x.api.deps import _build_engine_pool

    get_settings.cache_clear()
    _build_registry.cache_clear()
    _build_router.cache_clear()
    _build_engine_pool.cache_clear()
    yield
    get_settings.cache_clear()
    _build_registry.cache_clear()
    _build_router.cache_clear()
    _build_engine_pool.cache_clear()


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


def _write_models_yaml(config_dir, extra: str = "") -> None:
    (config_dir / "models.yaml").write_text(
        "models:\n"
        "  - name: alpha\n"
        "    engine: vllm\n"
        "    model_path: org/alpha\n" + extra
    )


class _RecordingVLLMEngine:
    captured_configs: list[dict] = []

    def __init__(self, model_config, **kwargs):
        _RecordingVLLMEngine.captured_configs.append(model_config)

    def is_healthy(self) -> bool:
        return True

    def shutdown(self) -> None:
        pass


def test_build_engine_pool_applies_tier_knobs(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_models_yaml(config_dir)
    monkeypatch.setenv("INFERENCE_X_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("INFERENCE_X_LOADED_MODELS", "alpha")
    get_settings.cache_clear()

    _RecordingVLLMEngine.captured_configs = []
    monkeypatch.setattr(deps_module, "VLLMEngine", _RecordingVLLMEngine)
    monkeypatch.setattr(
        deps_module, "_resolve_vram_tier_for_pool", lambda config_dir: _tier()
    )
    monkeypatch.setattr(deps_module, "validate_pool_fits", lambda *a, **kw: None)
    monkeypatch.setattr(deps_module, "probe_gpu_memory_gib", lambda: (4.0, 8.0))

    from inference_x.api.deps import _build_engine_pool

    _build_engine_pool(str(config_dir), ("alpha",))

    assert len(_RecordingVLLMEngine.captured_configs) == 1
    config = _RecordingVLLMEngine.captured_configs[0]
    assert config["max_num_seqs"] == 4
    assert config["max_num_batched_tokens"] == 2048
    assert config["block_size"] == 16
    assert config["kv_cache_dtype"] == "auto"
    assert config["enable_prefix_caching"] is False


def test_build_engine_pool_falls_back_when_tier_unresolvable(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_models_yaml(config_dir)
    monkeypatch.setenv("INFERENCE_X_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("INFERENCE_X_LOADED_MODELS", "alpha")
    get_settings.cache_clear()

    _RecordingVLLMEngine.captured_configs = []
    monkeypatch.setattr(deps_module, "VLLMEngine", _RecordingVLLMEngine)
    monkeypatch.setattr(deps_module, "_resolve_vram_tier_for_pool", lambda config_dir: None)
    monkeypatch.setattr(deps_module, "validate_pool_fits", lambda *a, **kw: None)
    monkeypatch.setattr(deps_module, "probe_gpu_memory_gib", lambda: (4.0, 8.0))

    from inference_x.api.deps import _build_engine_pool

    _build_engine_pool(str(config_dir), ("alpha",))

    config = _RecordingVLLMEngine.captured_configs[0]
    # ModelEntry declares max_num_seqs/max_num_batched_tokens (default None); block_size
    # and kv_cache_dtype aren't ModelEntry fields at all, so they're simply absent.
    assert config["max_num_seqs"] is None
    assert config["max_num_batched_tokens"] is None
    assert "block_size" not in config
    assert "kv_cache_dtype" not in config


def test_resolve_vram_tier_for_pool_fails_open_on_error(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    def _raise():
        raise RuntimeError("no GPU")

    fake_settings = type("FakeSettings", (), {"get_vram_tier": staticmethod(_raise)})()
    monkeypatch.setattr(deps_module, "get_settings", lambda: fake_settings)

    result = deps_module._resolve_vram_tier_for_pool(str(config_dir))
    assert result is None
