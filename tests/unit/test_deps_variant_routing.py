"""Tests that _build_engine_pool resolves family names to concrete variants
before constructing engines (see add-model-variant-routing)."""
from __future__ import annotations

import pytest

import inference_x.api.deps as deps_module
from inference_x.api.deps import _build_registry, _build_router, _resolve_loaded_model_names
from inference_x.core.settings import get_settings
from inference_x.schemas.model import ModelEntry
from inference_x.services.model_service import ModelRegistry
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


def _family_registry() -> ModelRegistry:
    return ModelRegistry(
        [
            ModelEntry(name="qwen-bf16", model_path="org/qwen-bf16", family="qwen"),
            ModelEntry(
                name="qwen-awq", model_path="org/qwen-awq", family="qwen", quantization="awq"
            ),
            ModelEntry(name="solo", model_path="org/solo"),
        ]
    )


class TestResolveLoadedModelNames:
    def test_concrete_name_bypasses_selector(self):
        registry = _family_registry()
        resolved = _resolve_loaded_model_names(registry, ("solo",), _tier(), 8.0)
        assert resolved == ["solo"]

    def test_family_name_resolves_to_best_fitting_variant(self, monkeypatch):
        registry = _family_registry()
        monkeypatch.setattr(
            "inference_x.routing.variant_selector.estimate_weight_gib",
            lambda path, q=None: 14.0 if q is None else 4.0,
        )
        # 8 GiB * 0.9 = 7.2 GiB budget -> bf16 (14) doesn't fit, awq (4) does.
        resolved = _resolve_loaded_model_names(registry, ("qwen",), _tier(), 8.0)
        assert resolved == ["qwen-awq"]

    def test_unresolvable_name_raises_not_registered_error(self):
        registry = _family_registry()
        with pytest.raises(ValueError, match="not registered"):
            _resolve_loaded_model_names(registry, ("nonexistent",), _tier(), 8.0)

    def test_unresolvable_name_raises_not_registered_error_when_no_tier(self):
        """No tier resolved -> even a real family name can't be sized, so it
        falls through to the standard registry.get() error, unchanged from
        before variant routing existed."""
        registry = _family_registry()
        with pytest.raises(ValueError, match="not registered"):
            _resolve_loaded_model_names(registry, ("qwen",), None, 8.0)

    def test_mixed_concrete_and_family_names(self, monkeypatch):
        registry = _family_registry()
        monkeypatch.setattr(
            "inference_x.routing.variant_selector.estimate_weight_gib",
            lambda path, q=None: 14.0 if q is None else 4.0,
        )
        resolved = _resolve_loaded_model_names(registry, ("solo", "qwen"), _tier(), 8.0)
        assert resolved == ["solo", "qwen-awq"]


class _RecordingVLLMEngine:
    captured_configs: list[dict] = []

    def __init__(self, model_config, **kwargs):
        _RecordingVLLMEngine.captured_configs.append(model_config)

    def is_healthy(self) -> bool:
        return True

    def shutdown(self) -> None:
        pass


def test_build_engine_pool_resolves_family_env_var(monkeypatch, tmp_path):
    """End-to-end: INFERENCE_X_LOADED_MODELS set to a family name resolves to
    a concrete variant before VLLMEngine is constructed."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "models.yaml").write_text(
        "models:\n"
        "  - name: qwen-bf16\n"
        "    engine: vllm\n"
        "    model_path: org/qwen-bf16\n"
        "    family: qwen\n"
        "  - name: qwen-awq\n"
        "    engine: vllm\n"
        "    model_path: org/qwen-awq\n"
        "    family: qwen\n"
        "    quantization: awq\n"
    )
    monkeypatch.setenv("INFERENCE_X_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("INFERENCE_X_LOADED_MODELS", "qwen")
    get_settings.cache_clear()

    _RecordingVLLMEngine.captured_configs = []
    monkeypatch.setattr(deps_module, "VLLMEngine", _RecordingVLLMEngine)
    monkeypatch.setattr(deps_module, "_resolve_vram_tier_for_pool", lambda config_dir: _tier())
    monkeypatch.setattr(deps_module, "validate_pool_fits", lambda *a, **kw: None)
    monkeypatch.setattr(deps_module, "probe_gpu_memory_gib", lambda: (4.0, 8.0))
    monkeypatch.setattr(
        "inference_x.routing.variant_selector.estimate_weight_gib",
        lambda path, q=None: 14.0 if q is None else 2.0,
    )

    from inference_x.api.deps import _build_engine_pool

    pool = _build_engine_pool(str(config_dir), ("qwen",))

    assert pool.loaded_models() == ["qwen-awq"]
    assert len(_RecordingVLLMEngine.captured_configs) == 1
