"""Tests for INFERENCE_X_DEFAULT_MODEL family-name resolution (DEC-042).

_resolve_default_model (api/deps.py) closes the DEC-041 scope boundary:
INFERENCE_X_DEFAULT_MODEL previously required an exact registered model
name; a family name (ModelEntry.family) now resolves to its best-fitting
variant via variant_selector.select_variant(), the same rule
_resolve_loaded_model_names already applies to INFERENCE_X_LOADED_MODELS.
"""
from __future__ import annotations

import pytest

import inference_x.api.deps as deps_module
from inference_x.api.deps import _build_registry, _build_router, _resolve_default_model
from inference_x.core.settings import get_settings
from inference_x.routing.variant_selector import NoVariantFitsError
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


class TestResolveDefaultModel:
    def test_family_name_resolves_to_highest_precision_fitting_variant(self, monkeypatch):
        registry = _family_registry()
        monkeypatch.setattr(
            "inference_x.routing.variant_selector.estimate_weight_gib",
            lambda path, q=None: 14.0 if q is None else 4.0,
        )
        # 8 GiB * 0.9 ceiling = 7.2 GiB budget -> bf16 (14) doesn't fit, awq (4) does.
        resolved = _resolve_default_model(registry, "qwen", _tier(), available_vram_gib=8.0)
        assert resolved == "qwen-awq"

    def test_concrete_model_name_passes_through_unchanged(self):
        registry = _family_registry()
        resolved = _resolve_default_model(registry, "solo", _tier(), available_vram_gib=8.0)
        assert resolved == "solo"

    def test_family_name_with_no_fitting_variant_raises_runtime_error(self, monkeypatch):
        registry = _family_registry()
        monkeypatch.setattr(
            "inference_x.routing.variant_selector.estimate_weight_gib",
            lambda path, q=None: 999.0,
        )
        with pytest.raises(RuntimeError, match="qwen"):
            _resolve_default_model(registry, "qwen", _tier(), available_vram_gib=8.0)
        # NoVariantFitsError is itself a RuntimeError -- confirm that's what's raised.
        with pytest.raises(NoVariantFitsError):
            _resolve_default_model(registry, "qwen", _tier(), available_vram_gib=8.0)

    def test_family_name_not_in_registry_is_treated_as_concrete_name(self):
        """An unrecognized value (not a concrete name, not a known family) is
        returned unchanged so TaskRouter's DefaultModelPolicy raises its own
        existing 'not in the registry' error -- unchanged from before this
        function existed."""
        registry = _family_registry()
        resolved = _resolve_default_model(
            registry, "nonexistent", _tier(), available_vram_gib=8.0
        )
        assert resolved == "nonexistent"

    def test_no_tier_leaves_value_unchanged(self):
        """No VRAM tier resolved -> can't size variants, so even a real family
        name is left unchanged (fail-open, consistent with the rest of this
        codebase's posture when tier resolution is unavailable)."""
        registry = _family_registry()
        resolved = _resolve_default_model(registry, "qwen", None, available_vram_gib=8.0)
        assert resolved == "qwen"


def test_build_router_resolves_family_default_model_end_to_end(monkeypatch, tmp_path):
    """End-to-end: INFERENCE_X_DEFAULT_MODEL set to a family name resolves to a
    concrete variant before TaskRouter/DefaultModelPolicy ever see it."""
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
    monkeypatch.setenv("INFERENCE_X_DEFAULT_MODEL", "qwen")
    get_settings.cache_clear()

    monkeypatch.setattr(deps_module, "_resolve_vram_tier_for_pool", lambda config_dir: _tier())
    monkeypatch.setattr(deps_module, "probe_gpu_memory_gib", lambda: (4.0, 8.0))
    monkeypatch.setattr(
        "inference_x.routing.variant_selector.estimate_weight_gib",
        lambda path, q=None: 14.0 if q is None else 2.0,
    )

    router = _build_router(str(config_dir), "qwen")
    from inference_x.schemas.chat import ChatCompletionRequest, ChatMessage

    routed = router.select(
        ChatCompletionRequest(model="unregistered-client-value", messages=[ChatMessage(role="user", content="hi")])
    )
    assert routed == "qwen-awq"
