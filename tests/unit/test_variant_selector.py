"""Unit tests for routing/variant_selector.py (add-model-variant-routing)."""
from __future__ import annotations

import pytest

from inference_x.routing import variant_selector
from inference_x.routing.variant_selector import NoVariantFitsError, select_variant
from inference_x.schemas.model import ModelEntry
from inference_x.services.model_service import ModelRegistry
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
def fixed_weights(monkeypatch):
    """Deterministic weight-size table keyed by (model_path, quantization)."""

    sizes = {
        ("org/qwen-bf16", None): 14.0,
        ("org/qwen-int8", "int8"): 7.0,
        ("org/qwen-awq", "awq"): 4.0,
    }

    def fake_estimate(model_path: str, quantization: str | None = None) -> float:
        return sizes[(model_path, quantization)]

    monkeypatch.setattr(variant_selector, "estimate_weight_gib", fake_estimate)


def _family_registry() -> ModelRegistry:
    return ModelRegistry(
        [
            ModelEntry(name="qwen-bf16", model_path="org/qwen-bf16", family="qwen"),
            ModelEntry(
                name="qwen-int8", model_path="org/qwen-int8", family="qwen", quantization="int8"
            ),
            ModelEntry(
                name="qwen-awq", model_path="org/qwen-awq", family="qwen", quantization="awq"
            ),
        ]
    )


class TestSelectVariant:
    def test_selects_highest_precision_that_fits(self, fixed_weights):
        registry = _family_registry()
        # 8 GiB budget * 0.9 ceiling = 7.2 GiB -> bf16 (14) doesn't fit, int8 (7) does.
        name = select_variant("qwen", registry, _tier(), available_vram_gib=8.0)
        assert name == "qwen-int8"

    def test_selects_bf16_when_everything_fits(self, fixed_weights):
        registry = _family_registry()
        name = select_variant("qwen", registry, _tier(), available_vram_gib=20.0)
        assert name == "qwen-bf16"

    def test_falls_back_to_lowest_precision_when_only_it_fits(self, fixed_weights):
        registry = _family_registry()
        # 5 GiB * 0.9 = 4.5 GiB -> only awq (4.0) fits.
        name = select_variant("qwen", registry, _tier(), available_vram_gib=5.0)
        assert name == "qwen-awq"

    def test_raises_when_nothing_fits(self, fixed_weights):
        registry = _family_registry()
        with pytest.raises(NoVariantFitsError, match="qwen-awq"):
            select_variant("qwen", registry, _tier(), available_vram_gib=1.0)

    def test_raises_when_family_unknown(self):
        registry = ModelRegistry([ModelEntry(name="solo", model_path="org/solo")])
        with pytest.raises(NoVariantFitsError, match="No model variants registered"):
            select_variant("nonexistent", registry, _tier(), available_vram_gib=100.0)

    def test_single_variant_family_returns_it_unconditionally_when_it_fits(self, monkeypatch):
        monkeypatch.setattr(variant_selector, "estimate_weight_gib", lambda path, q=None: 1.0)
        registry = ModelRegistry([ModelEntry(name="solo", model_path="org/solo")])
        assert select_variant("solo", registry, _tier(), available_vram_gib=8.0) == "solo"

    def test_single_variant_family_raises_when_it_does_not_fit(self, monkeypatch):
        monkeypatch.setattr(variant_selector, "estimate_weight_gib", lambda path, q=None: 100.0)
        registry = ModelRegistry([ModelEntry(name="solo", model_path="org/solo")])
        with pytest.raises(NoVariantFitsError):
            select_variant("solo", registry, _tier(), available_vram_gib=8.0)
