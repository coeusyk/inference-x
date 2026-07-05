"""Unit tests for VRAM tier resolution (config/vram_tiers.yaml)."""
from __future__ import annotations

import pytest

from inference_x.utils.vram_tiers import VramTier, load_tiers, resolve_tier

_TIERS_YAML = """
tiers:
  - name: 6gb
    min_vram_gb: 0
    description: floor tier
    gpu_memory_utilization_ceiling: 0.90
    max_model_len_cap: 2048
    max_num_seqs: 4
    block_size: 16
    kv_cache_dtype: auto

  - name: 12gb
    min_vram_gb: 10
    description: mid tier
    gpu_memory_utilization_ceiling: 0.90
    max_model_len_cap: 4096
    max_num_seqs: 8
    block_size: 16
    kv_cache_dtype: auto

  - name: 24gb
    min_vram_gb: 20
    description: high tier
    gpu_memory_utilization_ceiling: 0.92
    max_model_len_cap: 8192
    max_num_seqs: 16
    block_size: 16
    kv_cache_dtype: auto
"""


@pytest.fixture
def tiers_config_dir(tmp_path):
    (tmp_path / "vram_tiers.yaml").write_text(_TIERS_YAML)
    return str(tmp_path)


def test_load_tiers_sorted_ascending(tiers_config_dir):
    load_tiers.cache_clear()
    tiers = load_tiers(tiers_config_dir)
    assert [t.name for t in tiers] == ["6gb", "12gb", "24gb"]


def test_load_tiers_missing_file_raises(tmp_path):
    load_tiers.cache_clear()
    with pytest.raises(FileNotFoundError):
        load_tiers(str(tmp_path))


def test_load_tiers_empty_raises(tmp_path):
    (tmp_path / "vram_tiers.yaml").write_text("tiers: []\n")
    load_tiers.cache_clear()
    with pytest.raises(ValueError, match="No tiers defined"):
        load_tiers(str(tmp_path))


@pytest.mark.parametrize(
    ("vram_total_gb", "expected_tier"),
    [
        (6.0, "6gb"),
        (8.0, "6gb"),
        (9.99, "6gb"),
        (10.0, "12gb"),
        (16.0, "12gb"),
        (19.99, "12gb"),
        (20.0, "24gb"),
        (24.0, "24gb"),
        (48.0, "24gb"),
    ],
)
def test_resolve_tier_matches_floor(tiers_config_dir, vram_total_gb, expected_tier):
    load_tiers.cache_clear()
    tiers = load_tiers(tiers_config_dir)
    resolved = resolve_tier(vram_total_gb, tiers)
    assert resolved.name == expected_tier


def test_resolve_tier_falls_back_to_lowest_when_below_every_floor(tiers_config_dir, monkeypatch):
    load_tiers.cache_clear()
    tiers = load_tiers(tiers_config_dir)
    # Shift the config so every tier has a positive floor, then probe below all of them.
    shifted = tuple(
        VramTier(
            name=t.name,
            min_vram_gb=t.min_vram_gb + 4,
            description=t.description,
            gpu_memory_utilization_ceiling=t.gpu_memory_utilization_ceiling,
            max_model_len_cap=t.max_model_len_cap,
            max_num_seqs=t.max_num_seqs,
            block_size=t.block_size,
            kv_cache_dtype=t.kv_cache_dtype,
        )
        for t in tiers
    )
    # Assert on the logger call directly rather than caplog: config/logging.yaml sets
    # propagate: false on the "inference_x" logger once any test imports
    # inference_x.api.main (which applies the dictConfig), so records never reach
    # caplog's root-attached handler even though the console handler does fire.
    from inference_x.utils import vram_tiers as vram_tiers_module

    warnings: list[str] = []
    monkeypatch.setattr(
        vram_tiers_module.logger, "warning", lambda msg, *args, **kw: warnings.append(msg % args)
    )
    resolved = resolve_tier(0.0, shifted)
    assert resolved.name == shifted[0].name
    assert any("below the lowest tier floor" in w for w in warnings)


def test_resolve_tier_no_tiers_raises():
    with pytest.raises(ValueError, match="No VRAM tiers"):
        resolve_tier(8.0, tiers=())


def test_real_vram_tiers_yaml_loads_and_resolves():
    """The shipped config/vram_tiers.yaml parses and resolves sensibly."""
    load_tiers.cache_clear()
    tiers = load_tiers("config")
    assert [t.name for t in tiers] == ["6gb", "12gb", "24gb"]
    assert resolve_tier(6.0, tiers).name == "6gb"
    assert resolve_tier(12.0, tiers).name == "12gb"
    assert resolve_tier(24.0, tiers).name == "24gb"


def test_real_vram_tiers_yaml_has_engine_knobs():
    """max_num_batched_tokens/enable_prefix_caching are wired for every shipped tier."""
    load_tiers.cache_clear()
    tiers = load_tiers("config")
    by_name = {t.name: t for t in tiers}
    assert by_name["6gb"].max_num_batched_tokens == 2048
    assert by_name["6gb"].enable_prefix_caching is False
    assert by_name["12gb"].max_num_batched_tokens == 4096
    assert by_name["12gb"].enable_prefix_caching is True
    assert by_name["24gb"].max_num_batched_tokens == 8192
    assert by_name["24gb"].enable_prefix_caching is True


def test_from_dict_defaults_new_knobs_when_absent(tiers_config_dir):
    """Old-style vram_tiers.yaml files without the new fields keep working."""
    load_tiers.cache_clear()
    tiers = load_tiers(tiers_config_dir)
    for tier in tiers:
        assert tier.max_num_batched_tokens is None
        assert tier.enable_prefix_caching is False


def test_from_dict_parses_new_knobs_when_present(tmp_path):
    (tmp_path / "vram_tiers.yaml").write_text(
        """
tiers:
  - name: 6gb
    min_vram_gb: 0
    description: floor tier
    gpu_memory_utilization_ceiling: 0.90
    max_model_len_cap: 2048
    max_num_seqs: 4
    block_size: 16
    kv_cache_dtype: auto
    max_num_batched_tokens: 2048
    enable_prefix_caching: false
"""
    )
    load_tiers.cache_clear()
    tiers = load_tiers(str(tmp_path))
    assert tiers[0].max_num_batched_tokens == 2048
    assert tiers[0].enable_prefix_caching is False
