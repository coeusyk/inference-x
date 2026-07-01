"""VRAM tier resolution: map probed GPU VRAM to a named capacity tier.

Tiers are declarative data (config/vram_tiers.yaml) describing the model/context/
concurrency envelope InferenceX targets on a given class of GPU (see the file's
comments for the reasoning behind each tier's numbers). This module only resolves
*which* tier applies to the current machine; it does not enforce tier limits — that
is the admission controller's job (Phase 2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VramTier:
    """One VRAM capacity tier from config/vram_tiers.yaml."""

    name: str
    min_vram_gb: float
    description: str
    gpu_memory_utilization_ceiling: float
    max_model_len_cap: int
    max_num_seqs: int
    block_size: int
    kv_cache_dtype: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VramTier":
        return cls(
            name=str(data["name"]),
            min_vram_gb=float(data["min_vram_gb"]),
            description=str(data.get("description", "")).strip(),
            gpu_memory_utilization_ceiling=float(data["gpu_memory_utilization_ceiling"]),
            max_model_len_cap=int(data["max_model_len_cap"]),
            max_num_seqs=int(data["max_num_seqs"]),
            block_size=int(data.get("block_size", 16)),
            kv_cache_dtype=str(data.get("kv_cache_dtype", "auto")),
        )


def _tiers_path(config_dir: str) -> Path:
    return Path(config_dir) / "vram_tiers.yaml"


@lru_cache(maxsize=8)
def load_tiers(config_dir: str = "config") -> tuple[VramTier, ...]:
    """Load and sort tiers (ascending by min_vram_gb) from *config_dir*/vram_tiers.yaml.

    Raises:
        FileNotFoundError: the tiers file does not exist.
        ValueError: the file has no tiers defined.
    """
    path = _tiers_path(config_dir)
    if not path.is_file():
        raise FileNotFoundError(f"VRAM tiers config not found: {path}")

    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    raw_tiers = data.get("tiers") or []
    if not raw_tiers:
        raise ValueError(f"No tiers defined in {path}")

    tiers = [VramTier.from_dict(t) for t in raw_tiers]
    return tuple(sorted(tiers, key=lambda t: t.min_vram_gb))


def resolve_tier(vram_total_gb: float, tiers: tuple[VramTier, ...] | None = None) -> VramTier:
    """Return the highest tier whose min_vram_gb floor the GPU clears.

    Falls back to the lowest (most conservative) tier — and logs a warning — when
    *vram_total_gb* is below every tier's floor (e.g. VRAM probing failed and
    returned 0, or an unusually small GPU). Never guesses a higher tier than the
    evidence supports.
    """
    if tiers is None:
        tiers = load_tiers()
    if not tiers:
        raise ValueError("No VRAM tiers available to resolve against")

    matched = tiers[0]
    for tier in tiers:
        if vram_total_gb >= tier.min_vram_gb:
            matched = tier
        else:
            break

    if vram_total_gb < tiers[0].min_vram_gb:
        logger.warning(
            "Probed VRAM (%.1f GiB) is below the lowest tier floor (%s, >= %.0f GiB); "
            "using %s anyway as the most conservative option.",
            vram_total_gb,
            tiers[0].name,
            tiers[0].min_vram_gb,
            tiers[0].name,
        )
    return matched
