"""The one engine construction path (DEC-047 exit criterion 3; add-llama-cpp-backend D2).

A plain dispatch over the two registered backends, not a plugin mechanism.
Each branch keeps its own sizing: vLLM's tier knobs and VRAM fit check do not
apply to llama.cpp, which fits layers itself.
"""

from __future__ import annotations

import logging

from inference_x.engines.base import BaseEngine
from inference_x.schemas.model import ModelEntry
from inference_x.utils.vram_tiers import VramTier

logger = logging.getLogger(__name__)


def create_engine(
    entry: ModelEntry,
    tier: VramTier | None,
    free_vram_gib: float | None,
    total_vram_gib: float | None,
) -> BaseEngine:
    logger.info("Loading %s engine for model=%s", entry.engine, entry.name)
    if entry.engine == "llama_cpp":
        from inference_x.engines.llama_cpp_engine import LlamaCppEngine

        return LlamaCppEngine(entry.model_dump())

    from inference_x.engines.vllm_engine import VLLMEngine
    from inference_x.utils.vllm_pool_config import apply_tier_knobs, validate_model_fits

    model_config = apply_tier_knobs(entry.model_dump(), tier)
    total = total_vram_gib if total_vram_gib is not None else 8.0
    validate_model_fits(model_config, total_vram_gib=total, free_vram_gib=free_vram_gib)
    return VLLMEngine(model_config, free_vram_gib=free_vram_gib, total_vram_gib=total_vram_gib)
