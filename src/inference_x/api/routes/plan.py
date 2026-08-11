from typing import Annotated

from fastapi import APIRouter, Depends

from inference_x.api.deps import get_registry, get_vram_tier
from inference_x.schemas.diagnostics import PlanEntry, PlanResponse
from inference_x.services.model_service import ModelRegistry
from inference_x.utils.vllm_pool_config import (
    apply_tier_knobs,
    estimate_engine_footprint_gib,
    estimate_kv_cache_gib,
    estimate_weight_gib,
    probe_gpu_memory_gib,
    scale_model_config,
)
from inference_x.utils.vram_tiers import VramTier

router = APIRouter()


@router.get("/plan", response_model=PlanResponse, summary="Per-model VRAM sizing plan")
def plan(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    tier: Annotated[VramTier | None, Depends(get_vram_tier)],
) -> PlanResponse:
    """Report, for every registered model, the sizing the engine construction path would
    compute if that model were loaded now (add-plan-doctor design.md D1/D2). Read-only: no
    engine is started, stopped, or reconfigured to answer this request.
    """
    free_gib, total_gib = probe_gpu_memory_gib()
    total_vram = total_gib if total_gib is not None else 8.0

    entries: list[PlanEntry] = []
    for model_entry in registry.all():
        config = apply_tier_knobs(model_entry.model_dump(), tier)
        model_path = str(config["model_path"])
        quantization = config.get("quantization")
        max_model_len = int(config.get("max_model_len") or 2048)

        utilization: float | None = None
        try:
            scaled = scale_model_config(
                config, total_vram_gib=total_vram, free_vram_gib=free_gib
            )
            utilization = scaled["gpu_memory_utilization"]
        except ValueError:
            # Does not fit at the probed VRAM — GET /v1/doctor reports why;
            # plan reports no fabricated utilization for it (D2's never-fabricate rule).
            utilization = None

        entries.append(
            PlanEntry(
                model=model_entry.name,
                estimated_weight_gib=round(
                    estimate_weight_gib(model_path, quantization), 3
                ),
                estimated_kv_cache_gib=round(
                    estimate_kv_cache_gib(model_path, max_model_len), 3
                ),
                estimated_footprint_gib=round(
                    estimate_engine_footprint_gib(
                        model_path, max_model_len, quantization
                    ),
                    3,
                ),
                gpu_memory_utilization=utilization,
                block_size=config.get("block_size"),
                kv_cache_dtype=config.get("kv_cache_dtype"),
                enable_prefix_caching=config.get("enable_prefix_caching"),
                max_num_seqs=config.get("max_num_seqs"),
                max_num_batched_tokens=config.get("max_num_batched_tokens"),
            )
        )
    return PlanResponse(models=entries)
