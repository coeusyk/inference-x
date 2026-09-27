from typing import Annotated

from fastapi import APIRouter, Depends

from inference_x.api.deps import get_registry, get_vram_tier
from inference_x.schemas.model import ModelList, ModelObject
from inference_x.services.model_service import ModelRegistry
from inference_x.utils.vllm_pool_config import estimate_weight_gib
from inference_x.utils.vram_tiers import VramTier, effective_context

router = APIRouter()


@router.get("/models", response_model=ModelList)
def list_models(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    tier: Annotated[VramTier | None, Depends(get_vram_tier)],
) -> ModelList:
    """Return all registered models in OpenAI-compatible format.

    Extends the OpenAI schema (additively) with quantization/VRAM fields so
    clients can pick a variant that fits their tier without a separate call,
    and with the effective context window admission enforces (DEC-064).
    """
    data = []
    for entry in registry.all():
        env = (
            effective_context(tier, entry.max_model_len, entry.max_num_seqs)
            if tier is not None
            else None
        )
        data.append(
            ModelObject(
                id=entry.name,
                quantization=entry.quantization,
                max_model_len=entry.max_model_len,
                estimated_weights_gib=round(
                    estimate_weight_gib(entry.model_path, entry.quantization), 3
                ),
                context_window=env.max_model_len if env else None,
                max_num_seqs=env.max_num_seqs if env else None,
                context_composed=env.composed if env else None,
                context_tier_limited=env.tier_limited if env else None,
            )
        )
    return ModelList(data=data)
