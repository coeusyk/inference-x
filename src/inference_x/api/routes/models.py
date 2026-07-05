from typing import Annotated

from fastapi import APIRouter, Depends

from inference_x.api.deps import get_registry
from inference_x.schemas.model import ModelList, ModelObject
from inference_x.services.model_service import ModelRegistry
from inference_x.utils.vllm_pool_config import estimate_weight_gib

router = APIRouter()


@router.get("/models", response_model=ModelList)
def list_models(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> ModelList:
    """Return all registered models in OpenAI-compatible format.

    Extends the OpenAI schema (additively) with quantization/VRAM fields so
    clients can pick a variant that fits their tier without a separate call.
    """
    return ModelList(
        data=[
            ModelObject(
                id=entry.name,
                quantization=entry.quantization,
                max_model_len=entry.max_model_len,
                estimated_weights_gib=round(
                    estimate_weight_gib(entry.model_path, entry.quantization), 3
                ),
            )
            for entry in registry.all()
        ]
    )
