from typing import Annotated

from fastapi import APIRouter, Depends

from inference_x.api.deps import get_registry
from inference_x.schemas.model import ModelList, ModelObject
from inference_x.services.model_service import ModelRegistry

router = APIRouter()


@router.get("/models", response_model=ModelList)
def list_models(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> ModelList:
    """Return all registered models in OpenAI-compatible format."""
    return ModelList(
        data=[ModelObject(id=entry.name) for entry in registry.all()]
    )
