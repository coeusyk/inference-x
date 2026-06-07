from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ModelEntry(BaseModel):
    """Metadata for one configured model, as loaded from models.yaml."""

    name: str
    engine: Literal["vllm"] = "vllm"
    model_path: str
    gpu_memory_utilization: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_model_len: Optional[int] = Field(default=None, ge=1)
    quantization: Optional[str] = None


class ModelList(BaseModel):
    """OpenAI-compatible /v1/models response."""

    object: Literal["list"] = "list"
    data: list[ModelObject]


class ModelObject(BaseModel):
    """Single entry in the /v1/models response."""

    id: str
    object: Literal["model"] = "model"
    owned_by: str = "inferencex"
