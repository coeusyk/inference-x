from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ModelEntry(BaseModel):
    """Metadata for one configured model, as loaded from models.yaml."""

    name: str
    engine: Literal["vllm"] = "vllm"
    model_path: str
    gpu_memory_utilization: Optional[float | Literal["auto"]] = "auto"
    max_model_len: Optional[int] = Field(default=None, ge=1)
    max_num_seqs: Optional[int] = Field(
        default=None,
        ge=1,
        description="vLLM concurrent sequence cap (lower for hybrid/Mamba models on tight VRAM)",
    )
    quantization: Optional[str] = None
    gated: bool = False
    instruction_tuned: bool = True
    max_completion_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Per-request generation cap; overrides client max_tokens when set",
    )
    repetition_penalty: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="vLLM repetition_penalty; applied when instruction_tuned is false",
    )

    @field_validator("gpu_memory_utilization")
    @classmethod
    def _validate_gpu_memory_utilization(
        cls, value: Optional[float | Literal["auto"]]
    ) -> Optional[float | Literal["auto"]]:
        if value is None or value == "auto":
            return value
        if not 0.0 <= value <= 1.0:
            raise ValueError("gpu_memory_utilization must be between 0.0 and 1.0")
        return value


class ModelList(BaseModel):
    """OpenAI-compatible /v1/models response."""

    object: Literal["list"] = "list"
    data: list[ModelObject]


class ModelObject(BaseModel):
    """Single entry in the /v1/models response."""

    id: str
    object: Literal["model"] = "model"
    owned_by: str = "inferencex"
    quantization: Optional[str] = None
    max_model_len: Optional[int] = None
    estimated_weights_gib: Optional[float] = None
