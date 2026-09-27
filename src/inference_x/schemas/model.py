from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ModelEntry(BaseModel):
    """Metadata for one configured model, as loaded from models.yaml."""

    name: str
    engine: Literal["vllm"] = "vllm"
    model_path: str
    family: Optional[str] = Field(
        default=None,
        description=(
            "Groups variants of the same logical model (e.g. bf16/int8/awq of "
            "qwen2.5-7b) for load-time selection via routing/variant_selector.py. "
            "Entries with no family are their own family of one."
        ),
    )
    gpu_memory_utilization: Optional[float | Literal["auto"]] = "auto"
    max_model_len: Optional[int] = Field(default=None, ge=1)
    max_num_seqs: Optional[int] = Field(
        default=None,
        ge=1,
        description="vLLM concurrent sequence cap (lower for hybrid/Mamba models on tight VRAM)",
    )
    max_num_batched_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "vLLM per-step token budget override; clamped to the resolved VRAM "
            "tier's ceiling by apply_tier_knobs(), never looser"
        ),
    )
    quantization: Optional[str] = None
    gated: bool = False
    instruction_tuned: bool = True
    max_completion_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Per-model output-token cap, resolved by admission: a larger requested "
            "max_tokens is clamped with a substituted warning (rejected under strict)"
        ),
    )
    aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Other names a client may send as `model` for this entry. They route to "
            "this entry; `resolved.model` and the manifest still report the canonical "
            "name (DEC-065)"
        ),
    )
    tool_call_parser: Optional[Literal["hermes"]] = Field(
        default=None,
        description=(
            "Backend tool-call parser this model's output format needs (vLLM "
            "--tool-call-parser name). Unset means the model does not support "
            "tool calling and tool requests are rejected (add-streaming-tool-calling D4)"
        ),
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
    # DEC-064: effective context ceiling / concurrency on the resolved tier, which
    # can differ from the configured max_model_len. None when no tier resolved.
    context_window: Optional[int] = None
    max_num_seqs: Optional[int] = None
    context_composed: Optional[bool] = None
    context_tier_limited: Optional[bool] = None
