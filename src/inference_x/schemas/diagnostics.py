"""Response schemas for the read-only preflight endpoints (Phase C, C6 — add-plan-doctor).

`GET /v1/plan` and `GET /v1/doctor` never start an engine (design.md D5); every field here is
computed by calling `utils/vllm_pool_config.py`'s existing sizing/fit functions directly, not by
re-deriving them.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from inference_x.schemas.chat import ManifestHardware


class PlanEntry(BaseModel):
    """Sizing report for one registered model, as the engine construction path would compute it."""

    model: str
    estimated_weight_gib: float
    estimated_kv_cache_gib: float
    estimated_footprint_gib: float
    gpu_memory_utilization: Optional[float] = None
    block_size: Optional[int] = None
    kv_cache_dtype: Optional[str] = None
    enable_prefix_caching: Optional[bool] = None
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None


class PlanResponse(BaseModel):
    """`GET /v1/plan` response: one entry per registered model."""

    models: list[PlanEntry]


class DoctorModelFit(BaseModel):
    """Whether one registered model fits on the currently probed VRAM."""

    model: str
    fits: bool
    reason: Optional[str] = None


class DoctorResponse(BaseModel):
    """`GET /v1/doctor` response: environment profile plus per-model fit readiness."""

    hardware: ManifestHardware
    vram_free_gib: Optional[float] = None
    vram_total_gib: Optional[float] = None
    models: list[DoctorModelFit]
