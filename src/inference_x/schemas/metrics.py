"""Response schemas for GET /v1/metrics.

Two concerns, kept in one response since both answer "how is this deployment
doing right now": request-level metrics (latency/TTFT/tokens-per-sec, from
MetricsService) and VRAM-level metrics (weights/KV/free GiB per loaded model,
computed from probed hardware + the same estimators startup sizing uses).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ModelVramBreakdown(BaseModel):
    """VRAM accounting for one loaded model."""

    name: str
    quantization: Optional[str] = None
    estimated_weights_gib: float
    kv_capacity_tokens: Optional[int] = Field(
        default=None,
        description="Real post-load KV-pool capacity (num_gpu_blocks * block_size); "
        "None if the engine hasn't reported it (e.g. not yet loaded).",
    )
    max_model_len: Optional[int] = None


class VramSummary(BaseModel):
    """GPU-wide VRAM snapshot at the time of the request."""

    total_gib: Optional[float] = None
    free_gib: Optional[float] = None
    models: list[ModelVramBreakdown] = []


class MetricsResponse(BaseModel):
    """GET /v1/metrics response body."""

    total_requests: int
    error_count: int
    avg_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    avg_ttft_ms: Optional[float] = None
    avg_tokens_per_sec: Optional[float] = None
    vram: VramSummary
