"""Read-only observability route.

GET /v1/metrics — aggregated request metrics (latency/TTFT/tokens-per-sec)
plus a live VRAM breakdown (weights/KV-capacity per loaded model, free/total).
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from inference_x.api.deps import get_engine_pool, get_metrics_service, get_registry
from inference_x.engines.pool import EnginePool
from inference_x.schemas.metrics import MetricsResponse, ModelVramBreakdown, VramSummary
from inference_x.services.metrics_service import MetricsService
from inference_x.services.model_service import ModelRegistry
from inference_x.utils.vllm_pool_config import estimate_weight_gib, probe_gpu_memory_gib

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse, summary="Observability + VRAM snapshot")
def get_metrics(
    metrics_service: Annotated[MetricsService, Depends(get_metrics_service)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    pool: Annotated[EnginePool, Depends(get_engine_pool)],
) -> MetricsResponse:
    """Return request-level metrics plus a per-model VRAM breakdown for loaded models."""
    summary = metrics_service.summary()
    free_gib, total_gib = probe_gpu_memory_gib()

    models: list[ModelVramBreakdown] = []
    for name in pool.loaded_models():
        entry = registry.get(name)
        engine = pool.get(name)
        models.append(
            ModelVramBreakdown(
                name=name,
                quantization=entry.quantization,
                estimated_weights_gib=round(
                    estimate_weight_gib(entry.model_path, entry.quantization), 3
                ),
                kv_capacity_tokens=getattr(engine, "kv_capacity_tokens", None),
                max_model_len=entry.max_model_len,
            )
        )

    return MetricsResponse(
        total_requests=summary.total_requests,
        error_count=summary.error_count,
        avg_latency_ms=summary.avg_latency_ms,
        p95_latency_ms=summary.p95_latency_ms,
        avg_ttft_ms=summary.avg_ttft_ms,
        avg_tokens_per_sec=summary.avg_tokens_per_sec,
        vram=VramSummary(total_gib=total_gib, free_gib=free_gib, models=models),
    )
