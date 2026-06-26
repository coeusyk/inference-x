"""Pydantic schemas for benchmark results, hardware profiles, and advisor output."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class HardwareProfile(BaseModel):
    gpu_name: Optional[str] = None
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    cpu_cores: int = 1
    ram_total_gb: float = 0.0
    has_gpu: bool = False


class PromptResult(BaseModel):
    prompt_label: str
    tokens_generated: int
    ttft_ms: float
    total_latency_ms: float
    tokens_per_sec: float


class BenchmarkResult(BaseModel):
    model_name: str
    suite_version: str
    timestamp: str  # ISO 8601
    concurrency: int = 1
    prompt_results: list[PromptResult] = Field(default_factory=list)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    mean_throughput_tps: float = 0.0
    peak_vram_delta_gb: float = 0.0
    hardware: Optional[HardwareProfile] = None
    max_model_len: Optional[int] = None


class AdvisorResult(BaseModel):
    model_name: str
    score: float
    viable: bool
    throughput_tps: float
    ttft_ms: float
    vram_gb: float
    recommendation_str: str


class AdvisorReport(BaseModel):
    ranked: list[AdvisorResult]
    warnings: list[str] = Field(default_factory=list)
