"""Pydantic schemas for benchmark results, hardware profiles, and advisor output."""
from __future__ import annotations

from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, computed_field


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
    # Accept the deprecated ``peak_vram_delta_gb`` alias so historical result JSON
    # loads without migration; the canonical field wins on conflict (DEC-057).
    model_config = ConfigDict(populate_by_name=True)

    model_name: str
    suite_version: str
    timestamp: str  # ISO 8601
    concurrency: int = 1
    prompt_results: list[PromptResult] = Field(default_factory=list)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    mean_throughput_tps: float = 0.0
    # Device-wide occupied VRAM (total − min(free)); GiB. `peak_vram_delta_gb`
    # is a deprecated read alias retained through Phase A (DEC-057). Removal of
    # the alias requires a future ADR.
    vram_device_occupied_gib: float = Field(
        default=0.0,
        validation_alias=AliasChoices(
            "vram_device_occupied_gib", "peak_vram_delta_gb"
        ),
    )
    hardware: Optional[HardwareProfile] = None
    max_model_len: Optional[int] = None
    vram_budget_exceeded: bool = False
    vram_budget_warning: Optional[str] = None


class AdvisorResult(BaseModel):
    # Accept the deprecated ``vram_gb`` alias on input and keep emitting it on
    # output through Phase A; the canonical field wins on conflict (DEC-057).
    model_config = ConfigDict(populate_by_name=True)

    model_name: str
    score: float
    viable: bool
    throughput_tps: float
    ttft_ms: float
    vram_device_occupied_gib: float = Field(
        validation_alias=AliasChoices("vram_device_occupied_gib", "vram_gb"),
    )
    recommendation_str: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def vram_gb(self) -> float:
        """Deprecated alias for ``vram_device_occupied_gib`` (DEC-057).

        Retained in the serialised payload through Phase A for wire
        compatibility; removal requires a future ADR.
        """
        return self.vram_device_occupied_gib


class AdvisorReport(BaseModel):
    ranked: list[AdvisorResult]
    warnings: list[str] = Field(default_factory=list)
