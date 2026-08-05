"""Read-only benchmark routes.

GET /v1/benchmark/results  — returns stored BenchmarkResult list + hardware profile
GET /v1/benchmark/advise   — returns ModelAdvisor ranked output + hardware profile
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from inference_x.benchmarks.advisor import ModelAdvisor
from inference_x.benchmarks.hardware import profile_hardware
from inference_x.benchmarks.schemas import AdvisorResult, BenchmarkResult, HardwareProfile
from inference_x.benchmarks.storage import ResultStore
from inference_x.benchmarks.suite_identity import suite_version_of
from inference_x.core.settings import get_settings
from inference_x.services.model_service import ModelRegistry

router = APIRouter()

_store = ResultStore()
_advisor = ModelAdvisor()


class BenchmarkResultsResponse(BaseModel):
    results: list[BenchmarkResult]
    hardware: HardwareProfile


class BenchmarkAdviseResponse(BaseModel):
    ranked: list[AdvisorResult]
    hardware: HardwareProfile
    generated_at: str
    warnings: list[str] = []


@router.get("/results", response_model=BenchmarkResultsResponse)
def get_benchmark_results() -> BenchmarkResultsResponse:
    """Return all stored benchmark results and current hardware profile."""
    results = _store.all_results()
    hardware = profile_hardware()
    return BenchmarkResultsResponse(results=results, hardware=hardware)


@router.get("/advise", response_model=BenchmarkAdviseResponse)
def get_benchmark_advise() -> BenchmarkAdviseResponse:
    """Return a ranked advisor recommendation based on stored results.

    Only results matching the current suite version are ranked (DEC-055); an
    empty store and a suite mismatch are surfaced as distinct warnings.
    """
    expected_suite = suite_version_of()
    selection = _store.latest_per_model_for_suite(expected_suite)
    hardware = profile_hardware()

    if selection.status == "empty":
        return BenchmarkAdviseResponse(
            ranked=[],
            hardware=hardware,
            generated_at=datetime.now(timezone.utc).isoformat(),
            warnings=["No benchmark results exist. Run `make benchmark-all`."],
        )
    if selection.status == "suite_mismatch":
        return BenchmarkAdviseResponse(
            ranked=[],
            hardware=hardware,
            generated_at=datetime.now(timezone.utc).isoformat(),
            warnings=[
                "Benchmark results exist but none match the current suite version "
                f"{expected_suite}. Re-run `make benchmark-all` to refresh."
            ],
        )

    registry = ModelRegistry.from_config(get_settings().config_dir)
    model_max_lens = {m.name: m.max_model_len for m in registry.all()}
    report = _advisor.rank(
        hardware,
        list(selection.latest.values()),
        model_max_lens=model_max_lens,
    )
    return BenchmarkAdviseResponse(
        ranked=report.ranked,
        hardware=hardware,
        generated_at=datetime.now(timezone.utc).isoformat(),
        warnings=report.warnings,
    )
