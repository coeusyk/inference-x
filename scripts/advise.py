#!/usr/bin/env python3
"""InferenceX model advisor CLI.

Reads stored benchmark results from docs/benchmarks/ and prints a ranked
recommendation table based on the current hardware profile.

Usage:
    uv run python scripts/advise.py
    make advise
"""
from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from inference_x.benchmarks.advisor import ModelAdvisor
from inference_x.benchmarks.hardware import profile_hardware
from inference_x.benchmarks.storage import ResultStore


def main() -> None:
    store = ResultStore()
    latest = store.latest_per_model(output_dir="docs/benchmarks")

    if not latest:
        print(
            "No benchmark results found in docs/benchmarks/.\n"
            "Run benchmarks first:\n"
            "  make benchmark MODEL=qwen2.5-0.5b\n"
            "  make benchmark-all",
            file=sys.stderr,
        )
        sys.exit(1)

    hardware = profile_hardware()
    advisor = ModelAdvisor()
    report = advisor.rank(hardware, list(latest.values()))
    ranked = report.ranked

    if report.warnings:
        print("Warnings:", file=sys.stderr)
        for warning in report.warnings:
            print(f"  WARNING: {warning}", file=sys.stderr)
        print(file=sys.stderr)

    print("=== InferenceX Model Advisor ===\n")
    print(f"Hardware: {hardware.gpu_name or 'CPU only'}")
    if hardware.has_gpu:
        print(f"  VRAM: {hardware.vram_free_gb:.1f} GB free / {hardware.vram_total_gb:.1f} GB total")
    print(f"  CPU cores: {hardware.cpu_cores}  RAM: {hardware.ram_total_gb:.1f} GB\n")

    print(f"{'Rank':<5} {'Model':<25} {'Score':<8} {'Viable':<8} {'Recommendation'}")
    print("-" * 90)
    for i, result in enumerate(ranked, 1):
        viable_str = "YES" if result.viable else "NO"
        print(
            f"{i:<5} {result.model_name:<25} {result.score:<8.1f} {viable_str:<8} "
            f"{result.recommendation_str}"
        )


if __name__ == "__main__":
    main()
