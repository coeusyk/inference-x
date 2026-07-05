#!/usr/bin/env python3
"""InferenceX model advisor CLI.

Reads stored benchmark results from benchmarks/results/ and prints a ranked
recommendation table based on the current hardware profile.

Usage:
    uv run python scripts/advise.py
    make advise
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from inference_x.benchmarks.advisor import VRAM_SAFETY_BUFFER_GB, ModelAdvisor
from inference_x.benchmarks.hardware import profile_hardware
from inference_x.benchmarks.storage import DEFAULT_RESULTS_DIR, ResultStore
from inference_x.services.model_service import ModelRegistry

_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([bBmM])")


def _estimate_vram_gib(name: str, model_path: str) -> float | None:
    """Rough VRAM estimate from parameter count hints in name or model path."""
    for text in (name, model_path):
        match = _PARAM_RE.search(text)
        if not match:
            continue
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "m":
            params_b = value / 1000.0
        else:
            params_b = value
        return round(params_b * 3.0, 1)
    return None


def _static_vram_estimates(config_dir: str = "config") -> dict[str, float | None]:
    """Return per-model VRAM estimates from config or name/path heuristics."""
    path = Path(config_dir) / "models.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    estimates: dict[str, float | None] = {}
    for entry in data.get("models", []):
        model_name = entry.get("name", "")
        if not model_name:
            continue
        if "vram_required_gib" in entry:
            estimates[model_name] = float(entry["vram_required_gib"])
        else:
            estimates[model_name] = _estimate_vram_gib(
                model_name, entry.get("model_path", "")
            )
    return estimates


def _print_static_guidance(hardware, config_dir: str = "config") -> None:
    estimates = _static_vram_estimates(config_dir)
    models = ModelRegistry.from_config(config_dir).all()

    print("No benchmark data — showing static config estimates.")
    print("Run `make benchmark-all` for performance rankings on your hardware.\n")

    if hardware.has_gpu:
        print(f"Hardware: {hardware.gpu_name} ({hardware.vram_total_gb:.1f} GB)\n")
    else:
        print("Hardware: CPU only\n")

    print(f"{'Model':<18} {'Est. VRAM':<12} {'Fits?':<8} Note")
    for model in models:
        est = estimates.get(model.name)
        if est is None:
            est_str = "unknown"
            fits_str = "?"
            note = "VRAM estimate unavailable — run benchmark"
        else:
            est_str = f"~{est:.1f} GB"
            required = est + VRAM_SAFETY_BUFFER_GB
            if hardware.has_gpu:
                fits = required <= hardware.vram_total_gb
                fits_str = "YES" if fits else "NO"
                note = (
                    "Run benchmark to get tok/s ranking"
                    if fits
                    else "Exceeds available VRAM"
                )
            else:
                fits_str = "?"
                note = "Run benchmark to get tok/s ranking"
        print(f"{model.name:<18} {est_str:<12} {fits_str:<8} {note}")


def main() -> None:
    store = ResultStore()
    latest = store.latest_per_model(output_dir=DEFAULT_RESULTS_DIR)

    hardware = profile_hardware()

    if not latest:
        try:
            _print_static_guidance(hardware)
        except FileNotFoundError:
            print(
                f"No benchmark results found in {DEFAULT_RESULTS_DIR}/.\n"
                "Run benchmarks first:\n"
                "  make benchmark MODEL=qwen2.5-0.5b\n"
                "  make benchmark-all",
                file=sys.stderr,
            )
            sys.exit(1)
        return

    try:
        registry = ModelRegistry.from_config("config")
        model_max_lens = {m.name: m.max_model_len for m in registry.all()}
    except FileNotFoundError:
        model_max_lens = None
    advisor = ModelAdvisor()
    report = advisor.rank(
        hardware,
        list(latest.values()),
        model_max_lens=model_max_lens,
    )
    ranked = report.ranked

    if report.warnings:
        print("Warnings:", file=sys.stderr)
        for warning in report.warnings:
            print(f"  WARNING: {warning}", file=sys.stderr)
        print(file=sys.stderr)

    print("=== InferenceX Model Advisor ===\n")
    print(f"Hardware: {hardware.gpu_name or 'CPU only'}")
    if hardware.has_gpu:
        print(
            f"  VRAM: {hardware.vram_free_gb:.1f} GB free / "
            f"{hardware.vram_total_gb:.1f} GB total"
        )
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
