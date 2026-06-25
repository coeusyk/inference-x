"""Model advisor: scores and ranks benchmark results against hardware profile.

Scoring weights:
  40% throughput  — most visible performance signal
  30% TTFT        — inverted (lower is better) — latency matters for interactivity
  20% VRAM headroom — model must fit in available free VRAM
  10% quantization  — placeholder (currently 1.0 for all models, reserved for INT8/FP8)

A model with peak_vram_delta_gb > hardware.vram_free_gb is hard-gated:
  score = 0, viable = False regardless of other metrics.
CPU-only hardware (has_gpu=False) skips the VRAM gate and sets vram_headroom = 1.0.
"""
from __future__ import annotations

from inference_x.benchmarks.schemas import (
    AdvisorReport,
    AdvisorResult,
    BenchmarkResult,
    HardwareProfile,
)


def _safe_div(a: float, b: float, fallback: float = 0.0) -> float:
    return a / b if b > 0 else fallback


def _hardware_matches(saved: HardwareProfile, current: HardwareProfile) -> bool:
    saved_name = (saved.gpu_name or "").lower()
    current_name = (current.gpu_name or "").lower()
    if saved_name and current_name:
        if saved_name not in current_name and current_name not in saved_name:
            return False
    elif saved_name != current_name:
        return False
    if abs(saved.vram_total_gb - current.vram_total_gb) > 0.5:
        return False
    return True


def _format_gpu_label(hw: HardwareProfile) -> str:
    if hw.gpu_name:
        return f"{hw.gpu_name} ({hw.vram_total_gb:.1f} GB)"
    return "CPU only"


class ModelAdvisor:
    """Ranks benchmark results against a hardware profile."""

    def rank(
        self,
        hardware: HardwareProfile,
        results: list[BenchmarkResult],
    ) -> AdvisorReport:
        if not results:
            return AdvisorReport(ranked=[], warnings=[])

        warnings: list[str] = []
        eligible: list[BenchmarkResult] = []

        for result in results:
            if result.hardware is None:
                warnings.append(
                    f"No hardware recorded for {result.model_name} — "
                    f"re-run `make benchmark MODEL={result.model_name}`"
                )
                eligible.append(result)
                continue
            if not _hardware_matches(result.hardware, hardware):
                saved = _format_gpu_label(result.hardware)
                current = _format_gpu_label(hardware)
                warnings.append(
                    f"Skipped {result.model_name} — benchmark was run on {saved}, "
                    f"current GPU is {current}. "
                    f"Re-run `make benchmark MODEL={result.model_name}` to get fresh results."
                )
                continue
            eligible.append(result)

        if not eligible:
            return AdvisorReport(ranked=[], warnings=warnings)

        max_throughput = max(r.mean_throughput_tps for r in eligible)
        max_ttft = max(
            (r.prompt_results[0].ttft_ms if r.prompt_results else 0.0) for r in eligible
        )

        advisor_results: list[AdvisorResult] = []

        for result in eligible:
            throughput = result.mean_throughput_tps
            ttft = result.prompt_results[0].ttft_ms if result.prompt_results else 0.0
            vram_used = result.peak_vram_delta_gb

            # Hard gate: if VRAM required exceeds free VRAM, mark not viable
            if hardware.has_gpu and vram_used > hardware.vram_free_gb:
                score = 0.0
                viable = False
                rec = (
                    f"{result.model_name} — requires {vram_used:.1f}GB VRAM, "
                    f"only {hardware.vram_free_gb:.1f}GB free. Skip."
                )
            else:
                viable = True

                # Throughput component (higher is better, normalized 0–1)
                throughput_score = _safe_div(throughput, max_throughput)

                # TTFT component (lower is better, inverted, normalized 0–1)
                ttft_score = 1.0 - _safe_div(ttft, max_ttft)

                # VRAM headroom component (free capacity after model load, clamped 0–1)
                if not hardware.has_gpu:
                    vram_score = 1.0
                else:
                    if hardware.vram_free_gb > 0:
                        vram_score = max(
                            0.0,
                            min(1.0, (hardware.vram_free_gb - vram_used) / hardware.vram_free_gb),
                        )
                    else:
                        vram_score = 1.0

                # Quantization placeholder (always 1.0 — no quantization data yet)
                quant_score = 1.0

                score = (
                    0.40 * throughput_score
                    + 0.30 * ttft_score
                    + 0.20 * vram_score
                    + 0.10 * quant_score
                ) * 100.0

                rec = (
                    f"{result.model_name} — {throughput:.0f} tok/s, "
                    f"{ttft:.0f}ms TTFT, {vram_used:.1f}GB VRAM"
                )

            advisor_results.append(
                AdvisorResult(
                    model_name=result.model_name,
                    score=round(score, 2),
                    viable=viable,
                    throughput_tps=throughput,
                    ttft_ms=ttft,
                    vram_gb=vram_used,
                    recommendation_str=rec,
                )
            )

        # Sort by score descending (viable first implicitly since non-viable = 0)
        advisor_results.sort(key=lambda r: r.score, reverse=True)
        return AdvisorReport(ranked=advisor_results, warnings=warnings)
