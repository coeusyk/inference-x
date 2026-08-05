"""Model advisor: scores and ranks benchmark results against hardware profile.

Scoring weights (uncalibrated editorial preference, not a reasoned inference from
data — they are not tuned against any outcome and may change; DEC-056):
  4/9 throughput      — relative to the fastest viable model in the report
  1/3 TTFT            — inverted (lower is better), relative to the report
  2/9 VRAM headroom   — remaining capacity after load, clamped 0–1

The three weights sum to 1 and preserve the prior 0.40 : 0.30 : 0.20 ratio; the
deleted `quant_score` placeholder contributed a constant additive floor and is gone.

`viable` is the sole viability signal. A model whose VRAM footprint plus safety
buffer exceeds hardware.vram_total_gb is hard-gated to score = 0.0, viable = False;
a viable-but-worst model may also be 0.0 — the two are distinguished only by
`viable`, never by the score value.

`score` is a within-report ordinal used only to rank viable models produced from
the same benchmark suite. It is NOT portable across benchmark suites, NOT portable
across hardware, NOT portable across future weighting changes, and NOT an absolute
quality metric: normalisation is relative to the report, so a single result yields a
zero TTFT component and adding a model can change the others' scores.

CPU-only hardware (has_gpu=False) skips the VRAM gate and sets vram_score = 1.0.
"""
from __future__ import annotations

import statistics

from inference_x.benchmarks.schemas import (
    AdvisorReport,
    AdvisorResult,
    BenchmarkResult,
    HardwareProfile,
)

VRAM_SAFETY_BUFFER_GB = 0.5  # reserve for driver overhead and system processes


def _safe_div(a: float, b: float, fallback: float = 0.0) -> float:
    return a / b if b > 0 else fallback


def _hardware_matches(saved: HardwareProfile, current: HardwareProfile) -> bool:
    saved_name = (saved.gpu_name or "").lower().strip()
    current_name = (current.gpu_name or "").lower().strip()
    if saved_name and current_name and saved_name != current_name:
        return False
    if abs(saved.vram_total_gb - current.vram_total_gb) > 0.5:
        return False
    return True


def _format_gpu_label(hw: HardwareProfile) -> str:
    if hw.gpu_name:
        return f"{hw.gpu_name} ({hw.vram_total_gb:.1f} GB)"
    return "CPU only"


def _warm_ttft_ms(result: BenchmarkResult) -> float:
    warm_results = (
        result.prompt_results[1:]
        if len(result.prompt_results) > 1
        else result.prompt_results
    )
    return statistics.mean(p.ttft_ms for p in warm_results) if warm_results else 0.0


class ModelAdvisor:
    """Ranks benchmark results against a hardware profile."""

    def rank(
        self,
        hardware: HardwareProfile,
        results: list[BenchmarkResult],
        *,
        model_max_lens: dict[str, int | None] | None = None,
    ) -> AdvisorReport:
        if not results:
            return AdvisorReport(ranked=[], warnings=[])

        warnings: list[str] = []
        eligible: list[BenchmarkResult] = []

        def _maybe_warn_max_model_len(result: BenchmarkResult) -> None:
            if model_max_lens is None:
                return
            current_max_model_len = model_max_lens.get(result.model_name)
            if (
                result.max_model_len is not None
                and current_max_model_len is not None
                and result.max_model_len != current_max_model_len
            ):
                warnings.append(
                    f"{result.model_name}: benchmark used max_model_len="
                    f"{result.max_model_len}, current config is {current_max_model_len}"
                    f" — re-run `make benchmark MODEL={result.model_name}` to refresh."
                )

        for result in results:
            if result.hardware is None:
                warnings.append(
                    f"No hardware recorded for {result.model_name} — "
                    f"re-run `make benchmark MODEL={result.model_name}`"
                )
                eligible.append(result)
                _maybe_warn_max_model_len(result)
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
            _maybe_warn_max_model_len(result)

        if not eligible:
            return AdvisorReport(ranked=[], warnings=warnings)

        max_throughput = max(r.mean_throughput_tps for r in eligible)
        max_ttft = max(_warm_ttft_ms(r) for r in eligible)

        advisor_results: list[AdvisorResult] = []

        for result in eligible:
            throughput = result.mean_throughput_tps
            ttft = _warm_ttft_ms(result)
            vram_used = result.vram_device_occupied_gib

            if hardware.has_gpu:
                vram_required = vram_used + VRAM_SAFETY_BUFFER_GB
                viable_by_vram = vram_required <= hardware.vram_total_gb
            else:
                viable_by_vram = True

            if not viable_by_vram:
                score = 0.0
                viable = False
                rec = (
                    f"{result.model_name} — requires {vram_used:.2f} GB + "
                    f"{VRAM_SAFETY_BUFFER_GB:.1f} GB buffer = {vram_required:.2f} GB, "
                    f"only {hardware.vram_total_gb:.1f} GB total. Skip."
                )
            else:
                viable = True

                # Throughput component (higher is better, normalized 0–1)
                throughput_score = _safe_div(throughput, max_throughput)

                # TTFT component (lower is better, inverted, normalized 0–1)
                ttft_score = 1.0 - _safe_div(ttft, max_ttft)

                # VRAM headroom component (remaining capacity after load, clamped 0–1)
                if not hardware.has_gpu:
                    vram_score = 1.0
                else:
                    vram_required = vram_used + VRAM_SAFETY_BUFFER_GB
                    vram_score = max(
                        0.0,
                        min(
                            1.0,
                            (hardware.vram_total_gb - vram_required)
                            / hardware.vram_total_gb,
                        ),
                    )

                # Measured-only composition (DEC-056). Exact fractions preserve the
                # prior 0.40 : 0.30 : 0.20 ratio; no constant quantization floor.
                score = (
                    (4 / 9) * throughput_score
                    + (1 / 3) * ttft_score
                    + (2 / 9) * vram_score
                ) * 100.0

                rec = (
                    f"{result.model_name} — {throughput:.0f} tok/s, "
                    f"{ttft:.0f}ms TTFT, {vram_used:.2f} GB VRAM"
                )

            advisor_results.append(
                AdvisorResult(
                    model_name=result.model_name,
                    score=round(score, 2),
                    viable=viable,
                    throughput_tps=throughput,
                    ttft_ms=ttft,
                    vram_device_occupied_gib=vram_used,
                    recommendation_str=rec,
                )
            )

        # Sort by score descending (viable first implicitly since non-viable = 0)
        advisor_results.sort(key=lambda r: r.score, reverse=True)
        return AdvisorReport(ranked=advisor_results, warnings=warnings)
