"""Benchmark runner: measures throughput, TTFT, and latency for a model.

The runner POSTs each prompt from the suite to the inference server with
stream=True, measuring time-to-first-token (TTFT) and total latency.  It
does not require vLLM or GPU access — it communicates over HTTP.
"""
from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from inference_x.benchmarks.hardware import profile_hardware
from inference_x.benchmarks.schemas import BenchmarkResult, HardwareProfile, PromptResult
from inference_x.services.model_service import ModelRegistry
from inference_x.utils.vllm_pool_config import estimate_engine_footprint_gib

_VRAM_BUDGET_SLACK_GB = 0.5


def _load_suite(suite_path: str) -> tuple[str, list[dict]]:
    """Return (suite_version, prompts_list) from a suite JSON file."""
    data = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    return data["suite_version"], data["prompts"]


def _run_prompt_stream(
    client: httpx.Client,
    base_url: str,
    model_name: str,
    prompt_text: str,
    prompt_label: str,
) -> PromptResult:
    """POST one prompt with stream=True and measure timing metrics."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 256,
        "stream": True,
        # Ask for the terminal usage event (DEC-049). Without it the server
        # reports no token count, and this runner would have nothing truthful
        # to measure throughput from.
        "stream_options": {"include_usage": True},
    }

    tokens_generated = 0
    ttft_ms: Optional[float] = None
    t_start = time.perf_counter()

    with client.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=httpx.Timeout(10.0, read=120.0),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        # Engine-accounted count from the terminal usage event.
                        tokens_generated = int(usage.get("completion_tokens", 0))
                        continue
                    delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content and ttft_ms is None:
                        # TTFT still comes from the first content event.
                        ttft_ms = (time.perf_counter() - t_start) * 1000
                except (json.JSONDecodeError, IndexError):
                    continue

    total_latency_ms = (time.perf_counter() - t_start) * 1000
    if ttft_ms is None:
        ttft_ms = total_latency_ms
    tokens_per_sec = (tokens_generated / total_latency_ms * 1000) if total_latency_ms > 0 else 0.0

    return PromptResult(
        prompt_label=prompt_label,
        tokens_generated=tokens_generated,
        ttft_ms=round(ttft_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
        tokens_per_sec=round(tokens_per_sec, 2),
    )


def _percentile(data: list[float], p: int) -> float:
    """p-th percentile via linear interpolation (p in 1..99). Caller passes 50/95/99."""
    if len(data) < 2:
        return data[0] if data else 0.0
    return statistics.quantiles(data, n=100, method="inclusive")[p - 1]


def _check_vram_budget(
    model_name: str,
    peak_vram_delta_gb: float,
    model_path: str | None,
    max_model_len: int | None,
    quantization: str | None,
) -> tuple[bool, str | None]:
    """Compare a measured VRAM delta against the estimated engine footprint + slack."""
    if model_path is None:
        return False, None
    budget_gb = (
        estimate_engine_footprint_gib(model_path, max_model_len or 2048, quantization)
        + _VRAM_BUDGET_SLACK_GB
    )
    if peak_vram_delta_gb <= budget_gb:
        return False, None
    return True, (
        f"{model_name}: measured VRAM delta {peak_vram_delta_gb:.2f} GB exceeds "
        f"estimated budget {budget_gb:.2f} GB (weights + KV + overhead + "
        f"{_VRAM_BUDGET_SLACK_GB:.1f} GB slack)."
    )


def _peak_vram_footprint_gb(before: HardwareProfile, after: HardwareProfile) -> float:
    """VRAM footprint from GPU snapshots (works when model is already loaded).

    Uses total minus minimum free VRAM seen before/after the run. A simple
    free-before minus free-after delta is ~0 when the server already holds weights.
    """
    if not before.has_gpu:
        return 0.0
    min_free = min(before.vram_free_gb, after.vram_free_gb)
    return max(0.0, before.vram_total_gb - min_free)


class BenchmarkRunner:
    """Runs a fixed prompt suite against a model and returns a BenchmarkResult."""

    def run(
        self,
        model_name: str,
        suite_path: str,
        base_url: str = "http://127.0.0.1:8000",
        concurrency: int = 1,
        config_dir: str = "config",
    ) -> BenchmarkResult:
        base_url = base_url.rstrip("/")
        suite_version, prompts = _load_suite(suite_path)

        max_model_len: int | None = None
        model_path: str | None = None
        quantization: str | None = None
        try:
            entry = ModelRegistry.from_config(config_dir).get(model_name)
            max_model_len = entry.max_model_len
            model_path = entry.model_path
            quantization = entry.quantization
        except (FileNotFoundError, ValueError):
            pass

        hardware_before = profile_hardware()
        prompt_results: list[PromptResult] = []

        with httpx.Client() as client:
            for prompt in prompts:
                result = _run_prompt_stream(
                    client=client,
                    base_url=base_url,
                    model_name=model_name,
                    prompt_text=prompt["text"],
                    prompt_label=prompt["label"],
                )
                prompt_results.append(result)

        hardware_after = profile_hardware()

        latencies = [r.total_latency_ms for r in prompt_results]
        throughputs = [r.tokens_per_sec for r in prompt_results]

        peak_vram_delta = _peak_vram_footprint_gb(hardware_before, hardware_after)
        vram_budget_exceeded, vram_budget_warning = _check_vram_budget(
            model_name, peak_vram_delta, model_path, max_model_len, quantization
        )

        return BenchmarkResult(
            model_name=model_name,
            suite_version=suite_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            concurrency=concurrency,
            prompt_results=prompt_results,
            p50_latency_ms=round(_percentile(latencies, 50), 2),
            p95_latency_ms=round(_percentile(latencies, 95), 2),
            p99_latency_ms=round(_percentile(latencies, 99), 2),
            mean_throughput_tps=round(statistics.mean(throughputs) if throughputs else 0.0, 2),
            peak_vram_delta_gb=round(peak_vram_delta, 2),
            hardware=hardware_before,
            max_model_len=max_model_len,
            vram_budget_exceeded=vram_budget_exceeded,
            vram_budget_warning=vram_budget_warning,
        )
