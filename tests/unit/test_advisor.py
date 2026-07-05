"""Unit tests for ModelAdvisor scoring and ranking."""
from __future__ import annotations

import pytest

from inference_x.benchmarks.advisor import ModelAdvisor
from inference_x.benchmarks.schemas import (
    AdvisorResult,
    BenchmarkResult,
    HardwareProfile,
    PromptResult,
)


def _rank(advisor: ModelAdvisor, hw: HardwareProfile, results: list[BenchmarkResult]) -> list[AdvisorResult]:
    return advisor.rank(hw, results).ranked


def _make_result(
    model_name: str,
    throughput: float,
    ttft_ms: float,
    peak_vram_delta_gb: float,
    *,
    hardware: HardwareProfile | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        model_name=model_name,
        suite_version="test",
        timestamp="2026-06-08T12:00:00+00:00",
        concurrency=1,
        prompt_results=[
            PromptResult(
                prompt_label="q1",
                tokens_generated=20,
                ttft_ms=ttft_ms,
                total_latency_ms=800.0,
                tokens_per_sec=throughput,
            )
        ],
        p50_latency_ms=800.0,
        p95_latency_ms=850.0,
        p99_latency_ms=900.0,
        mean_throughput_tps=throughput,
        peak_vram_delta_gb=peak_vram_delta_gb,
        hardware=hardware,
    )


def _hw(vram_free: float, vram_total: float, has_gpu: bool = True) -> HardwareProfile:
    return HardwareProfile(
        gpu_name="Test GPU",
        vram_total_gb=vram_total,
        vram_free_gb=vram_free,
        cpu_cores=8,
        ram_total_gb=32.0,
        has_gpu=has_gpu,
    )


class TestModelAdvisorRanking:
    def test_higher_throughput_ranks_first(self):
        results = [
            _make_result("slow-model", throughput=10.0, ttft_ms=300.0, peak_vram_delta_gb=1.0),
            _make_result("fast-model", throughput=50.0, ttft_ms=100.0, peak_vram_delta_gb=2.0),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=6.0, vram_total=8.0), results)
        assert ranked[0].model_name == "fast-model"
        assert ranked[1].model_name == "slow-model"

    def test_vram_exceeded_marks_not_viable(self):
        results = [
            _make_result("big-model", throughput=60.0, ttft_ms=50.0, peak_vram_delta_gb=8.0),
            _make_result("small-model", throughput=20.0, ttft_ms=200.0, peak_vram_delta_gb=1.5),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=6.0, vram_total=8.0), results)

        big = next(r for r in ranked if r.model_name == "big-model")
        small = next(r for r in ranked if r.model_name == "small-model")

        assert big.viable is False
        assert big.score == 0.0
        assert small.viable is True
        assert small.score > 0.0

    def test_non_viable_model_ranks_last(self):
        results = [
            _make_result("too-big", throughput=100.0, ttft_ms=10.0, peak_vram_delta_gb=10.0),
            _make_result("fits", throughput=30.0, ttft_ms=200.0, peak_vram_delta_gb=2.0),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=6.0, vram_total=8.0), results)
        assert ranked[-1].model_name == "too-big"

    def test_cpu_only_hardware_skips_vram_gate(self):
        results = [
            _make_result("model-a", throughput=10.0, ttft_ms=500.0, peak_vram_delta_gb=8.0),
        ]
        advisor = ModelAdvisor()
        hw = _hw(vram_free=0.0, vram_total=0.0, has_gpu=False)
        ranked = _rank(advisor, hw, results)
        assert ranked[0].viable is True
        assert ranked[0].score > 0.0

    def test_empty_results_returns_empty_list(self):
        advisor = ModelAdvisor()
        report = advisor.rank(_hw(vram_free=6.0, vram_total=8.0), [])
        assert report.ranked == []

    def test_recommendation_str_viable_format(self):
        results = [
            _make_result("qwen", throughput=42.0, ttft_ms=150.0, peak_vram_delta_gb=2.1),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=6.0, vram_total=8.0), results)
        rec = ranked[0].recommendation_str
        assert "qwen" in rec
        assert "tok/s" in rec
        assert "GB VRAM" in rec

    def test_recommendation_str_not_viable_format(self):
        results = [
            _make_result("llama-big", throughput=80.0, ttft_ms=50.0, peak_vram_delta_gb=9.0),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=6.0, vram_total=8.0), results)
        rec = ranked[0].recommendation_str
        assert "Skip" in rec or "skip" in rec or "requires" in rec

    def test_24gb_hardware_accepts_large_model(self):
        results = [
            _make_result("llama-8b", throughput=25.0, ttft_ms=200.0, peak_vram_delta_gb=7.5),
            _make_result("qwen-small", throughput=50.0, ttft_ms=80.0, peak_vram_delta_gb=1.2),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=20.0, vram_total=24.0), results)
        assert all(r.viable is True for r in ranked)

    def test_6gb_hardware_rejects_large_model(self):
        results = [
            _make_result("llama-8b", throughput=25.0, ttft_ms=200.0, peak_vram_delta_gb=7.5),
            _make_result("qwen-small", throughput=50.0, ttft_ms=80.0, peak_vram_delta_gb=1.2),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=4.0, vram_total=6.0), results)
        llama = next(r for r in ranked if r.model_name == "llama-8b")
        qwen = next(r for r in ranked if r.model_name == "qwen-small")
        assert llama.viable is False
        assert qwen.viable is True

    def test_scores_are_normalized_0_to_100(self):
        results = [
            _make_result("model-a", throughput=40.0, ttft_ms=100.0, peak_vram_delta_gb=1.0),
            _make_result("model-b", throughput=20.0, ttft_ms=300.0, peak_vram_delta_gb=2.0),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=6.0, vram_total=8.0), results)
        for r in ranked:
            assert 0.0 <= r.score <= 100.0

    def test_warm_ttft_skips_cold_start_prompt(self):
        cold = PromptResult(
            prompt_label="q1",
            tokens_generated=20,
            ttft_ms=1456.0,
            total_latency_ms=2000.0,
            tokens_per_sec=10.0,
        )
        warm = PromptResult(
            prompt_label="q2",
            tokens_generated=20,
            ttft_ms=13.0,
            total_latency_ms=800.0,
            tokens_per_sec=40.0,
        )
        results = [
            BenchmarkResult(
                model_name="qwen",
                suite_version="test",
                timestamp="2026-06-08T12:00:00+00:00",
                mean_throughput_tps=40.0,
                peak_vram_delta_gb=1.0,
                prompt_results=[cold, warm],
            ),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=6.0, vram_total=8.0), results)
        assert ranked[0].ttft_ms == 13.0


class TestVramSafetyBuffer:
    def test_marginal_footprint_not_viable_with_buffer(self):
        """5.6 GB footprint + 0.5 GB buffer = 6.1 GB > 6.0 GB total — must not pass gate."""
        results = [
            _make_result("marginal", throughput=40.0, ttft_ms=100.0, peak_vram_delta_gb=5.6),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=3.3, vram_total=6.0), results)
        assert ranked[0].viable is False
        assert ranked[0].score == 0.0
        assert "5.60 GB + 0.5 GB buffer = 6.10 GB" in ranked[0].recommendation_str

    def test_footprint_within_total_is_viable(self):
        """5.38 GB footprint + 0.5 GB buffer = 5.88 GB <= 6.0 GB total — must pass gate."""
        results = [
            _make_result("qwen", throughput=40.0, ttft_ms=13.0, peak_vram_delta_gb=5.38),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=0.5, vram_total=6.0), results)
        assert ranked[0].viable is True
        assert ranked[0].score > 0.0


class TestModelAdvisorHardware:
    def test_skips_mismatched_hardware(self, make_hardware):
        desktop = make_hardware(gpu_name="RTX 4090", vram_total_gb=24.0, vram_free_gb=20.0)
        laptop = make_hardware(gpu_name="RTX 3060 Laptop GPU", vram_total_gb=6.0, vram_free_gb=4.9)
        results = [
            _make_result(
                "big-model",
                throughput=50.0,
                ttft_ms=100.0,
                peak_vram_delta_gb=2.0,
                hardware=desktop,
            ),
        ]
        advisor = ModelAdvisor()
        report = advisor.rank(laptop, results)
        assert report.ranked == []
        assert any("Skipped big-model" in w for w in report.warnings)

    def test_accepts_matching_hardware(self, make_hardware):
        hw = make_hardware()
        results = [
            _make_result(
                "small-model",
                throughput=30.0,
                ttft_ms=150.0,
                peak_vram_delta_gb=1.5,
                hardware=hw,
            ),
        ]
        advisor = ModelAdvisor()
        report = advisor.rank(hw, results)
        assert len(report.ranked) == 1
        assert report.ranked[0].model_name == "small-model"
        assert not any("Skipped" in w for w in report.warnings)

    def test_rejects_substring_gpu_name_match(self, make_hardware):
        """RTX 3060 benchmark must not match RTX 3060 Ti (different GPU)."""
        rtx3060 = make_hardware(gpu_name="RTX 3060", vram_total_gb=6.0, vram_free_gb=4.9)
        rtx3060ti = make_hardware(gpu_name="RTX 3060 Ti", vram_total_gb=8.0, vram_free_gb=6.0)
        results = [
            _make_result(
                "cached-model",
                throughput=40.0,
                ttft_ms=100.0,
                peak_vram_delta_gb=2.0,
                hardware=rtx3060,
            ),
        ]
        advisor = ModelAdvisor()
        report = advisor.rank(rtx3060ti, results)
        assert report.ranked == []
        assert any("Skipped cached-model" in w for w in report.warnings)

    def test_legacy_result_without_hardware_warns_but_ranks(self):
        results = [
            _make_result("legacy-model", throughput=25.0, ttft_ms=120.0, peak_vram_delta_gb=1.0),
        ]
        advisor = ModelAdvisor()
        report = advisor.rank(_hw(vram_free=6.0, vram_total=8.0), results)
        assert len(report.ranked) == 1
        assert any("No hardware recorded" in w for w in report.warnings)


class TestAdvisorMaxModelLenWarning:
    def test_warns_on_mismatch(self, make_hardware):
        hw = make_hardware()
        results = [
            BenchmarkResult(
                model_name="qwen2.5-0.5b",
                suite_version="test",
                timestamp="2026-06-08T12:00:00+00:00",
                mean_throughput_tps=40.0,
                peak_vram_delta_gb=1.5,
                hardware=hw,
                max_model_len=4096,
                prompt_results=[
                    PromptResult(
                        prompt_label="q1",
                        tokens_generated=20,
                        ttft_ms=100.0,
                        total_latency_ms=800.0,
                        tokens_per_sec=40.0,
                    )
                ],
            ),
        ]
        advisor = ModelAdvisor()
        report = advisor.rank(
            hw,
            results,
            model_max_lens={"qwen2.5-0.5b": 8192},
        )
        assert any("max_model_len=4096" in w and "8192" in w for w in report.warnings)
        assert len(report.ranked) == 1

    def test_no_warning_when_matching(self, make_hardware):
        hw = make_hardware()
        results = [
            BenchmarkResult(
                model_name="qwen2.5-0.5b",
                suite_version="test",
                timestamp="2026-06-08T12:00:00+00:00",
                mean_throughput_tps=40.0,
                peak_vram_delta_gb=1.5,
                hardware=hw,
                max_model_len=8192,
            ),
        ]
        advisor = ModelAdvisor()
        report = advisor.rank(
            hw,
            results,
            model_max_lens={"qwen2.5-0.5b": 8192},
        )
        assert not any("max_model_len" in w for w in report.warnings)
